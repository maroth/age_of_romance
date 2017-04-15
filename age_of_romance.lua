require 'nn'
require 'optim'
require 'lfs'
require 'image'

require 'date_logic'
require 'load_logic'
require 'helpers'
require 'neural_network'
require 'test_network'

local tds = require 'tds'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')


-- CONFIGURATION
train_frame_dir = "/mnt/e/age_of_romance/sanity_check/"
test_frame_dir = "/mnt/e/age_of_romance/sanity_check/"

-- command line argument 1 overrides training frame directory
if arg[1] ~= nil then
    train_frame_dir = arg[1]
end

-- command line argument 2 overrides testing frame directory
if arg[2] ~= nil then
    test_frame_dir = arg[2]
end

-- all images in a minibatch are fed into the network at the same time
-- optimize this so the network still fits into RAM
local minibatch_size = 20

-- number of total epochs
local epochs = 5

local sgd_params = {
    learningRate = 0.001,
    learningRateDecay = 0.0001,
    weightDecay = 0,
    dampening = 0,
    nesterov = false,
    momentum = 0.01
}

local sgd_state = {}

-- set the log theshold
-- messages with a higher or equal number than this are displayed
-- set to 1 or 2 for debugging purposes, 5 or so for actual training
set_log_level(8)

-- END CONFIGURATION

-- initialize cross-thread variables for minibatch exchange
local cross_thread_minibatch_frames = tds.Hash()
local cross_thread_minibatch_dates = tds.Hash()

-- load the neural network
local neural_network, criterion = build_neural_network()
local weights, weight_gradients = neural_network:getParameters()

-- logger for accuracy loggin
local logger = optim.Logger('accuracy.log')
logger:setNames{'Training accuracy', 'Testing accuracy'}
logger:style{'+-', '+-'}

-- remember starting time so we can estimate time till completion during training
local starting_time = os.time()

-- train a complete epoch, while asynchronously loading image data from the loading thread
function train_epoch(load_data_mutex, train_data_mutex, epoch_index) 

    local number_of_frames = count_frames(frame_files)
    local number_of_minibatches = math.floor(number_of_frames / minibatch_size)

    local err_sum = 0
    for minibatch_index = 1, number_of_minibatches do
        log(3, "\nStarting minibatch " .. minibatch_index)

        -- lock mutex so loading thread won't overwrite the next batch as we are reading it
        log(1, "Main thread: waiting for load mutex")
        load_data_mutex:lock()
        log(1, "Main thread: unlocked load mutex")

        -- load the next minibatch frames and true dates from the loading thread
        local minibatch = cross_thread_minibatch_frames[1]
        local minibatch_dates = cross_thread_minibatch_dates[1]

        -- unlock the train mutex so the loading thread can start loading the next minibatch
        train_data_mutex:unlock()
        log(1, "Main thread: locked train mutex")

        local prediction = {}
        local err = {}

        function feval(new_weights)
            -- copy new weights, not sure if this is necessary
            if new_weights ~= weights then
                weights:copy(new_weights)
            end

            -- reset weight gradients
            weight_gradients:zero()

            -- forward the minibatch through the network, getting the prediction
            prediction = neural_network:forward(minibatch)
            log(2, "Fed minibatch " .. minibatch_index .. " into network")

            -- forward the prediction through the criterion to the the error
            local err = criterion:forward(prediction, minibatch_dates)
            log(2, "Forwarded prediction for minibatch " .. minibatch_index .. " into criterion, error is " .. err)

            -- calculate the gradient error by feeding the prediction 
            -- and the ground truth backwards through the criterion
            local grad_criterion = criterion:backward(prediction, minibatch_dates)
            log(2, "Backwarded prediction for minibatch " .. minibatch_index .. " into criterion, mean grad_criterion is " .. grad_criterion:mean())

            -- feed the gradients backward through the network
            neural_network:backward(minibatch, grad_criterion)
            
            return err, weight_gradients
        end

        local new_weights, err = optim.sgd(feval, weights, sgd_params, sgd_state)
        
        -- accumulate error for logging purposes
        err_sum = err_sum + err[1]

        log(7, minibatch_summary(minibatch_index, number_of_minibatches, epoch_index, epochs, starting_time, err_sum))
        log(5, minibatch_detail(minibatch_size, prediction, minibatch_dates, err[1]))
    end

    logger:add{err_sum}
    logger:plot()
    log(10, epoch_summary(epoch_index, epochs, err_sum, minibatch_size, starting_time))
    return err_sum
end


function train_data(frame_dir) 
    log(5, "Starting training on data in directory " .. frame_dir)

    local frame_size = get_frame_size(frame_dir)
    log(5, "Frame size: " ..  frame_size[1] .. " x " .. frame_size[2] .. " x " .. frame_size[3])

    log(10, "Testing network size compatibility...")
    local test_network, test_criterion  = build_neural_network()

    sanity_check(test_network, test_criterion, frame_size)

    log(10, "Neural network test success!")

    local frame_files, frame_films = build_frame_set(frame_dir)
    log(5, "Number of frames: " ..  #frame_files)

    local load_data_thread_pool = threads.Threads(1, function(thread_id) end)

    for epoch_index = 1, epochs do
        log(3, "\nStarting epoch " .. epoch_index)

        local load_data_mutex = threads.Mutex()
        local train_data_mutex = threads.Mutex()
        local load_data_mutex_id = load_data_mutex:id()
        local train_data_mutex_id = train_data_mutex:id()
        log(1, "Main thread: waiting for load mutex")
        load_data_mutex:lock()
        log(1, "Main thread: locked load mutex (initial lock)")

        load_images_async(frame_files, frame_films, load_data_mutex_id, train_data_mutex_id, load_data_thread_pool)
        train_epoch(load_data_mutex, train_data_mutex, epoch_index)

        load_data_thread_pool:synchronize()
        load_data_mutex:free()
        train_data_mutex:free()

    end

    load_data_thread_pool:terminate()
    return neural_network
end

function test_data(neural_network, frame_dir)

    log(10, "\n\nTesting data with files from " .. frame_dir)
    frame_files, frame_films = build_frame_set(frame_dir)
    log(5, "Number of test frames: " ..  #frame_files)

    local number_of_frames = count_frames(frame_files)
    local frame_size = get_frame_size(frame_dir)

    local last_film = frame_films[1]
    local sum_prediction = 0
    local sum_err = 0
    local current_film_frame_count = 0
    local film_count = 0

    for i = 1, number_of_frames do
        -- load test frame from disk
        local frame = image.load(frame_files[i], 3, 'double')
        local film = frame_films[i]

        if film ~= last_film then
            film_count = film_count + 1
            local average_prediction = sum_prediction / current_film_frame_count
            local denormalized_prediction = denormalize_date(average_prediction)        
            print("")
            print(last_film.title)
            print("actual date: " .. last_film.date, "predicted date: " .. denormalized_prediction)
            sum_prediction = 0
            current_film_frame_count = 0
            sum_err = sum_err + math.abs(average_prediction[1] - last_film.normalized_date[1])
            last_film = film
        end

        log(1, "feeding frame " .. frame_files[i] .. " to test network")
        local prediction = neural_network:forward(frame)
        log(1, "prediction: " .. prediction[1])

        sum_prediction = sum_prediction + prediction
        current_film_frame_count = current_film_frame_count + 1
    end

    local total_avg_err = sum_err / film_count
    print ("\n\nTOTAL ERROR: " .. total_avg_err)
    print ("TOTAL DURATION: " .. os.time() - starting_time)
end

-- start a second thread each epoch that loads the image data for the current minibatch while it is being trained on
function load_images_async(frame_files, frame_films, load_data_mutex_id, train_data_mutex_id, thread_pool) 

    -- load the frist frame in the list to get the size of the tensor we need to create
    local example_frame = image.load(frame_files[1], 3, 'double')
    log(1, "Load thread: example frame loaded with size " .. example_frame:size(1) .. " x " .. example_frame:size(2) .. " x " .. example_frame:size(3))

    thread_pool:addjob(
        function()

            -- as this thread runs in a seperate context, we need to re-import the requirements
            local threads = require 'threads'
            local torch = require 'torch'
            local image = require 'image'
            require 'helpers'

            set_log_level(7)

            log(3, "Load thread started")

            -- Mutex objects are not shared between threads, so we reconstruct them with their IDs
            local load_data_mutex = threads.Mutex(load_data_mutex_id)
            local train_data_mutex = threads.Mutex(train_data_mutex_id)
            log(1, "Load thread: created mutexes")

            -- calculate metrics
            local number_of_frames = count_frames(frame_files)
            local number_of_minibatches = math.floor(number_of_frames / minibatch_size)

            -- shuffle the inputs according to a random permutation
            local randomized_indexes = torch.randperm(number_of_frames):long()
            local shuffled_frame_files = {}
            local shuffled_frame_films = {}
            for index = 1, number_of_frames do
                shuffled_frame_files[index] = frame_files[randomized_indexes[index]]
                shuffled_frame_films[index] = frame_films[randomized_indexes[index]]
            end
            log(1, "Load thread: shuffled inputs")

            -- create empty tensors for current minibatch
            local minibatch_frames = torch.DoubleTensor(minibatch_size, example_frame:size(1), example_frame:size(2), example_frame:size(3))
            local minibatch_dates = torch.DoubleTensor(minibatch_size)

            -- iterate over current epoch
            for minibatch_index = 1, number_of_minibatches do
                log(3, "Load thread: starting loading for minibatch " .. minibatch_index)

                -- iterate over input data and fill minibatch tensor
                for intra_minibatch_index = 1, minibatch_size do
                    local abs_index = minibatch_index + intra_minibatch_index - 1
                    log(1, "trying to load image to memory: " .. shuffled_frame_files[abs_index])
                    local frame = image.load(shuffled_frame_files[abs_index], 3, 'double')
                    local film = shuffled_frame_films[abs_index]
                    minibatch_frames[intra_minibatch_index] = frame
                    minibatch_dates[intra_minibatch_index] = film.normalized_date
                    log(1, "loaded frame with normalized date " .. film.normalized_date[1])
                end

                log(3, "Load thread: loading complete for minibatch " .. minibatch_index)

                -- lock train mutex so training thread can read read data until we are done writing it
                log(1, "Load thread: waiting for train mutex")
                train_data_mutex:lock()
                log(1, "Load thread: locked train mutex")

                -- write current minibatch to cross-thread sharing variables
                cross_thread_minibatch_frames[1] = torch.DoubleTensor(minibatch_frames:size()):copy(minibatch_frames)
                cross_thread_minibatch_dates[1] = torch.DoubleTensor(minibatch_dates:size()):copy(minibatch_dates)

                -- unlock data mutex so train thread can start working on this minibatch
                load_data_mutex:unlock()
                log(1, "Load thread: unlocked load mutex")

                -- garbage collection so our RAM usage stays low
                collectgarbage()
            end
        end
    )
end

-- start training the network
local neural_network = train_data(train_frame_dir)

-- test the network
test_data(neural_network, test_frame_dir)
