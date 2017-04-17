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

local sgd_state = {}

-- initialize cross-thread variables for minibatch exchange
local cross_thread_minibatch_frames = tds.Hash()
local cross_thread_minibatch_dates = tds.Hash()

-- load the neural network
local neural_network, criterion = build_neural_network()
local weights, weight_gradients = neural_network:getParameters()

-- logger for accuracy loggin
local logger = optim.Logger('training_error.log')
logger:setlogscale()
logger:setNames{'Training error'}
logger:style{'+-'}

-- remember starting time so we can estimate time till completion during training
local starting_time = os.time()

function train_data(params, train_frame_dir) 
    set_log_level(params.log_level)
    log(5, "Starting training on data in directory " .. train_frame_dir)

    local frame_size = get_frame_size(train_frame_dir)
    log(5, "Frame size: " ..  frame_size[1] .. " x " .. frame_size[2] .. " x " .. frame_size[3])

    log(10, "Testing network size compatibility...")
    local test_network, test_criterion  = build_neural_network()
    sanity_check(test_network, test_criterion, frame_size)
    log(10, "Neural network test success!")

    local frame_files, frame_films = build_frame_set(train_frame_dir, params.max_frames_per_directory)
    log(5, "Number of frames: " ..  #frame_files)

    local load_data_thread_pool = threads.Threads(1, function(thread_id) end)

    for epoch_index = 1, params.epochs do
        log(3, "\nStarting epoch " .. epoch_index)

        local load_data_mutex = threads.Mutex()
        local train_data_mutex = threads.Mutex()
        local load_data_mutex_id = load_data_mutex:id()
        local train_data_mutex_id = train_data_mutex:id()
        log(1, "Main thread: waiting for load mutex")
        load_data_mutex:lock()
        log(1, "Main thread: locked load mutex (initial lock)")

        load_images_async(params, frame_files, frame_films, load_data_mutex_id, train_data_mutex_id, load_data_thread_pool)
        local train_err = train_epoch(params, load_data_mutex, train_data_mutex, epoch_index)

        load_data_thread_pool:synchronize()
        load_data_mutex:free()
        train_data_mutex:free()

    end

    load_data_thread_pool:terminate()
    return neural_network
end


-- train a complete epoch, while asynchronously loading image data from the loading thread
function train_epoch(params, load_data_mutex, train_data_mutex, epoch_index) 

    local number_of_frames = count_frames(frame_files)
    local number_of_minibatches = math.floor(number_of_frames / params.minibatch_size)
    if number_of_minibatches == 0 then
        log(10, "ERROR: frame count smaller than minibatch size")
        os.exit()
    end

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

        collectgarbage()
        collectgarbage()

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

        local new_weights, err = optim.sgd(feval, weights, params, sgd_state)
        
        -- accumulate error for logging purposes
        err_sum = err_sum + err[1]

        log(7, minibatch_summary(minibatch_index, number_of_minibatches, epoch_index, params.epochs, starting_time, err_sum))
        log(5, minibatch_detail(params.minibatch_size, prediction, minibatch_dates, err[1]))
    end

    logger:add{err_sum}
    logger:plot()

    log(10, epoch_summary(epoch_index, params.epochs, err_sum, params.minibatch_size, starting_time))
    return err_sum
end


function test_data(params, test_frame_dir)
    log(3, "\n\nTesting data with files from " .. test_frame_dir)
    local films = load_films(test_frame_dir, params.max_frames_per_directory)
    log(3, "Number of test films: " ..  #films)

    local sum_mean_error = 0
    local sum_median_error = 0

    local film_logger = optim.Logger('results.log')
    film_logger:setNames{'Truth', 'Mean Prediction', 'Median Prediction'}
    film_logger:style{'+-', '+-', '+-', '+-'}

    for _, film in pairs(films) do
        local sum_prediction = 0
        local frame_count = 0
        local predictions = {}
        for frame_index, frame_dir in pairs(film.frames) do
            local frame = image.load(frame_dir, 3, 'double')
            local minibatch = torch.DoubleTensor(1, frame:size(1), frame:size(2), frame:size(3))
            minibatch[1] = frame
            log(1, "feeding frame " .. frame_dir .. " to test network")
            local prediction = neural_network:forward(minibatch)
            log(1, "prediction: " .. prediction[1])
            sum_prediction = sum_prediction + prediction[1]
            local err = math.abs((prediction[1] - film.normalized_date)[1])
            frame_count = frame_count + 1
            table.insert(predictions, prediction[1])
        end

        local mean_prediction = sum_prediction / frame_count
        local median_prediction = median(predictions)

        local denormalized_mean = denormalize_date(mean_prediction)
        local denormalized_median = denormalize_date(median_prediction)

        local film_mean_error = math.abs((mean_prediction - film.normalized_date)[1])
        local film_median_error = math.abs((median_prediction - film.normalized_date)[1])
        
        sum_mean_error = sum_mean_error + film_mean_error
        sum_median_error = sum_median_error + film_median_error

        film_logger:add{film.normalized_date[1], mean_prediction, median_prediction}

        log(3, "")
        log(3, film.title)
        log(3, "actual date: " .. film.date ..  "\tmean prediction: " .. denormalized_mean .. "\tmedian prediction: " .. denormalized_median)
        film_logger:plot()
    end

    local mean_error = sum_mean_error / #films
    local median_error = sum_median_error / #films
    log(8, "mean of mean error on test set: " .. mean_error)
    log(8, "mean of median error on test set: " .. median_error)


    return mean_error, median_error
end

-- start a second thread each epoch that loads the image data for the current minibatch while it is being trained on
function load_images_async(params, frame_files, frame_films, load_data_mutex_id, train_data_mutex_id, thread_pool) 

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

            set_log_level(9)

            log(3, "Load thread started")

            -- Mutex objects are not shared between threads, so we reconstruct them with their IDs
            local load_data_mutex = threads.Mutex(load_data_mutex_id)
            local train_data_mutex = threads.Mutex(train_data_mutex_id)
            log(1, "Load thread: created mutexes")

            -- calculate metrics
            local number_of_frames = count_frames(frame_files)
            local number_of_minibatches = math.floor(number_of_frames / params.minibatch_size)

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
            local minibatch_frames = torch.DoubleTensor(params.minibatch_size, example_frame:size(1), example_frame:size(2), example_frame:size(3))
            local minibatch_dates = torch.DoubleTensor(params.minibatch_size)

            -- iterate over current epoch
            for minibatch_index = 1, number_of_minibatches do
                log(3, "Load thread: starting loading for minibatch " .. minibatch_index)

                -- iterate over input data and fill minibatch tensor
                for intra_minibatch_index = 1, params.minibatch_size do
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
                collectgarbage()
            end
        end
    )
end

