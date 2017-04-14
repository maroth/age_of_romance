require 'image'
require 'nn'
require 'lfs'
require 'date_logic'
require 'load_logic'
require 'helpers'
require 'neural_network'

local tds = require 'tds'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

train_frame_dir = "/mnt/e/age_of_romance/sanity_check/"
test_frame_dir = "/mnt/e/age_of_romance/sanity_check/"

if arg[1] ~= nil then
    train_frame_dir = arg[1]
end

if arg[2] ~= nil then
    test_frame_dir = arg[2]
end

local learning_rate = 0.01
local minibatch_size = 2
local epochs = 5

local cross_thread_minibatch_frames = tds.Hash()
local cross_thread_minibatch_dates = tds.Hash()

local neural_network = build_neural_network()
local criterion = nn.MSECriterion()


local starting_time = os.time()

-- start a second thread each epoch that loads the image data for the current minibatch while it is being trained on
function load_images_async(frame_files, frame_films, load_data_mutex_id, train_data_mutex_id, thread_pool) 
    local example_frame = image.load(frame_files[1], 3, 'double')

    thread_pool:addjob(
        function()
            local threads = require 'threads'
            local torch = require 'torch'
            local image = require 'image'
            require 'helpers'

            local load_data_mutex = threads.Mutex(load_data_mutex_id)
            local train_data_mutex = threads.Mutex(train_data_mutex_id)

            local number_of_frames = count_frames(frame_files)
            local number_of_minibatches = math.floor(number_of_frames / minibatch_size)

            local randomized_indexes = torch.randperm(number_of_frames):long()
            local shuffled_frame_files = {}
            local shuffled_frame_films = {}

            for index = 1, number_of_frames do
                shuffled_frame_files[index] = frame_files[randomized_indexes[index]]
                shuffled_frame_films[index] = frame_films[randomized_indexes[index]]
            end

            local minibatch_frames = torch.DoubleTensor(minibatch_size, example_frame:size(1), example_frame:size(2), example_frame:size(3))
            local minibatch_dates = torch.DoubleTensor(minibatch_size)

            for minibatch_index = 1, number_of_minibatches do

                for frame_index = 1, minibatch_size do
                    local frame = image.load(shuffled_frame_files[minibatch_index + frame_index], 3, 'double')
                    local film = shuffled_frame_films[minibatch_index + frame_index]
                    minibatch_frames[frame_index] = frame
                    minibatch_dates[frame_index] = film.normalized_date
                end

                train_data_mutex:lock()
                cross_thread_minibatch_frames[1] = minibatch_frames
                cross_thread_minibatch_dates[1] = minibatch_dates
                load_data_mutex:unlock()
                collectgarbage()
            end
        end
    )
end

function train_epoch(learning_rate, load_data_mutex, train_data_mutex, epoch_index) 
    local number_of_frames = count_frames(frame_files)
    local number_of_minibatches = math.floor(number_of_frames / minibatch_size)

    local err_sum = 0
    for minibatch_index = 1, number_of_minibatches do

        load_data_mutex:lock()

        print("")
        print(print_status(minibatch_index, number_of_minibatches, epoch_index, epochs, starting_time, err_sum))

        local frame = cross_thread_minibatch_frames[1]
        local date = cross_thread_minibatch_dates[1]

        train_data_mutex:unlock()

        local prediction = neural_network:forward(frame)

        neural_network:zeroGradParameters()
        local err = criterion:forward(prediction, date)
        print("pred: ", string.format("%.3f", prediction[1][1]), "date: ", string.format("%.3f", date[1]), "err: ", string.format("%.3f", err))
        err_sum = err_sum + err

        local grad_criterion = criterion:backward(prediction, date)
        neural_network:backward(frame, grad_criterion)
        neural_network:updateParameters(learning_rate)
    end
    print("")
    print ("error: ", err_sum / number_of_minibatches * minibatch_size)
end

function train_data(frame_dir) 
    local adaptive_learning_rate = learning_rate

    local frame_size = get_frame_size(frame_dir)
    local input_size = frame_size[1] * frame_size[2] * frame_size[3]

    local frame_files, frame_films = build_frame_set(frame_dir)

    local load_data_thread_pool = threads.Threads(1, function(thread_id) end)

    for epoch_index = 1, epochs do

        local load_data_mutex = threads.Mutex()
        local train_data_mutex = threads.Mutex()
        local load_data_mutex_id = load_data_mutex:id()
        local train_data_mutex_id = train_data_mutex:id()
        load_data_mutex:lock()

        print("epoch " .. epoch_index)

        load_images_async(frame_files, frame_films, load_data_mutex_id, train_data_mutex_id, load_data_thread_pool)
        train_epoch(adaptive_learning_rate, load_data_mutex, train_data_mutex, epoch_index)

        load_data_thread_pool:synchronize()
        load_data_mutex:free()
        train_data_mutex:free()

        adaptive_learning_rate = adaptive_learning_rate * 0.75
    end

    load_data_thread_pool:terminate()
    return neural_network
end

function test_data(neural_network, frame_dir)
    frame_files, frame_films = build_frame_set(frame_dir)

    local number_of_frames = count_frames(frame_files)
    local frame_size = get_frame_size(frame_dir)
    local input_size = frame_size[1] * frame_size[2] * frame_size[3]

    local last_film = frame_films[1]
    local sum_prediction = 0
    local current_film_frame_count = 0
    for i = 1, number_of_frames do
        local frame = image.load(frame_files[i], 3, 'double')
        current_film_frame_count = current_film_frame_count + 1
        local film = frame_films[i]
        local prediction = neural_network:forward(frame)
        sum_prediction = sum_prediction + prediction
        if film ~= last_film then
            last_film = film
            local average_prediction = sum_prediction / current_film_frame_count
            local denormalized_prediction = denormalize_date(average_prediction)        
            print("")
            print(film.title)
            print("actual date: " .. film.date, "predicted date: " .. denormalized_prediction)
        end
    end
end


local neural_network = train_data(train_frame_dir)
test_data(neural_network, test_frame_dir)
