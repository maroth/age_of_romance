require 'image'
require 'nn'
require 'lfs'

local tds = require 'tds'
local threads = require 'threads'
require 'date_logic'
require 'helpers'

train_frame_dir = "/mnt/e/age_of_romance/mini_frames/"
test_frame_dir = "/mnt/e/age_of_romance/mini_frames/"

local learning_rate = 0.001
local minibatch_size = 5
local epochs = 5

local frame_size = torch.LongStorage(3)
frame_size[1] = 3
frame_size[2] = 189
frame_size[3] = 320

local input_size = frame_size[1] * frame_size[2] * frame_size[3]

local current_minibatch_frames = tds.Hash()
local current_minibatch_dates = tds.Hash()

threads.Threads.serialization('threads.sharedserialize')

function build_frame_set(frame_dir)
    local index = 0
    frame_files = {}
    frame_films = {}
    for film_dir in lfs.dir(frame_dir) do    
        local info_file_path = frame_dir .. "/" .. film_dir .. "/info.json"
        if file_exists(info_file_path) then
            local film  = parse_info_file(info_file_path)
            film.normalized_date = normalize_date(parse_date(film.date))
            for frame_file in lfs.dir(frame_dir .. "/" .. film_dir) do
                if (string.ends(frame_file, ".png")) then
                    index = index + 1
                    update_output("loading frame : " .. index)
                    frame_files[index] = frame_dir .. film_dir .. "/" .. frame_file
                    frame_films[index] = film
                end
            end
        end
    end
    print("")
    return frame_files, frame_films
end

function build_shuffled_frame_set(films)
    local number_of_frames = get_films_frame_count(films)
    local randomized_indexes = torch.randperm(number_of_frames):long()

    local frame_size = get_frame_size(films)
    frames = torch.DoubleTensor(number_of_frames, frame_size[1], frame_size[2], frame_size[3])
    dates = torch.DoubleTensor(number_of_frames)
    index = 0
    for film_index, film in ipairs(films) do
        for frame_index, frame in ipairs(film.frames) do
            index = index + 1
            frames[index] = frame
            dates[index] = film.normalized_date
        end
    end

    shuffled_frames = frames:index(1, randomized_indexes)
    shuffled_dates = dates:index(1, randomized_indexes)
    return shuffled_frames, shuffled_dates
end


function start_asynch_loading_thread(frame_files, frame_films, minibatch_size, load_data_mutex_id, train_data_mutex_id, thread_pool) 
    local load_mutex_id = load_data_mutex_id
    local train_mutex_id = train_data_mutex_id
    thread_pool:addjob(
        function()
            local threads = require 'threads'
            local torch = require 'torch'
            local image = require 'image'

            local load_data_mutex = threads.Mutex(load_mutex_id)
            local train_data_mutex = threads.Mutex(train_mutex_id)

            local number_of_frames = 0
            for _, frame in ipairs(frame_files) do
                number_of_frames = number_of_frames + 1
            end

            local number_of_minibatches = math.floor(number_of_frames / minibatch_size)

            local randomized_indexes = torch.randperm(number_of_frames):long()

            local shuffled_frame_files = {}
            local shuffled_frame_films = {}
            for index = 1, number_of_frames do
                shuffled_frame_files[index] = frame_files[randomized_indexes[index]]
                shuffled_frame_films[index] = frame_films[randomized_indexes[index]]
            end

            for minibatch_index = 1, number_of_minibatches do
                train_data_mutex:lock()
                for frame_index = 1, minibatch_size do
                    local frame = image.load(shuffled_frame_files[minibatch_index + frame_index], 3, 'double')
                    current_minibatch_frames[frame_index] = frame

                    local film = shuffled_frame_films[minibatch_index + frame_index]
                    current_minibatch_dates[frame_index] = film.normalized_date
                end
                load_data_mutex:unlock()
                collectgarbage()
                collectgarbage()
            end

        end
    )
end

function train_minibatch(neural_network, criterion, learning_rate, load_data_mutex, train_data_mutex) 
    local number_of_frames = 0
    local err_sum = 0
    for _, frame in ipairs(frame_files) do
        number_of_frames = number_of_frames + 1
    end

    local number_of_minibatches = math.floor(number_of_frames / minibatch_size)

    for minibatch_index = 1, number_of_minibatches do
        update_output("minibatch: " .. minibatch_index)
        load_data_mutex:lock()

        local frame_size = current_minibatch_frames[1]:size()
        local input_size = frame_size[1] * frame_size[2] * frame_size[3]
    
        for frame_index = 1, minibatch_size do
            local frame = current_minibatch_frames[frame_index]
            local date = current_minibatch_dates[frame_index]
            local reshaped_frame = torch.reshape(frame, input_size)
            local prediction = neural_network:forward(reshaped_frame)
            neural_network:zeroGradParameters()
            local err = criterion:forward(prediction, date)
            err_sum = err_sum + err
            local grad_criterion = criterion:backward(prediction, date)
            neural_network:backward(reshaped_frame, grad_criterion)
            neural_network:updateParameters(learning_rate)
        end
        train_data_mutex:unlock()
    end
    print("")
    print ("error: ", err_sum / number_of_minibatches * minibatch_size)
end

function train_data(frame_dir) 
    local adaptive_learning_rate = learning_rate

    local neural_network = nn.Sequential()
    neural_network:add(nn.Linear(input_size, 1))
    neural_network:add(nn.Sigmoid())
    local criterion = nn.MSECriterion()
    frame_files, frame_films = build_frame_set(frame_dir)

    local load_data_thread_pool = threads.Threads(1, function(thread_id) end)
    for epoch_index = 1, epochs do
        local load_data_mutex = threads.Mutex()
        local train_data_mutex = threads.Mutex()
        local load_data_mutex_id = load_data_mutex:id()
        local train_data_mutex_id = train_data_mutex:id()
        load_data_mutex:lock()

        print("epoch " .. epoch_index)
        start_asynch_loading_thread(frame_files, frame_films, minibatch_size, load_data_mutex_id, train_data_mutex_id, load_data_thread_pool)
        train_minibatch(neural_network, criterion, learning_rate, load_data_mutex, train_data_mutex, load_data_mutex, train_data_mutex)

        adaptive_learning_rate = learning_rate * 0.75

        load_data_thread_pool:synchronize()
        load_data_mutex:free()
        train_data_mutex:free()
    end

    load_data_thread_pool:terminate()
    return neural_network
end

function test_data(neural_network, frame_dir)
    frame_files, frame_films = build_frame_set(frame_dir)

    local number_of_frames = 0
    for _, frame in ipairs(frame_files) do
        number_of_frames = number_of_frames + 1
    end

    for i = 1, number_of_frames do
        local frame = image.load(frame_files[i], 3, 'double')
        local film = frame_films[i]
        local reshaped_frame = torch.reshape(frame, input_size)
        local prediction = neural_network:forward(reshaped_frame)
        local denormalized_prediction = denormalize_date(prediction)        
        print("")
        print(film.title)
        print("actual date: " .. film.date, "predicted date: " .. denormalized_prediction)
    end
end


local neural_network = train_data(train_frame_dir)

test_data(neural_network, test_frame_dir)
