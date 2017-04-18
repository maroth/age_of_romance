require 'nn'
require 'optim'
require 'lfs'
require 'image'

require 'date_logic'
require 'load_logic'
require 'helpers'
require 'test_network'

local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

function train(neural_network, criterion, params, train_frame_dir) 
    set_log_level(params.log_level)

    local frame_size = get_frame_size(train_frame_dir)
    local frame_files, frame_films = build_frame_set(train_frame_dir, params.max_frames_per_directory)
    sanity_check(neural_network, criterion, frame_size, params)

    log(10, "Starting training on data in directory " .. train_frame_dir)
    log(7, "Frame size: " ..  frame_size[1] .. " x " .. frame_size[2] .. " x " .. frame_size[3])
    log(10, "Number of frames: " ..  #frame_files)

    local pool = threads.Threads(1, function(thread_id) end)
    local starting_time = os.time()
    for epoch_index = 1, params.epochs do
        train_epoch(neural_network, criterion, params, frame_files, frame_films, frame_size, pool, starting_time)
    end
end

function train_epoch(neural_network, criterion, params, frame_files, frame_films, frame_size, pool, starting_time)
    local shuffled_data = shuffle_data(frame_files, frame_films)
    local current_minibatch, next_minibatch = create_minibatch_storage(params.minibatch_size, frame_size)
    load_minibatch(params, frame_size, current_minibatch, shuffled_data)

    local err_sum = 0
    local number_of_minibatches = get_number_of_minibatches(#frame_files, params.minibatch_size)
    for minibatch_index = 1, number_of_minibatches - 1 do
        pool:addjob(function()
            local image = require 'image'
            require 'helpers'
            set_log_level(1)
            load_minibatch(params, frame_size, next_minibatch, shuffled_data)
        end)
        err_sum = err_sum + train_minibatch(neural_network, criterion, params, current_minibatch, frame_size)
        pool:synchronize()
        swap_current_and_next(current_minibatch, next_minibatch)
    end

    err_sum = err_sum + train_minibatch(neural_network, criterion, params, current_minibatch, frame_size)
end

function train_minibatch(neural_network, criterion, params, minibatch)
    local weights, weight_gradients = neural_network:getParameters()

    function feval(new_weights)
        -- copy new weights, not sure if this is necessary
        if new_weights ~= weights then
            weights:copy(new_weights)
        end

        -- reset weight gradients
        weight_gradients:zero()

        -- forward the minibatch through the network, getting the prediction
        local prediction = neural_network:forward(minibatch.frames)
        log(2, "Fed minibatch " .. minibatch.index .. " into network, prediction is " .. prediction[1][1])

        -- forward the prediction through the criterion to the the error
        print(minibatch.dates)
        local err = criterion:forward(prediction, minibatch.dates)
        log(2, "Forwarded prediction for minibatch " .. minibatch.index .. " into criterion, error is " .. err)

        -- calculate the gradient error by feeding the prediction 
        -- and the ground truth backwards through the criterion
        local grad_criterion = criterion:backward(prediction, minibatch.dates)
        log(2, "Backwarded prediction for minibatch " .. minibatch.index .. " into criterion, mean grad_criterion is " .. grad_criterion:mean())

        -- feed the gradients backward through the network
        neural_network:backward(minibatch.frames, grad_criterion)

        --log(7, minibatch_summary(minibatch.index, number_of_minibatches, epoch_index, params.epochs, starting_time, err_sum))
        --log(5, minibatch_detail(params.minibatch_size, prediction, minibatch_dates, err[1]))
        
        return err, weight_gradients
    end


    local new_weights, err = optim.sgd(feval, weights, params)
    
    -- accumulate error for logging purposes
    return err[1]
end

function get_number_of_minibatches(frame_count, minibatch_size)
    local number_of_minibatches = math.floor(#frame_files / minibatch_size)
    if number_of_minibatches == 0 then
        log(10, "ERROR: frame count smaller than minibatch size")
        os.exit()
    end
    return number_of_minibatches
end

function shuffle_data(frame_files, frame_films)
    local randomized_indexes = torch.randperm(#frame_files):long()
    local shuffled_data = {
        files = {},
        films = {}
    }
    for index = 1, #frame_files do
        shuffled_data.files[index] = frame_files[randomized_indexes[index]]
        shuffled_data.films[index] = frame_films[randomized_indexes[index]]
    end
    log(1, "shuffled inputs")
    return shuffled_data
end

function swap_current_and_next(current_minibatch, next_minibatch)
    current_minibatch.frames = torch.DoubleTensor(next_minibatch.frames:size()):copy(next_minibatch.frames)
    current_minibatch.dates = torch.DoubleTensor(next_minibatch.dates:size()):copy(next_minibatch.dates)
    current_minibatch.index = next_minibatch.index 

    next_minibatch.index = next_minibatch.index + 1
end

function create_minibatch_storage(minibatch_size, frame_size)
    local current_minibatch = {
        frames = torch.DoubleTensor(minibatch_size, frame_size[1], frame_size[2], frame_size[3]),
        dates = torch.DoubleTensor(minibatch_size),
        index = 1
    }

    local next_minibatch = {
        frames = torch.DoubleTensor(minibatch_size, frame_size[1], frame_size[2], frame_size[3]),
        dates = torch.DoubleTensor(minibatch_size),
        index = 2
    }
    return current_minibatch, next_minibatch
end
