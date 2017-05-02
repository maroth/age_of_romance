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

local current_minibatch = {}
local next_minibatch = {}


function train(neural_network, criterion, params, train_frame_dir, validate_frame_dir) 

    weights, weight_gradients = neural_network:getParameters()

    set_log_level(params.log_level)

    local frame_size = get_frame_size(train_frame_dir, params)
    local frame_files, frame_films = build_frame_set(train_frame_dir, params.max_frames_per_directory, params.number_of_bins)
    current_minibatch, next_minibatch = create_minibatch_storage(params.minibatch_size, frame_size, params.number_of_bins)

    sanity_check(neural_network, criterion, frame_size, params)
    local validate_files, validate_films = build_frame_set(validate_frame_dir, params.max_validate_frames_per_directory, params.number_of_bins)

    current_minibatch.frames = current_minibatch.frames:cuda()

    log(10, "Starting training on data in directory " .. train_frame_dir)
    log(7, "Frame size: " ..  frame_size[1] .. " x " .. frame_size[2] .. " x " .. frame_size[3])
    log(10, "Number of training frames: " ..  #frame_files)
    log(10, "Number of validation frames: " ..  #validate_files)

    local logger = optim.Logger('training_error.log')
    logger:setlogscale()
    logger:setNames{'Training error', 'Validation error'}
    logger:style{'+-', '+-'}
    logger:display(false)

    local pool = threads.Threads(1, function(thread_id) end)
    local starting_time = os.time()
    local number_of_train_minibatches = get_number_of_minibatches(#frame_files, params.minibatch_size)
    local number_of_validate_minibatches = get_number_of_minibatches(#validate_files, params.minibatch_size)
    for epoch_index = 1, params.epochs do
        current_minibatch.index = 1
        next_minibatch.index = 2

        local train_err = train_epoch(neural_network, criterion, params, frame_files, frame_films, frame_size, pool, starting_time, epoch_index, number_of_train_minibatches)

        local validate_err = validate(neural_network, criterion, params, validate_files, validate_films, frame_size, pool, number_of_validate_minibatches)

        log(9, epoch_summary(epoch_index, params.epochs, train_err, validate_err, params.minibatch_size, starting_time))

        logger:add{train_err, validate_err}
        logger:plot()

        if epoch_index % params.save_frequency == 0 then
            torch.save("models/" .. params.model_filename .. epoch_index .. ".model", neural_network)
        end
    end

    logger:plot()

    return neural_network
end

function train_epoch(neural_network, criterion, params, frame_files, frame_films, frame_size, pool, starting_time, epoch_index, number_of_train_minibatches)

    local shuffled_data = shuffle_data(frame_files, frame_films, number_of_train_minibatches)

    load_minibatch(params, frame_size, current_minibatch, shuffled_data)

    local err_sum = 0
    for minibatch_index = 1, number_of_train_minibatches - 1 do

        pool:addjob(function()
            local image = require 'image'
            require 'helpers'
            set_log_level(params.log_level)
            load_minibatch(params, frame_size, next_minibatch, shuffled_data)
        end)

        local err = train_minibatch(neural_network, criterion, params, current_minibatch, epoch_index, number_of_train_minibatches, starting_time)
        err_sum = err_sum + err

        pool:synchronize()

        current_minibatch.frames = torch.CudaTensor(next_minibatch.frames:size()):copy(next_minibatch.frames:cuda())
        current_minibatch.bins = torch.LongTensor(next_minibatch.bins):copy(next_minibatch.bins)

        current_minibatch.index = next_minibatch.index 
        next_minibatch.index = next_minibatch.index + 1
    end

    err_sum = err_sum + train_minibatch(neural_network, criterion, params, current_minibatch, epoch_index, number_of_train_minibatches, starting_time)

    return err_sum

end

function validate(neural_network, criterion, params, validate_files, validate_films, frame_size, pool, number_of_validate_minibatches)

    local data = {files =  validate_files, films =  validate_films}
    load_minibatch(params, frame_size, current_minibatch, data)

    local err_sum = 0
    for minibatch_index = 1, number_of_validate_minibatches - 1 do

        pool:addjob(function()
            local image = require 'image'
            require 'helpers'
            set_log_level(params.log_level)
            load_minibatch(params, frame_size, next_minibatch, data, current_minibatch)
        end)

        local err = validate_minibatch(neural_network, criterion, params, current_minibatch)
        err_sum = err_sum + err

        pool:synchronize()

        current_minibatch.frames = torch.CudaTensor(next_minibatch.frames:size()):copy(next_minibatch.frames:cuda())
        current_minibatch.bins = torch.LongTensor(next_minibatch.bins):copy(next_minibatch.bins)

        current_minibatch.index = next_minibatch.index 
        next_minibatch.index = next_minibatch.index + 1
    end

    err_sum = err_sum + validate_minibatch(neural_network, criterion, params, current_minibatch)

    return err_sum
end


function train_minibatch(neural_network, criterion, params, minibatch, epoch_index, number_of_minibatches, starting_time)

    function feval(new_weights)
        -- copy new weights, not sure if this is necessary
        if new_weights ~= weights then
            weights:copy(new_weights)
        end

        -- reset weight gradients
        weight_gradients:zero()

        -- forward the minibatch through the network, getting the prediction
        local prediction = neural_network:forward(minibatch.frames)
        if (prediction:size(1) > 1) then
            local_prediction = prediction[1][1]
        else
            local_prediction = prediction[1]
        end
        --log(2, "Fed minibatch " .. minibatch.index .. " into network, prediction is " .. local_prediction)

        -- forward the prediction through the criterion to the the error
        -- local err = criterion:forward(prediction, minibatch.dates)
        local err = criterion:forward(prediction, minibatch.bins)
        --log(2, "Forwarded prediction for minibatch " .. minibatch.index .. " into criterion, error is " .. err)

        -- calculate the gradient error by feeding the prediction 
        -- and the ground truth backwards through the criterion
        -- local grad_criterion = criterion:backward(prediction, minibatch.dates)
        local grad_criterion = criterion:backward(prediction, minibatch.bins)
        --log(2, "Backwarded prediction for minibatch " .. minibatch.index .. " into criterion, mean grad_criterion is " .. grad_criterion:mean())

        -- feed the gradients backward through the network
        neural_network:backward(minibatch.frames, grad_criterion)

        log(7, minibatch_summary(minibatch.index, number_of_minibatches, epoch_index, params.epochs, starting_time, err))
        
        return err, weight_gradients
    end

    local new_weights, err = optim.sgd(feval, weights, params)
    
    return err[1]
end

function validate_minibatch(neural_network, criterion, params, minibatch)
    local prediction = neural_network:forward(minibatch.frames)
    if (prediction:size(1) > 1) then
        local_prediction = prediction[1][1]
    else
        local_prediction = prediction[1]
    end
    local err = criterion:forward(prediction, minibatch.bins)
    return err
end

function get_number_of_minibatches(frame_count, minibatch_size)
    local number_of_minibatches = math.floor(frame_count / minibatch_size)
    if number_of_minibatches == 0 then
        print("ERROR: frame count smaller than minibatch size")
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
    --log(1, "shuffled inputs")
    return shuffled_data
end

function create_minibatch_storage(minibatch_size, frame_size, number_of_bins)
    local current_minibatch = {
        frames = torch.DoubleTensor(minibatch_size, frame_size[1], frame_size[2], frame_size[3]),
        bins = torch.LongTensor(minibatch_size),
        index = 1
    }

    local next_minibatch = {
        frames = torch.DoubleTensor(minibatch_size, frame_size[1], frame_size[2], frame_size[3]),
        bins = torch.LongTensor(minibatch_size),
        index = 2
    }
    return current_minibatch, next_minibatch
end
