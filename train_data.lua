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

    if (params.load_saved_model) then
        log(10, "loading model from " .. params.model_filename)
        neural_network = torch.load(params.model_filename)
    end

    local frame_size = get_frame_size(train_frame_dir)
    local frame_files, frame_films = build_frame_set(train_frame_dir, params.max_frames_per_directory, params.number_of_bins)
    sanity_check(neural_network, criterion, frame_size, params)

    log(10, "Starting training on data in directory " .. train_frame_dir)
    log(7, "Frame size: " ..  frame_size[1] .. " x " .. frame_size[2] .. " x " .. frame_size[3])
    log(10, "Number of frames: " ..  #frame_files)

    local logger = optim.Logger('training_error.log')
    logger:setlogscale()
    logger:setNames{'Training error'}
    logger:style{'+-'}
    logger:display(params.display_plot)

    local pool = threads.Threads(1, function(thread_id) end)
    local starting_time = os.time()
    for epoch_index = 1, params.epochs do
        train_epoch(neural_network, criterion, params, frame_files, frame_films, frame_size, pool, starting_time, epoch_index, logger)
    end

    logger:plot()

    return neural_network
end

function train_epoch(neural_network, criterion, params, frame_files, frame_films, frame_size, pool, starting_time, epoch_index, logger)
    local shuffled_data = shuffle_data(frame_files, frame_films)
    local current_minibatch, next_minibatch = create_minibatch_storage(params.minibatch_size, frame_size, params.number_of_bins)

    if (params.use_cuda) then
        current_minibatch.frames = current_minibatch.frames:cuda()
        current_minibatch.dates = current_minibatch.dates:cuda()
        current_minibatch.probability_vectors = current_minibatch.probability_vectors:cuda()
    end

    load_minibatch(params, frame_size, current_minibatch, shuffled_data)

    local err_sum = 0
    local number_of_minibatches = get_number_of_minibatches(#frame_files, params.minibatch_size)
    for minibatch_index = 1, number_of_minibatches - 1 do

        pool:addjob(function()
            local image = require 'image'
            require 'helpers'
            set_log_level(params.log_level)
            load_minibatch(params, frame_size, next_minibatch, shuffled_data)
        end)

        err = train_minibatch(neural_network, criterion, params, current_minibatch, epoch_index, number_of_minibatches, starting_time)
        err_sum = err_sum + err

        pool:synchronize()

        if (params.use_cuda) then
            current_minibatch.frames = torch.CudaTensor(next_minibatch.frames:size()):copy(next_minibatch.frames:cuda())
            current_minibatch.dates = torch.CudaTensor(next_minibatch.dates:size()):copy(next_minibatch.dates:cuda())
            current_minibatch.probability_vectors = torch.CudaTensor(next_minibatch.probability_vectors:size()):copy(next_minibatch.probability_vectors:cuda())

        else
            current_minibatch.frames = torch.DoubleTensor(next_minibatch.frames:size()):copy(next_minibatch.frames)
            current_minibatch.dates = torch.DoubleTensor(next_minibatch.dates:size()):copy(next_minibatch.dates)
            current_minibatch.probability_vectors = torch.DoubleTensor(next_minibatch.probability_vectors:size()):copy(next_minibatch.probability_vectors)
        end

        current_minibatch.bins = torch.LongTensor(next_minibatch.bins):copy(next_minibatch.bins)

        current_minibatch.index = next_minibatch.index 
        next_minibatch.index = next_minibatch.index + 1
    end

    err_sum = err_sum + train_minibatch(neural_network, criterion, params, current_minibatch, epoch_index, number_of_minibatches, starting_time)

    log(9, epoch_summary(epoch_index, params.epochs, err_sum, params.minibatch_size, starting_time))

    logger:add{err_sum}
    if (params.display_plot) then
        logger:plot()
    end

    torch.save(params.model_filename .. epoch_index .. ".model", neural_network)
end

function train_minibatch(neural_network, criterion, params, minibatch, epoch_index, number_of_minibatches, starting_time)
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
        if (prediction:size(1) > 1) then
            local_prediction = prediction[1][1]
        else
            local_prediction = prediction[1]
        end
        log(2, "Fed minibatch " .. minibatch.index .. " into network, prediction is " .. local_prediction)

        -- forward the prediction through the criterion to the the error
        -- local err = criterion:forward(prediction, minibatch.dates)
        local err = criterion:forward(prediction, minibatch.bins)
        log(2, "Forwarded prediction for minibatch " .. minibatch.index .. " into criterion, error is " .. err)

        -- calculate the gradient error by feeding the prediction 
        -- and the ground truth backwards through the criterion
        -- local grad_criterion = criterion:backward(prediction, minibatch.dates)
        local grad_criterion = criterion:backward(prediction, minibatch.bins)
        log(2, "Backwarded prediction for minibatch " .. minibatch.index .. " into criterion, mean grad_criterion is " .. grad_criterion:mean())

        -- feed the gradients backward through the network
        neural_network:backward(minibatch.frames, grad_criterion)

        log(7, minibatch_summary(minibatch.index, number_of_minibatches, epoch_index, params.epochs, starting_time, err))
        log(5, minibatch_detail(params.minibatch_size, prediction, minibatch.dates, err))
        
        return err, weight_gradients
    end


    local new_weights, err = optim.sgd(feval, weights, params)
    
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
end

function create_minibatch_storage(minibatch_size, frame_size, number_of_bins)
    local current_minibatch = {
        frames = torch.DoubleTensor(minibatch_size, frame_size[1], frame_size[2], frame_size[3]),
        dates = torch.DoubleTensor(minibatch_size),
        bins = torch.LongTensor(minibatch_size),
        probability_vectors = torch.DoubleTensor(minibatch_size, number_of_bins),
        index = 1
    }

    local next_minibatch = {
        frames = torch.DoubleTensor(minibatch_size, frame_size[1], frame_size[2], frame_size[3]),
        dates = torch.DoubleTensor(minibatch_size),
        bins = torch.LongTensor(minibatch_size),
        probability_vectors = torch.DoubleTensor(minibatch_size, number_of_bins),
        index = 2
    }
    return current_minibatch, next_minibatch
end
