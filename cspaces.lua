require 'nn'
require 'optim'
require 'lfs'
require 'image'

require 'date_logic'
require 'load_logic'
require 'helpers'
require 'test_network'


function train(neural_network, criterion, params, files)

    weights, weight_gradients = neural_network:getParameters()

    set_log_level(params.log_level)

    local frame_size = torch.LongStorage{3, 176}

    local train_frame_cspaces = torch.load(files.train_file_cspaces)
    local train_frame_bins = torch.load(files.train_file_normalized_dates):apply(function(x) return get_bin(x, params.number_of_bins) end)
    --local train_frame_bins = torch.load(files.train_file_ids)
    
    local validate_frame_cspaces = torch.load(files.validate_file_cspaces)
    local validate_frame_bins = torch.load(files.validate_file_normalized_dates):apply(function(x) return get_bin(x, params.number_of_bins) end)
    --local validate_frame_bins = torch.load(files.validate_file_ids)

    sanity_check(neural_network, criterion, frame_size, params)

    log(10, "Number of training frames: " ..  train_frame_cspaces:size(1))
    log(10, "Number of validation frames: " ..  validate_frame_cspaces:size(1))

    local logger = optim.Logger(params.name .. "-" .. 'training-error.log')
    logger:setlogscale()
    logger:setNames{'Training error', 'Validation error'}
    logger:style{'+-', '+-'}
    logger:display(false)

    local starting_time = os.time()

    local epoch_index = 1
    local last_validate_err = nil
    local validate_err = 0
    local validate_error_increasing = 0
    for epoch_index = 1, params.epochs do

        local train_err = train_epoch(neural_network, criterion, params, train_frame_cspaces, train_frame_bins, frame_size, pool, starting_time, epoch_index, number_of_train_minibatches)

        if epoch_index % params.save_frequency == 0 or epoch_index == 1 then
            torch.save("models/" .. params.name .. "_" .. epoch_index .. ".model", neural_network)
            validate_err = validate(neural_network, criterion, params, validate_frame_cspaces, validate_frame_bins, frame_size, pool, number_of_validate_minibatches)
        end
        
        logger:add{train_err, validate_err}
        log(9, epoch_summary(epoch_index, params.epochs, train_err, validate_err, params.minibatch_size, starting_time))
        logger:plot()
        
        if last_validate_err == nil then
            last_validate_err = validate_err
        elseif last_validate_err < validate_err then
            validate_error_increasing = validate_error_increasing + 1
            if validate_error_increasing > 5 then
                break
            end
        else 
            validate_error_increasing = 0
        end
        
        last_validate_err = validate_err
    end

    torch.save("models/" .. params.name .. "_" .. epoch_index .. ".model", neural_network)
    return neural_network
end

function train_epoch(neural_network, criterion, params, train_frame_cspaces, train_frame_bins, frame_size, pool, starting_time, epoch_index, number_of_train_minibatches)

    local err_sum = 0
    
    local bins_shuffled, cspaces_shuffled = shuffle_data(train_frame_cspaces, train_frame_bins)
     
    local total_frames = bins_shuffled:size(1) * params.epochs

    for i = 1, train_frame_cspaces:size(1), params.minibatch_size do

        function feval(new_weights)
        
            local bins = bins_shuffled:sub(i, math.min(i + params.minibatch_size - 1, train_frame_cspaces:size(1)))
            local cspaces = cspaces_shuffled:sub(i, math.min(i + params.minibatch_size - 1, train_frame_cspaces:size(1)))
            
            if new_weights ~= weights then
                weights:copy(new_weights)
            end

            weight_gradients:zero()
            local prediction = neural_network:forward(cspaces)       
            local err = criterion:forward(prediction, bins)
            local grad_criterion = criterion:backward(prediction, bins)
            neural_network:backward(cspaces, grad_criterion)          
            return err, weight_gradients           
        end

        local new_weights, err = optim.sgd(feval, weights, params)
        
        err_sum = err_sum + err[1]         
    end    

    return err_sum / (train_frame_cspaces:size(1) / params.minibatch_size)

end

function validate(neural_network, criterion, params, validate_frame_cspaces, validate_frame_bins, frame_size, pool, number_of_validate_minibatches)
    local err_sum = 0
    for i = 1, validate_frame_bins:size(1) do
        local prediction = neural_network:forward(validate_frame_cspaces[i])
        local err = criterion:forward(prediction, validate_frame_bins[i])
        err_sum = err_sum + err
    end
    return err_sum / validate_frame_bins:size(1)
end


function shuffle_data (cspaces, bins)
	local shuffle_indexes = torch.randperm(bins:size(1))
	local cspaces_shuffled = torch.CudaTensor(cspaces:size())
	local bins_shuffled = torch.CudaTensor(bins:size())
	for i = 1, bins:size(1), 1 do
		bins_shuffled[i] = bins[shuffle_indexes[i]]
		cspaces_shuffled[i] = cspaces[shuffle_indexes[i]]
	end
	return bins_shuffled, cspaces_shuffled
end
