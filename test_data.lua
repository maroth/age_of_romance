require 'nn'
require 'image'

require 'date_logic'
require 'load_logic'
require 'helpers'
require 'optim'

function test(neural_network, criterion, params, files)
    print(train_d, train_l)
    local test_predictions, test_pm5, test_count = test_set(neural_network, criterion, params, files.test_file_cspaces, files.test_file_ids)
    local validate_predictions, validate_pm5, validate_count = test_set(neural_network, criterion, params, files.validate_file_cspaces, files.validate_file_ids)
    local train_predictions, train_pm5, train_count  = test_set(neural_network, criterion, params, files.train_file_cspaces, files.train_file_ids)

    logger = optim.Logger(params.name .. "-test.log")
    logger:setNames{'Test', 'Validate', 'Train'}
    logger:style{'+-', '+-', '+-'}
    logger:display(false)
    for i = 1, params.number_of_bins do
        local test_accuracy = test_predictions[i] / test_count
        local validate_accuracy = validate_predictions[i] / validate_count
        local train_accuracy = train_predictions[i] / train_count
        log(8, "\ntest top " .. i .. " accuracy: " .. test_accuracy)
        log(8, "\nvalidate top " .. i .. " accuracy: " .. validate_accuracy)
        log(8, "\ntrain top " .. i .. " accuracy: " .. train_accuracy)
        logger:add{test_accuracy, validate_accuracy, train_accuracy}
    end

    print("Test set +-5 correct: ", test_pm5 / test_count)
    print("Validate set +-5 correct: ", validate_pm5 / validate_count)
    print("Train set +-5 correct: ", train_pm5 / train_count)

    logger:plot()
end

function test_set(neural_network, criterion, params, data_file, labels_file) 

    local data  = torch.load(data_file)
    local labels = torch.load(labels_file)
    set_log_level(params.log_level)

    local correct_predictions = {}
    for i = 1, params.number_of_bins do
        correct_predictions[i] = 0
    end

    local total_predictions = 0
    local plus_minus_n_correct_predictions = 0

    local starting_time = os.time()
    local fraction_done = 0

    for frame_index = 1, labels:size(1), 10 do

        local prediction = neural_network:forward(data[frame_index])
       
        v, top_index = prediction:topk(1, true, true)
        if math.abs(top_index - labels[frame_index]) <= 5 then
            plus_minus_n_correct_predictions  = plus_minus_n_correct_predictions + 1
        end

        for bin_index = 1, params.number_of_bins do
            values, indexes = prediction:topk(bin_index, true, true)

            for j = 1, bin_index do
                if labels[frame_index] == indexes[j] then
                    correct_predictions[bin_index] = correct_predictions[bin_index] + 1
                end
            end
        end


        total_predictions = total_predictions + 1

        fraction_done = frame_index / labels:size(1)
        
        print(fraction_done)
    end

    return correct_predictions, plus_minus_n_correct_predictions, total_predictions
end

