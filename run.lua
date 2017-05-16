require 'cspaces'
require 'test_data'
require 'neural_network'
require 'nn'
require 'load_logic'
require 'cunn'

local params = {
    name = 'experiment',
    save_frequency = 25,
    epochs = 1000,
    learningRate = 0.01,
    number_of_bins = 250,
    --learningRateDecay = 0.0001,
    --weightDecay = 0.00001,
    momentum = 0.01,
    --dampening = 0,
    --nesterov = false,
    log_level = 7,
}

if arg[1] == "continuous" then
    train_file_cspaces = "cspaces/continuous_train_data_250.bin"
    train_file_bins = "cspaces/continuous_train_labels_250.bin"
    validate_file_cspaces = "cspaces/continuous_validate_data_250.bin"
    validate_file_bins = "cspaces/continuous_validate_labels_250.bin"
    test_file_cspaces = "cspaces/continuous_test_data_250.bin"
    test_file_bins = "cspaces/continuous_test_labels_250.bin"
elseif arg[1] == "distributed" then
    train_file_cspaces = "cspaces/distributed_train_data_250.bin"
    train_file_bins = "cspaces/distributed_train_labels_250.bin"
    validate_file_cspaces = "cspaces/distributed_validate_data_250.bin"
    validate_file_bins = "cspaces/distributed_validate_labels_250.bin"
    test_file_cspaces = "cspaces/distributed_test_data_250.bin"
    test_file_bins = "cspaces/distributed_test_labels_250.bin"
elseif arg[1] == "separate" then
    train_file_cspaces = "cspaces/separate_train_data_250.bin"
    train_file_bins = "cspaces/separate_train_labels_250.bin"
    validate_file_cspaces = "cspaces/separate_validate_data_250.bin"
    validate_file_bins = "cspaces/separate_validate_labels_250.bin"
    test_file_cspaces = "cspaces/separate_test_data_250.bin"
    test_file_bins = "cspaces/separate_test_labels_250.bin"
elseif arg[1] == "preprocess" then
    all_cspaces()
end

local criterion = nn.ClassNLLCriterion():cuda()
local network = {}

if arg[2] == "full" then
    network = colorspace(params):cuda()
    params.name = arg[3] .. "_" .. arg[1] .. "_"
    neural_network = train(network, criterion, params, train_file_cspaces, train_file_bins, validate_file_cspaces, validate_file_bins)
    test(network, criterion, params, train_file_cspaces, train_file_bins, validate_file_cspaces, validate_file_bins, test_file_cspaces, test_file_bins)
end

if arg[2] == "train" then
    if arg[3] == "new" then 
        network = colorspace(params):cuda()
    else
        log(10, "loading model from " .. arg[3])        
        network = torch.load(arg[3])
        log(10, "model loaded")
    end
    
    params.name = arg[3] .. "_" .. arg[1] .. "_"

    neural_network = train(network, criterion, params, train_file_cspaces, train_file_bins, validate_file_cspaces, validate_file_bins)
end

if arg[2] == "test" then
    log(10, "loading model from " .. arg[3])
    network = torch.load(arg[3])
    log(10, "model loaded")
    print(train_file_cspaces)
    
    params.name = arg[3] .. "_" .. arg[1] .. "_"

    test(network, criterion, params, train_file_cspaces, train_file_bins, validate_file_cspaces, validate_file_bins, test_file_cspaces, test_file_bins)
end


