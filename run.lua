require 'cspaces'
require 'test_data'
require 'neural_network'
require 'nn'
require 'load_logic'
pcall(require, "cunn")


train_file_cspaces = "cspaces/distributed_train_data_250.bin"
train_file_bins = "cspaces/distributed_train_labels_250.bin"
validate_file_cspaces = "cspaces/distributed_validate_data_250.bin"
validate_file_bins = "cspaces/distributed_validate_labels_250.bin"
test_file_cspaces = "cspaces/distributed_test_data_250.bin"
test_file_bins = "cspaces/distributed_test_labels_250.bin"


local params = {
    model_filename = 'continuous_relu',
    save_frequency = 25,
    epochs = 100000,
    learningRate = 0.01,
    number_of_bins = 250,
    --learningRateDecay = 0.0001,
    --weightDecay = 0.00001,
    --momentum = 0.01,
    --dampening = 0,
    --nesterov = false,
    log_level = 7,
}

local criterion = nn.ClassNLLCriterion():cuda()
local network = {}

if arg[1] == "train" then
    if arg[2] == "new" then 
        network = colorspace(params):cuda()
    else
        log(10, "loading model from " .. arg[2])        
        network = torch.load(arg[2])
        log(10, "model loaded")
    end

    neural_network = train(network, criterion, params, train_file_cspaces, train_file_bins, validate_file_cspaces, validate_file_bins)
end

if arg[1] == "test" then
    log(10, "loading model from " .. arg[2])
    network = torch.load(arg[2])
    log(10, "model loaded")
    print(train_file_cspaces)

    test(network, criterion, params, train_file_cspaces, train_file_bins, validate_file_cspaces, validate_file_bins, test_file_cspaces, test_file_bins)
end

if arg[1] == "preprocess" then
    all_cspaces()
end
