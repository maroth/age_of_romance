require 'cspaces'
require 'test_data'
require 'neural_network'
require 'nn'
pcall(require, "cunn")


train_file_cspaces = "cspaces/separate_train_data_250.bin"
train_file_bins = "cspaces/separate_train_labels_250.bin"
validate_file_cspaces = "cspaces/separate_validate_data_250.bin"
validate_file_bins = "cspaces/separate_validate_labels_250.bin"
test_file_cspaces = "cspaces/separate_test_data_250.bin"
test_file_bins = "cspaces/separate_test_labels_250.bin"


local params = {
    model_filename = 'continuous_relu',
    save_frequency = 2,
    epochs = 1000,
    learningRate = 0.001,
    number_of_bins = 250,
    learningRateDecay = 0,
    weightDecay = 0,
    momentum = 0.01,
    --dampening = 0,
    --nesterov = false,
    log_level = 7,
}

local criterion = nn.ClassNLLCriterion():cuda()
local network = {}

if arg[1] == "train" then
    network = colorspace(params):cuda()

    neural_network = train(network, criterion, params, train_file_cspaces, train_file_bins, validate_file_cspaces, validate_file_bins)
end

if arg[1] == "test" then
    log(10, "loading model from " .. arg[2])
    network = torch.load(arg[2])
    log(10, "model loaded")
    print(train_file_cspaces)

    test(network, criterion, params, train_file_cspaces, train_file_bins, validate_file_cspaces, validate_file_bins, test_file_cspaces, test_file_bins)
end
