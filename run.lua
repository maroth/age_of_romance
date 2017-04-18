require 'age_of_romance'
require 'train_data'
require 'neural_network'
require 'nn'
pcall(require, "cunn")

train_frame_dir = "/mnt/e/age_of_romance/micro_frames_test/"
test_frame_dir = "/mnt/e/age_of_romance/micro_frames_test/"

-- command line argument 1 overrides training frame directory
if arg[1] ~= nil then
    train_frame_dir = arg[1]
end

-- command line argument 2 overrides testing frame directory
if arg[2] ~= nil then
    test_frame_dir = arg[2]
end

local params = {
    use_cuda = false,
    log_level = 1,
    minibatch_size = 1,
    epochs = 4,
    max_frames_per_directory = 1,
    learningRate = 0.001,
    learningRateDecay = 0,
    weightDecay = 0,
    dampening = 0,
    nesterov = false,
    momentum = 0,
}

local neural_network = toy()

local criterion = nn.MSECriterion();

if (params.use_cuda) then
    neural_network = neural_network:cuda()
    criterion = criterion:cuda()
end

train(neural_network, criterion, params, train_frame_dir)
--test_data(neural_network, criterion, params, test_frame_dir)
