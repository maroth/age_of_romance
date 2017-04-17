require 'age_of_romance'
require 'nn'
pcall(require, "cunn")


-- start training the network

-- CONFIGURATION
train_frame_dir = "/mnt/e/age_of_romance/mini_frames_test/"
test_frame_dir = "/mnt/e/age_of_romance/mini_frames_test/"

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
    log_level = 8,
    minibatch_size = 10,
    epochs = 1000,
    max_frames_per_directory = nil,
    learningRate = 0.001,
    learningRateDecay = 0,
    weightDecay = 0,
    dampening = 0,
    nesterov = false,
    momentum = 0,
}

local neural_network = nn.Sequential()
neural_network:add(nn.SpatialAveragePooling(9, 10, 5, 6))
neural_network:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.SpatialAveragePooling(3, 3, 3, 3))
neural_network:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.SpatialAveragePooling(6, 3, 6, 3))
neural_network:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
neural_network:add(nn.ReLU(true))
neural_network:add(nn.SpatialAveragePooling(2, 2, 2, 2))

neural_network:add(nn.View(64))
neural_network:add(nn.Linear(64, 16))
neural_network:add(nn.Linear(16, 1))

local criterion = nn.MSECriterion();

if (params.use_cuda) then
    neural_network = neural_network:cuda()
    criterion = criterion:cuda()
end

train_data(neural_network, criterion, params, train_frame_dir)
test_data(neural_network, criterion, params, test_frame_dir)
