require 'train_data'
require 'test_data'
require 'neural_network'
require 'nn'
pcall(require, "cunn")

--train_frame_dir = "frames_211x176/"
--test_frame_dir = "frames_211x176/"

train_frame_dir = "./mnist/training/"
test_frame_dir  = "./mnist/training/"

-- command line argument 1 overrides training frame directory
if arg[1] ~= nil then
    train_frame_dir = arg[1]
end

-- command line argument 2 overrides testing frame directory
if arg[2] ~= nil then
    test_frame_dir = arg[2]
end

local params = {
    use_cuda = true,
    channels = 3,
    display_plot = false,
    model_filename = 'sanity_vgg',
    load_saved_model = false,
    number_of_bins = 10,
    minibatch_size = 2048,
    epochs = 30, 
    max_frames_per_directory = nil,
    learningRate = 0.1,
    learningRateDecay = 0.0001,
    weightDecay = 0.001,
    momentum = 0.0001,
    --dampening = 0,
    --nesterov = false,
    log_level = 7
}

local network = nn.Sequential()

--network:add(nn.View(28*28):setNumInputDims(3))

network:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
network:add(nn.SpatialBatchNormalization(16))
network:add(nn.ReLU(true))
network:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
network:add(nn.SpatialBatchNormalization(16))
network:add(nn.ReLU(true))
network:add(nn.SpatialMaxPooling(2, 2, 2, 2))

network:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
network:add(nn.SpatialBatchNormalization(32))
network:add(nn.ReLU(true))
network:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
network:add(nn.SpatialBatchNormalization(32))
network:add(nn.ReLU(true))
network:add(nn.SpatialMaxPooling(2, 2, 2, 2))

network:add(nn.View(32*7*7):setNumInputDims(3))
network:add(nn.Linear(32*7*7, 1024))
network:add(nn.ReLU())
network:add(nn.Linear(1024, 1024))
network:add(nn.ReLU())
network:add(nn.Linear(1024, params.number_of_bins))
network:add(nn.LogSoftMax())

print(network)

--local criterion = nn.CrossEntropyCriterion();
local criterion = nn.ClassNLLCriterion()

if (params.use_cuda) then
    network = network:cuda()
    criterion = criterion:cuda()
end

neural_network = train(network, criterion, params, train_frame_dir)
test(neural_network, criterion, params, test_frame_dir)
