require 'train_data'
require 'test_data'
require 'neural_network'
require 'nn'
pcall(require, "cunn")

train_frame_dir = "frames_105x88/"
test_frame_dir  = "frames_105x88/"

train_frame_dir = "./mnist/training/"
test_frame_dir  = "./mnist/training/"


local params = {
    use_cuda = true,
    channels = 3,
    display_plot = false,
    model_filename = 'simple',
    number_of_bins = 10,
    minibatch_size = 128,
    epochs = 5, 
    max_frames_per_directory = 1000,
    learningRate = 0.5,
    learningRateDecay = 0.0001,
    weightDecay = 0.001,
    momentum = 0.0001,
    --dampening = 0,
    --nesterov = false,
    log_level = 7,
}

local criterion = nn.ClassNLLCriterion()
local network = {}

if arg[1] == "train" then
    --network = vgg_105_88(params)
    network = vgg_mnist(params)

    if (params.use_cuda) then
        network = network:cuda()
        criterion = criterion:cuda()
    end

    neural_network = train(network, criterion, params, train_frame_dir)
end

if arg[1] == "test" then
    log(10, "loading model from " .. arg[2])
    network = torch.load(arg[2])
    log(10, "model loaded")

    test(network, criterion, params, test_frame_dir)
end
