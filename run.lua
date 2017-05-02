require 'train_data'
require 'test_data'
require 'neural_network'
require 'nn'
pcall(require, "cunn")

validate_frame_dir = "frames_105x88_distributed/validate/"
train_frame_dir = "frames_105x88_distributed/train/"
test_frame_dir = "frames_105x88_distributed/test/"

train_frame_dir = "./mnist/training/"
test_frame_dir  = "./mnist/training/"
validate_frame_dir = "./mnist/training/"


local params = {
    use_cuda = true,
    channels = 3,
    model_filename = 'mnist',
    save_frequency = 5,
    number_of_bins = 10,
    minibatch_size = 100,
    epochs = 200,
    max_frames_per_directory = 10,
    max_validate_frames_per_directory = 10,
    learningRate = 0.01,
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
    --network = vgg_105_88_tiny(params)
    network = vgg_mnist(params)

    network = network:cuda()
    criterion = criterion:cuda()

    neural_network = train(network, criterion, params, train_frame_dir, validate_frame_dir)
end

if arg[1] == "test" then
    log(10, "loading model from " .. arg[2])
    network = torch.load(arg[2])
    log(10, "model loaded")

    test(network, criterion, params, test_frame_dir, validate_frame_dir, train_frame_dir)
end
