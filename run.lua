require 'train_data'
require 'test_data'
require 'neural_network'
require 'nn'
pcall(require, "cunn")

--validate_frame_dir = "frames_105x88_distributed/validate/"
--train_frame_dir = "frames_105x88_distributed/train/"
--test_frame_dir = "frames_105x88_distributed/test/"

validate_frame_dir = "frames_211x176_distributed_tiny/validate/"
train_frame_dir = "frames_211x176_distributed_tiny/train/"
test_frame_dir = "frames_211x176_distributed_tiny/test/"

--train_frame_dir = "./mnist/training/"
--test_frame_dir  = "./mnist/training/"
--validate_frame_dir = "./mnist/training/"


local params = {
    use_cuda = true,
    channels = 3,
    model_filename = 'colorspace',
    save_frequency = 5,
    number_of_bins = 6,
    minibatch_size = 32,
    epochs = 1024,
    max_frames_per_directory = nil,
    max_validate_frames_per_directory = nil,
    learningRate = 0.01,
    learningRateDecay = 0.0001,
    weightDecay = 0.0005,
    momentum = 0.01,
    --dampening = 0,
    --nesterov = false,
    log_level = 7,
}

local criterion = nn.ClassNLLCriterion():cuda()
local network = {}

if arg[1] == "train" then
    network = colorspace(params):cuda()
    --network = vgg_105_88(params):cuda()
    --network = vgg_mnist(params):cuda()

    neural_network = train(network, criterion, params, train_frame_dir, validate_frame_dir)
end

if arg[1] == "test" then
    log(10, "loading model from " .. arg[2])
    network = torch.load(arg[2])
    log(10, "model loaded")

    test(network, criterion, params, test_frame_dir, validate_frame_dir, train_frame_dir)
end
