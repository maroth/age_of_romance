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
    display_plot = false,
    model_filename = 'sanity_vgg',
    load_saved_model = false,
    number_of_bins = 10,
    minibatch_size = 16,
    epochs = 300, 
    max_frames_per_directory = 10,
    learningRate = 1e-2,
    learningRateDecay = 1e-4,
    weightDecay = 1e-3,
    momentum = 1e-4,
    --dampening = 0,
    --nesterov = false,
    log_level = 7
}

local network = nn.Sequential()

--network:add(nn.View(28*28):setNumInputDims(3))
network:add(nn.Reshape(3*28*28))
network:add(nn.Linear(28*28*3, 90))
network:add(nn.Tanh(true))
network:add(nn.Linear(90, params.number_of_bins))

print(network)

local criterion = nn.CrossEntropyCriterion();

if (params.use_cuda) then
    network = network:cuda()
    criterion = criterion:cuda()
end

neural_network = train(network, criterion, params, train_frame_dir)
test(neural_network, criterion, params, test_frame_dir)
