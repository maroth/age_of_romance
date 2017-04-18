require 'train_data'
require 'test_data'
require 'neural_network'
require 'nn'
pcall(require, "cunn")

train_frame_dir = "/mnt/e/age_of_romance/sanity_check/"
test_frame_dir = "/mnt/e/age_of_romance/sanity_check/"

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
    display_plot = true,
    model_filename = 'model.bin',
    load_saved_model = false,
    log_level = 6,
    minibatch_size = 6,
    epochs = 120, 
    max_frames_per_directory = 1,
    learningRate = 0.05,
    learningRateDecay = 0.01,
    weightDecay = 0,
    dampening = 0,
    nesterov = false,
    momentum = 0.01,
}

local neural_network = toy()
local criterion = nn.AbsCriterion();

if (params.use_cuda) then
    neural_network = neural_network:cuda()
    criterion = criterion:cuda()
end

neural_network = train(neural_network, criterion, params, train_frame_dir)
test(neural_network, criterion, params, test_frame_dir)
