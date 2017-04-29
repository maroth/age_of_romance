require 'train_data'
require 'test_data'
require 'neural_network'
require 'nn'
pcall(require, "cunn")

--train_frame_dir = "frames_211x176/"
--test_frame_dir = "frames_211x176/"

train_frame_dir = "/media/markus/Data/age_of_romance/sanity_check/"
test_frame_dir = "/media/markus/Data/age_of_romance/sanity_check/"

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
    display_plot = true,
    model_filename = 'sanity_vgg',
    load_saved_model = false,
    number_of_bins = 5,
    minibatch_size = 10,
    epochs = 20, 
    max_frames_per_directory = 10,
    learningRate = 0.01,
    learningRateDecay = 0,
    weightDecay = 0,
    dampening = 0,
    nesterov = false,
    momentum = 0.01,
    log_level = 7,
}

local network = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
    network:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
    network:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
    network:add(nn.ReLU(true))
    return network
end

-- Will use "ceil" MaxPooling because we want to save as much feature space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,64):add(nn.Dropout(0.3))
ConvBNReLU(64,64)
network:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(64,128):add(nn.Dropout(0.4))
ConvBNReLU(128,128)
network:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(128,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256)
network:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(256,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
network:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
network:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
network:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
network:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
network:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
network:add(MaxPooling(2,2,2,2):ceil())
network:add(nn.View(512))
network:add(nn.Dropout(0.5))
network:add(nn.Linear(512,512))
network:add(nn.BatchNormalization(512))
network:add(nn.ReLU(true))
network:add(nn.Dropout(0.5))
network:add(nn.Linear(512, params.number_of_bins))

local criterion = nn.CrossEntropyCriterion();

if (params.use_cuda) then
    network = network:cuda()
    criterion = criterion:cuda()
end

neural_network = train(network, criterion, params, train_frame_dir)
test(neural_network, criterion, params, test_frame_dir)
