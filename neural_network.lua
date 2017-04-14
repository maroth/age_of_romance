require 'nn'

local SpatialConvolution = nn.SpatialConvolution
local SpatialMaxPooling = nn.SpatialMaxPooling

function build_neural_network()
    return vgg16()
end

function alexNet()
    -- an AlexNet implementation takes from https://github.com/eladhoffer/ImageNet-Training/blob/master/Models/AlexNet.lua

    local features = nn.Sequential()
    features:add(SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
    features:add(nn.ReLU(true))
    features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
    features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
    features:add(nn.ReLU(true))
    features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
    features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
    features:add(nn.ReLU(true))
    features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
    features:add(nn.ReLU(true))
    features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
    features:add(nn.ReLU(true))
    features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

    local classifier = nn.Sequential()
    classifier:add(nn.View(256*6*6))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(256*6*6, 4096))
    --classifier:add(nn.Threshold(0, 1e-6))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    --classifier:add(nn.Threshold(0, 1e-6))
    classifier:add(nn.Linear(4096, 1000))
    classifier:add(nn.Linear(1000, 100))
    classifier:add(nn.Linear(100, 1))
    classifier:add(nn.Sigmoid())

    local model = nn.Sequential()
    model:add(features):add(classifier)

    return model
end

function vgg16() 
    -- vgg16 taken from https://arxiv.org/pdf/1409.1556v6.pdf
    local vgg16 = nn.Sequential()

    vgg16:add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg16:add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg16:add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg16:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg16:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    vgg16:add(nn.ReLU(true))
    vgg16:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg16:add(nn.View(512*5*10))
    vgg16:add(nn.Linear(512*5*10, 4095))
    vgg16:add(nn.ReLU(true))
    vgg16:add(nn.Linear(4095, 4095))
    vgg16:add(nn.ReLU(true))
    vgg16:add(nn.Linear(4095, 1000))
    vgg16:add(nn.ReLU(true))
    vgg16:add(nn.Linear(1000, 1))
    vgg16:add(nn.Sigmoid())
    return vgg16
end
