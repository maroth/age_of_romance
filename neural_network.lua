require 'nn'

function build_neural_network() 
    -- an AlexNet implementation takes from https://github.com/eladhoffer/ImageNet-Training/blob/master/Models/AlexNet.lua
    local SpatialConvolution = nn.SpatialConvolution
    local SpatialMaxPooling = nn.SpatialMaxPooling

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
    classifier:add(nn.Threshold(0, 1e-6))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(4096, 4096))
    classifier:add(nn.Threshold(0, 1e-6))
    classifier:add(nn.Linear(4096, 1000))
    classifier:add(nn.Linear(1000, 100))
    classifier:add(nn.Linear(100, 1))
    classifier:add(nn.Sigmoid())

    local model = nn.Sequential()
    model:add(features):add(classifier)

    return model
end
