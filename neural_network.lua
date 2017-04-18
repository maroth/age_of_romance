require 'nn'

local SpatialConvolution = nn.SpatialConvolution
local SpatialMaxPooling = nn.SpatialMaxPooling

function build_neural_network()
    return vgg_micro()
end

function toy()
    local toy = nn.Sequential()
    toy:add(nn.View(3*189*320))
    toy:add(nn.Sigmoid())
    toy:add(nn.Linear(3*189*320, 128))
    toy:add(nn.Sigmoid())
    toy:add(nn.Linear(128, 32))
    toy:add(nn.Sigmoid())
    toy:add(nn.Linear(32, 4))
    toy:add(nn.Sigmoid())
    toy:add(nn.Linear(4, 1))
    toy:add(nn.Sigmoid())


    local criterion = nn.AbsCriterion()

    return toy, criterion
end

function vgg_micro()
    local vgg = nn.Sequential()
    vgg:add(nn.SpatialAveragePooling(9, 10, 5, 6))
    vgg:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.SpatialAveragePooling(3, 3, 3, 3))
    vgg:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.SpatialAveragePooling(6, 3, 6, 3))
    vgg:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.SpatialAveragePooling(2, 2, 2, 2))

    vgg:add(nn.View(64))
    vgg:add(nn.Linear(64, 16))
    vgg:add(nn.Linear(16, 1))

    local criterion = nn.MSECriterion();

    return vgg, criterion
end

function vgg_tiny()
    local vgg = nn.Sequential()
    nn.SpatialBatchNormalization(3)
    vgg:add(nn.SpatialConvolution(3, 9, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    nn.SpatialBatchNormalization(3)
    vgg:add(nn.SpatialMaxPooling(4, 4, 6, 4))
    vgg:add(nn.SpatialConvolution(9, 18, 3, 3, 1, 1, 1, 1))
    nn.SpatialBatchNormalization(3)
    vgg:add(nn.ReLU(true))
    vgg:add(nn.SpatialMaxPooling(4, 4, 5, 4))
    vgg:add(nn.SpatialConvolution(18, 27, 3, 3, 1, 1, 1, 1))
    nn.SpatialBatchNormalization(3)
    vgg:add(nn.ReLU(true))
    vgg:add(nn.SpatialMaxPooling(8, 8, 8, 8))

    vgg:add(nn.View(27))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Linear(27, 25))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Linear(25, 1))

    local criterion = nn.SmoothL1Criterion();

    return vgg, criterion
end

function vgg_mini() 
    -- vgg16 taken from https://arxiv.org/pdf/1409.1556v6.pdf
    local vgg = nn.Sequential()

    vgg:add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    vgg:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg:add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    vgg:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg:add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    vgg:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    vgg:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg:add(nn.View(512*11*20))
    vgg:add(nn.Linear(512*11*20, 4095))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Linear(4095, 4095))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Linear(4095, 1000))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Linear(1000, 1))

    local criterion = nn.AbsCriterion()

    return vgg, criterion

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

    local criterion = nn.AbsCriterion()

    return vgg16, criterion

end
