require 'nn'

local SpatialConvolution = nn.SpatialConvolution
local SpatialMaxPooling = nn.SpatialMaxPooling

function build_neural_network()
    return toy()
end

function toy()
    local toy = nn.Sequential()
    toy:add(nn.SpatialConvolution(3, 3, 320, 189))
    toy:add(nn.View(3))
    toy:add(nn.Linear(3, 1))

    local criterion = nn.MSECriterion();

    return toy, criterion
end

function vgg_tiny()
    local vgg = nn.Sequential()
    vgg:add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.ReLU(true))
    vgg:add(SpatialMaxPooling(2, 2, 2, 2))

    vgg:add(nn.View(64*94*160))
    vgg:add(nn.Linear(64*94*160, 100))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Linear(100, 10))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Linear(10, 1))

    local criterion = nn.AbsCriterion()

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
