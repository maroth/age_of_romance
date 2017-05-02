require 'nn'

function vgg_105_88_tiny(params)
    local vgg = nn.Sequential()

    local function ConvBNReLU(nInputPlane, nOutputPlane)
        vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
        vgg:add(nn.ReLU(true))
        return vgg
    end

    ConvBNReLU(3,6):add(nn.Dropout(0.3))
    vgg:add(nn.SpatialMaxPooling(3,3,3,3))

    ConvBNReLU(6, 8):add(nn.Dropout(0.4))
    vgg:add(nn.SpatialMaxPooling(3,3,3,3))

    ConvBNReLU(8, 10):add(nn.Dropout(0.4))
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    ConvBNReLU(10, 12):add(nn.Dropout(0.4))
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    ConvBNReLU(12, 14):add(nn.Dropout(0.4))
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    vgg:add(nn.View(14):setNumInputDims(3))

    classifier = nn.Sequential()
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(14, 14))
    classifier:add(nn.BatchNormalization(14))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(14, 14))
    classifier:add(nn.BatchNormalization(14))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(14, 14))
    classifier:add(nn.BatchNormalization(14))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(14, params.number_of_bins))
    classifier:add(nn.LogSoftMax())
    vgg:add(classifier)

    -- initialize weights and bias of convoluations
    local function MSRinit(net)
        for k, v in pairs(net:findModules('nn.SpatialConvolution')) do
            local n = v.kW * v.kH * v.nOutputPlane
            v.weight:normal(0, math.sqrt(2 / n))
            v.bias:zero()
        end
    end

    MSRinit(vgg)

    return vgg
end

function vgg_105_88_small(params)
    local vgg = nn.Sequential()

    local function ConvBNReLU(nInputPlane, nOutputPlane)
        vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
        vgg:add(nn.ReLU(true))
        return vgg
    end

    ConvBNReLU(3,8):add(nn.Dropout(0.3))
    ConvBNReLU(8, 8)
    vgg:add(nn.SpatialMaxPooling(3,3,3,3))

    ConvBNReLU(8, 16):add(nn.Dropout(0.4))
    ConvBNReLU(16, 16)
    vgg:add(nn.SpatialMaxPooling(3,3,3,3))

    ConvBNReLU(16, 32):add(nn.Dropout(0.4))
    ConvBNReLU(32, 32)
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    ConvBNReLU(32, 64):add(nn.Dropout(0.4))
    ConvBNReLU(64, 64)
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    ConvBNReLU(64, 128):add(nn.Dropout(0.4))
    ConvBNReLU(128, 128)
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    vgg:add(nn.View(128):setNumInputDims(3))

    classifier = nn.Sequential()
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(128, 128))
    classifier:add(nn.BatchNormalization(128))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(128, params.number_of_bins))
    classifier:add(nn.LogSoftMax())
    vgg:add(classifier)

    -- initialize weights and bias of convoluations
    local function MSRinit(net)
        for k, v in pairs(net:findModules('nn.SpatialConvolution')) do
            local n = v.kW * v.kH * v.nOutputPlane
            v.weight:normal(0, math.sqrt(2 / n))
            v.bias:zero()
        end
    end

    MSRinit(vgg)

    return vgg
end

function vgg_105_88(params)
    local vgg = nn.Sequential()

    local function ConvBNReLU(nInputPlane, nOutputPlane)
        vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
        vgg:add(nn.ReLU(true))
        return vgg
    end

    ConvBNReLU(3,64):add(nn.Dropout(0.3))
    ConvBNReLU(64,64)
    vgg:add(nn.SpatialMaxPooling(3,3,3,3))

    ConvBNReLU(64,128):add(nn.Dropout(0.4))
    ConvBNReLU(128,128)
    vgg:add(nn.SpatialMaxPooling(3,3,3,3))

    ConvBNReLU(128,256):add(nn.Dropout(0.4))
    ConvBNReLU(256,256):add(nn.Dropout(0.4))
    ConvBNReLU(256,256)
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    ConvBNReLU(256,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    vgg:add(nn.SpatialMaxPooling(2,2,2,2))

    vgg:add(nn.View(512):setNumInputDims(3))

    classifier = nn.Sequential()
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(512,512))
    classifier:add(nn.BatchNormalization(512))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(512, params.number_of_bins))
    classifier:add(nn.LogSoftMax())
    vgg:add(classifier)

    -- initialize weights and bias of convoluations
    local function MSRinit(net)
        for k, v in pairs(net:findModules('nn.SpatialConvolution')) do
            local n = v.kW * v.kH * v.nOutputPlane
            v.weight:normal(0, math.sqrt(2 / n))
            v.bias:zero()
        end
    end

    MSRinit(vgg)

    return vgg
end

function vgg_mnist(params)
    local network = nn.Sequential()

    network:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
    network:add(nn.SpatialBatchNormalization(16))
    network:add(nn.ReLU(true))
    network:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
    network:add(nn.SpatialBatchNormalization(16))
    network:add(nn.ReLU(true))
    network:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    network:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
    network:add(nn.SpatialBatchNormalization(32))
    network:add(nn.ReLU(true))
    network:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
    network:add(nn.SpatialBatchNormalization(32))
    network:add(nn.ReLU(true))
    network:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    network:add(nn.View(32*7*7):setNumInputDims(3))

    network:add(nn.Linear(32*7*7, 1024))
    network:add(nn.ReLU())
    network:add(nn.Linear(1024, 1024))
    network:add(nn.ReLU())
    network:add(nn.Linear(1024, params.number_of_bins))
    network:add(nn.LogSoftMax())

    return network
end


