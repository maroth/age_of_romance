require 'nn'
require 'cunn'

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

function alexnet(params)
    local SpatialConvolution = nn.SpatialConvolution
    local SpatialMaxPooling = nn.SpatialMaxPooling

    local features = nn.Sequential()
    features:add(SpatialConvolution(3,64,11,11,4,4,2,2))
    features:add(nn.ReLU(true))
    features:add(SpatialMaxPooling(3,3,2,2))
    features:add(SpatialConvolution(64,192,5,5,1,1,2,2))
    features:add(nn.ReLU(true))
    features:add(SpatialMaxPooling(3,3,2,2))
    features:add(SpatialConvolution(192,384,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(SpatialConvolution(384,256,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(SpatialConvolution(256,256,3,3,1,1,1,1))
    features:add(nn.ReLU(true))
    features:add(SpatialMaxPooling(3,3,2,2))

    features:add(nn.View(256*4*5):setNumInputDims(3))
    features:add(nn.Dropout(0.5))
    features:add(nn.Linear(256*4*5, 4096))
    features:add(nn.Threshold(0, 1e-6))
    features:add(nn.Dropout(0.5))
    features:add(nn.Linear(4096, 4096))
    features:add(nn.Threshold(0, 1e-6))
    features:add(nn.Linear(4096, 50))
    features:add(nn.LogSoftMax())

    print(features)

    return features
end

function vgg_211_176(params)
    local vgg = nn.Sequential()

    local function ConvBNReLU(nInputPlane, nOutputPlane)
        vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
        vgg:add(nn.ReLU(true))
        return vgg
    end

    ConvBNReLU(3,64):add(nn.Dropout(0.3))
    ConvBNReLU(64,64)
    vgg:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    ConvBNReLU(64,128):add(nn.Dropout(0.4))
    ConvBNReLU(128,128)
    vgg:add(nn.SpatialMaxPooling(2, 2, 2, 2))

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
    classifier:add(nn.Linear(4096,4096))
    classifier:add(nn.BatchNormalization(4096))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(4096,4096))
    classifier:add(nn.BatchNormalization(4096))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(4096, params.number_of_bins))
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

function simpleCase(params)
  model = nn.Sequential()
  model:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
  model:add(nn.Threshold())
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2))
  model:add(nn.Threshold())
  model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
  model:add(nn.Threshold())
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2))
  model:add(nn.Threshold())
  model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  model:add(nn.Threshold())
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
  model:add(nn.Threshold())
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2))
  model:add(nn.Threshold())

  model:add(nn.View(256*4*4))
  model:add(nn.Linear(256*4*4, 4096))
  model:add(nn.Threshold())
  model:add(nn.Linear(4096, 2048))
  model:add(nn.Threshold())
  model:add(nn.Linear(2048, 50))
  model:add(nn.LogSoftMax());
  return model
end

function aor_net(params)
    local model = nn.Sequential()

    model:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2,2))
    model:add(nn.Threshold())

    model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2,2))
    model:add(nn.Threshold())

    model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2,2))
    model:add(nn.Threshold())

    model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2,2))
    model:add(nn.Threshold())

    model:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2,2))
    model:add(nn.Threshold())

    model:add(nn.SpatialConvolution(1024, 4096, 3, 3, 1, 1, 1, 1))
    model:add(nn.Threshold())
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2,2))
    model:add(nn.Threshold())

    model:add(nn.View(4096))
    model:add(nn.Linear(4096, 4096))
    model:add(nn.Threshold())
    model:add(nn.Linear(4096, 2048))
    model:add(nn.Threshold())
    model:add(nn.Linear(2048, 1024))
    model:add(nn.Threshold())
    model:add(nn.Linear(1024, 50))
    model:add(nn.LogSoftMax());
    return model
end
