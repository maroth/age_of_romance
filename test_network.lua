require 'nn'
local status, lfs = pcall(require, "cunn")

function sanity_check(test_network, test_criterion, frame_size, params)

    local value = torch.DoubleTensor(params.minibatch_size, frame_size[1], frame_size[2], frame_size[3])
    local target = torch.LongTensor(params.minibatch_size)

    for index = 1, params.minibatch_size do
        target[index] = math.random(1, params.number_of_bins)
    end

    if (params.use_cuda) then
        value = value:cuda()
        -- target = target:cuda()
    end
    
    local test_prediction = test_network:forward(value)
    test_criterion:forward(test_prediction, target)

    if (params.use_cuda) then
        test_prediction = test_prediction:cuda()
        target = target:cuda()
    end

    local test_grad_criterion = test_criterion:backward(test_prediction, target)
    test_network:zeroGradParameters()
    test_network:backward(value, test_grad_criterion)
    test_network:updateParameters(0.001)
end
