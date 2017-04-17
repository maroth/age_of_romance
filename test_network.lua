require 'nn'
require 'cunn'

function sanity_check(test_network, test_criterion, frame_size)
    local value = torch.DoubleTensor(20, frame_size[1], frame_size[2], frame_size[3])
    value = value:cuda()
    local target = torch.DoubleTensor(20, 1)
    target = target:cuda()
    
    local test_prediction = test_network:forward(value)
    test_criterion:forward(test_prediction, target)
    test_prediction = test_prediction:cuda()
    target = target:cuda()
    local test_grad_criterion = test_criterion:backward(test_prediction, target)
    test_network:zeroGradParameters()
    test_network:backward(value, test_grad_criterion)
    test_network:updateParameters(0.001)
end
