require 'nn'

function sanity_check(test_network, test_criterion, frame_size)
    local test_prediction = test_network:forward(torch.DoubleTensor(20, frame_size[1], frame_size[2], frame_size[3]))
    test_criterion:forward(test_prediction, torch.DoubleTensor(20, 1))
    local test_grad_criterion = test_criterion:backward(test_prediction, torch.DoubleTensor(20, 1))
    test_network:zeroGradParameters()
    test_network:backward(torch.DoubleTensor(20, frame_size[1], frame_size[2], frame_size[3]), test_grad_criterion)
    test_network:updateParameters(0.001)
end
