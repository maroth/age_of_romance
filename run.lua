require 'age_of_romance'

-- start training the network

-- CONFIGURATION
train_frame_dir = "/mnt/e/age_of_romance/micro_frames_train/"
test_frame_dir = "/mnt/e/age_of_romance/micro_frames_train/"

-- command line argument 1 overrides training frame directory
if arg[1] ~= nil then
    train_frame_dir = arg[1]
end

-- command line argument 2 overrides testing frame directory
if arg[2] ~= nil then
    test_frame_dir = arg[2]
end

local params = {
    log_level = 1,
    minibatch_size = 2,
    epochs = 700,
    max_frames_per_directory = 6,
    learningRate = 0.001,
    learningRateDecay = 0.0001,
    weightDecay = 0,
    dampening = 0,
    nesterov = false,
    momentum = 0,
}

train_data(params, train_frame_dir)
test_data(params, test_frame_dir)
