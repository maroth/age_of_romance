require 'cspaces'
require 'test_data'
require 'nn'
require 'load_logic'
require 'cunn'

function colorspace(params)
   local net = nn.Sequential()
   net:add(nn.View(3*176))

   net:add(nn.Linear(3*176, 176))
   net:add(nn.ReLU(true))
   net:add(nn.Linear(176, 1024))
   net:add(nn.ReLU(true))
   net:add(nn.Linear(1024, 1024))
   net:add(nn.ReLU(true))
   net:add(nn.Linear(1024, 128))
   net:add(nn.ReLU(true))
   net:add(nn.Linear(128, params.number_of_bins))
   net:add(nn.LogSoftMax())

   return net
end

local params = {
    name = 'experiment',
    save_frequency = 10,
    epochs = 1000000,
    learningRate = 0.01,
    number_of_bins = 5,
    minibatch_size = 250,
    learningRateDecay = 0.00001,
    weightDecay = 0.005,
    --momentum = 0.1,
    --dampening = 0,
    --nesterov = false,
    log_level = 7,
}

local files = {}

function load_preprocessed_data(name)
    files.train_file_cspaces = "cspaces/" .. name .. "_train_data.bin"
    files.train_file_ids = "cspaces/" .. name .. "_train_ids.bin"
    files.train_file_normalized_dates = "cspaces/" .. name .. "_train_normalized_dates.bin"

    files.validate_file_cspaces = "cspaces/" .. name .. "_validate_data.bin"
    files.validate_file_ids = "cspaces/" .. name .. "_validate_ids.bin"
    files.validate_file_normalized_dates= "cspaces/" .. name .. "_validate_normalized_dates.bin"

    files.test_file_cspaces = "cspaces/" .. name .. "_test_data.bin"
    files.test_file_ids = "cspaces/" .. name .. "_test_ids.bin"
    files.test_file_normalized_dates = "cspaces/" .. name .. "_test_normalized_dates.bin"
end

if arg[1] == "continuous" or arg[1] == "distributed" or arg[1] == "separate" then
    load_preprocessed_data(arg[1])
elseif arg[1] == "preprocess" then
    all_cspaces()
end

local criterion = nn.ClassNLLCriterion():cuda()
local network = {}

if arg[2] == "full" then
    network = colorspace(params):cuda()
    params.name = arg[3] .. "_" .. arg[1]
    neural_network = train(network, criterion, params, files)
    test(network, criterion, params, files)
end

if arg[2] == "train" then
    if arg[3] == "new" then 
        network = colorspace(params):cuda()
    else
        log(10, "loading model from " .. arg[3])        
        network = torch.load(arg[3])
        log(10, "model loaded")
    end
    
    params.name = arg[4] 

    neural_network = train(network, criterion, params, files)
end

if arg[2] == "test" then
    log(10, "loading model from " .. arg[3])
    network = torch.load(arg[3])
    log(10, "model loaded")
    print(files.train_file_cspaces)
    
    params.name = string.gsub(arg[3], "/", "_")  .. "_" .. arg[1]

    test(network, criterion, params, files)
end
