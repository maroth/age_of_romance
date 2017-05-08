require 'nn'
require 'cunn'

function colorspace(params)
   local net = nn.Sequential()
   net:add(nn.View(3*176))

   net:add(nn.Linear(3*176, 4096))
   net:add(nn.ReLU(true))
   net:add(nn.Linear(4096, 4096))
   net:add(nn.ReLU(true))
   net:add(nn.Linear(4096, 4096))
   net:add(nn.ReLU(true))
   net:add(nn.Linear(4096, params.number_of_bins))
   net:add(nn.LogSoftMax())

   return net
end



