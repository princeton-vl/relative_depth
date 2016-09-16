
function inception(input_size, config) -- activations: input_resolution * (config[1][1] + (#config - 1) * (out_a + out_b))

   local concat = nn.Concat(2)

   -- Base 1 x 1 conv layer
   local conv = nn.Sequential()
   conv:add(cudnn.SpatialConvolution(input_size,config[1][1],1,1))
   conv:add(nn.SpatialBatchNormalization(config[1][1], nil, nil, false))
   conv:add(cudnn.ReLU(true)) -- input_R * config[1][1] * N 
   concat:add(conv)

   -- Additional layers
   local num_conv = table.getn(config)
   for i = 2,num_conv do
       conv = nn.Sequential()
       local filt = config[i][1]
       local pad = (filt - 1) / 2
       local out_a = config[i][2]
       local out_b = config[i][3]
       -- Reduction
       conv:add(cudnn.SpatialConvolution(input_size,out_a,1,1))
       conv:add(nn.SpatialBatchNormalization(out_a,nil,nil,false))
       conv:add(cudnn.ReLU(true))   -- input_R * out_a * N
       -- Spatial Convolution
       conv:add(cudnn.SpatialConvolution(out_a,out_b,filt,filt,1,1,pad,pad))
       conv:add(nn.SpatialBatchNormalization(out_b,nil,nil,false))
       conv:add(cudnn.ReLU(true))        -- input_R * out_b * N
       concat:add(conv)
   end

   return concat

end

