require 'paths'
paths.dofile('layers/inception_new.lua')


function get_model()
    require 'cudnn'  
    require 'cunn'  
    local model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3,128,7,7,1,1,3,3))
    model:add(nn.SpatialBatchNormalization(128))
    model:add(cudnn.ReLU(true))
    --model:add(nn.SpatialFractionalMaxPooling(2,2,128,128))

    --model:add(nn.SpatialFractionalMaxPooling(2,2,64,64))
    





    -- input to _1channels is 256
    local _1channels = nn.ConcatTable()
    _1channels:add(
        nn.Sequential():add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}}))
        )
    _1channels:add(
        nn.Sequential():add(
            nn.SpatialAveragePooling(2,2,2,2)
        ):add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            nn.SpatialUpSamplingNearest(2)                             -- up to 8x, 256 channel
        )
    )
    _1channels = nn.Sequential():add(_1channels):add(nn.CAddTable())


    -- input to _2channels is 256
    local _2channels = nn.ConcatTable()
    _2channels:add(
        nn.Sequential():add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            inception(256, {{64}, {3,64,64}, {7,64,64}, {11,64,64}})
        )
    )
    _2channels:add(
        nn.Sequential():add(
            nn.SpatialAveragePooling(2,2,2,2)                           -- 8x
        ):add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            _1channels                                                  -- down 16x then up to 8x
        ):add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            inception(256, {{64}, {3,64,64}, {7,64,64}, {11,64,64}})
        ):add(
            nn.SpatialUpSamplingNearest(2)                              -- up to 4x. 256 channel
        )
    )
    _2channels = nn.Sequential():add(_2channels):add(nn.CAddTable())


    -- input to _3channels is 128
    local _3channels = nn.ConcatTable()
    _3channels:add(
        nn.Sequential():add(
            cudnn.SpatialMaxPooling(2, 2, 2, 2)                         -- 4 x
        ):add(
            inception(128, {{32}, {3,32,32}, {5,32,32}, {7,32,32}})
        ):add(
            inception(128, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})     --256
        ):add(
            _2channels
        ):add(
            inception(256, {{64}, {3,32,64}, {5,32,64}, {7,32,64}})
        ):add(
            inception(256, {{32}, {3,32,32}, {5,32,32}, {7,32,32}})
        ):add(
            nn.SpatialUpSamplingNearest(2))                              -- up to 2x , output is 128 channel       
    )

    _3channels:add(
        nn.Sequential():add(
            inception(128, {{32}, {3,32,32}, {5,32,32}, {7,32,32}})     --128
        ):add(
            inception(128, {{32}, {3,64,32}, {7,64,32}, {11,64,32}})
        )
    )

    _3channels = nn.Sequential():add(_3channels):add(nn.CAddTable())


    -- input to _4channels is 128
    local _4channels = nn.ConcatTable()
    _4channels:add(
        nn.Sequential():add(
            cudnn.SpatialMaxPooling(2, 2, 2, 2)                         -- 2 x 
        ):add(
            inception(128, {{32}, {3,32,32}, {5,32,32}, {7,32,32}})
        ):add(
            inception(128, {{32}, {3,32,32}, {5,32,32}, {7,32,32}})     -- 128
        ):add(
            _3channels
        ):add(
            inception(128, {{32}, {3,64,32}, {5,64,32}, {7,64,32}})
        ):add(
            inception(128, {{16}, {3,32,16}, {7,32,16}, {11,32,16}})
        ):add(
            nn.SpatialUpSamplingNearest(2)                              -- up to original, 64 channel
        )

    )

    _4channels:add(
        nn.Sequential():add(
            inception(128, {{16}, {3,64,16}, {7,64,16}, {11,64,16}})
            --nn.Identity()
        )
    )

    _4channels = nn.Sequential():add(_4channels):add(nn.CAddTable())


    model:add(_4channels)

    --Final Output        
    model:add(cudnn.SpatialConvolution(64,1,3,3,1,1,1,1));    


    return model 
end


require('../criterion/relative_depth')
function get_criterion()
    return nn.relative_depth_crit()
end


function f_depth_from_model_output()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>    depth = model_output")
    return ____get_depth_from_model_output
end

function ____get_depth_from_model_output(model_output)
    return model_output
end