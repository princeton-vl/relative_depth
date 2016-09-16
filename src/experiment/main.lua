require 'hdf5'
require('optim')
require('os')
require('cunn')
require('paths')


cmd = torch.CmdLine()
cmd:option('-m', 'hourglass3', 'model file definition')
cmd:option('-bs', 4, 'batch size')
cmd:option('-it', 0, 'Iterations')
cmd:option('-lt', 1000, 'Loss file saving refresh interval (seconds)')
cmd:option('-mt', 10000, 'Model saving interval (iterations)')
cmd:option('-et', 3000, 'Model evaluation interval (iterations)')
cmd:option('-lr', 1e-2, 'Learning rate')
cmd:option('-t_depth_file','','Training file for relative depth')
cmd:option('-v_depth_file','','Validation file for relative depth')
cmd:option('-rundir', '', 'Running directory')
cmd:option('-ep', 10, 'Epochs')
cmd:option('-start_from','', 'Start from previous model')
cmd:option('-diw',false,'Is training on DIW dataset')

g_args = cmd:parse(arg)

-- Data Loader
if g_args.diw then
    paths.dofile('./DataLoader_DIW.lua')
    require('./validation_crit/validate_crit_DIW')
else
    paths.dofile('./DataLoader.lua')
    require('./validation_crit/validate_crit1')
end
paths.dofile('load_data.lua')
local train_loader = TrainDataLoader()
local valid_loader = ValidDataLoader()



----------to modify
if g_args.it == 0 then
    g_args.it = g_args.ep * (train_loader.n_relative_depth_sample) / g_args.bs
end

-- Run path
local jobid = os.getenv('PBS_JOBID')
local job_name = os.getenv('PBS_JOBNAME')
if g_args.rundir == '' then
    if jobid == '' then
        jobid = 'debug'
    else
        jobid = jobid:split('%.')[1]
    end
    g_args.rundir = '/home/wfchen/scratch/nips16_release/relative_depth/results/' .. g_args.m .. '/' .. job_name .. '/'
end
paths.mkdir(g_args.rundir)
torch.save(g_args.rundir .. '/g_args.t7', g_args)

-- Model
local config = {}
require('./models/' .. g_args.m)
if g_args.start_from ~= '' then
    require 'cudnn'
    print(g_args.rundir .. g_args.start_from)
    g_model = torch.load(g_args.rundir .. g_args.start_from);
    if g_model.period == nil then
        g_model.period = 1
    end
    g_model.period = g_model.period + 1
    config = g_model.config
else
    g_model = get_model()
    g_model.period = 1
end
g_model:training()
config.learningRate = g_args.lr



-- Criterion. get_criterion is a function, which is specified in the network model file
if get_criterion == nil then
    print("Error: no criterion specified!!!!!!!")
    os.exit()
end


-- Validation Criteria



-- Function that obtain depth from the model output, used in validation
get_depth_from_model_output = f_depth_from_model_output()
if get_depth_from_model_output == nil then
    print("Error: get_depth_from_model_output is undefined!!!!!!!")
    os.exit()    
end


-- Variables that used globally
g_criterion = get_criterion()
g_model = g_model:cuda()
g_criterion = g_criterion:cuda()
g_params, g_grad_params = g_model:getParameters()




local function default_feval(current_params)
	local batch_input, batch_target = train_loader:load_next_batch(g_args.bs)

    -- reset grad_params
    g_grad_params:zero()    

    --forward & backward
    local batch_output = g_model:forward(batch_input)    
    local batch_loss = g_criterion:forward(batch_output, batch_target)
    local dloss_dx = g_criterion:backward(batch_output, batch_target)
    g_model:backward(batch_input, dloss_dx)    


    collectgarbage()

    return batch_loss, g_grad_params
end

local function save_loss_accuracy(t_loss, t_WKDR, v_loss, v_WKDR)       -- to check
    -- first convert to tensor    
    local _v_loss_tensor = torch.Tensor(v_loss)    
    local _t_loss_tensor = torch.Tensor(t_loss)
    local _v_WKDR_tensor = torch.Tensor(v_WKDR)    
    local _t_WKDR_tensor = torch.Tensor(t_WKDR)

    -- first remove the existing file
    local _full_filename = g_args.rundir .. 'loss_accuracy_record_period' .. g_model.period .. '.h5'
    os.execute("rm " .. _full_filename)
    
    local myFile = hdf5.open(_full_filename, 'w')
    myFile:write('/t_loss', _t_loss_tensor)        
    myFile:write('/v_loss', _v_loss_tensor)        
    myFile:write('/t_WKDR', _t_WKDR_tensor)        
    myFile:write('/v_WKDR', _v_WKDR_tensor)  
    myFile:close()    
end

local function save_model(model, dir, current_iter, config)
    model:clearState()        
    model.config = config
    torch.save(dir .. '/model_period'.. model.period .. '_' .. current_iter  .. '.t7' , model)
end

local function save_best_model(model, dir, config, iter)
    model:clearState()        
    model.config = config
    model.iter = iter
    torch.save(dir .. '/Best_model_period' .. model.period .. '.t7' , model)
end









-----------------------------------------------------------------------------------------------------

if feval == nil then
	feval = default_feval
end


local best_valid_set_error_rate = 1
local train_loss = {};
local train_WKDR = {};
local valid_loss = {};
local valid_WKDR = {};

local lfile = torch.DiskFile(g_args.rundir .. '/training_loss_period' .. g_model.period .. '.txt', 'w')

for iter = 1, g_args.it do
    local params, current_loss = optim.rmsprop(feval, g_params, config)
    print(current_loss[1])
    lfile:writeString(current_loss[1] .. '\n')
    
    train_loss[#train_loss + 1] = current_loss[1]

    if iter % g_args.mt == 0 then        
        print(string.format('Saving model at iteration %d...', iter))
        save_model(g_model, g_args.rundir, iter, config)        
    end
    if iter % g_args.lt == 0 then
        print(string.format('Flusing training loss file at iteration %d...', iter))
        lfile:synchronize()        
    end
    if iter % g_args.et == 0 or iter == 1 then
        print(string.format('Evaluating at iteration %d...', iter))
        
        local train_eval_loss, train_eval_WKDR = evaluate( train_loader, g_model, g_criterion, 100 )  
        local valid_eval_loss, valid_eval_WKDR = evaluate( valid_loader, g_model, g_criterion, 100 )  


        -- record 
        valid_loss[#valid_loss + 1] = valid_eval_loss
        valid_WKDR[#valid_WKDR + 1] = valid_eval_WKDR
        train_WKDR[#train_WKDR + 1] = train_eval_WKDR
        save_loss_accuracy(train_loss, train_WKDR, valid_loss, valid_WKDR)

        -- to test
        if best_valid_set_error_rate > valid_eval_WKDR then
            best_valid_set_error_rate = valid_eval_WKDR
            save_best_model(g_model, g_args.rundir, config, iter)
        end
    end
end

-- evaluate(g_model, g_args.bs, valid_loader)
lfile:close()
train_loader:close()
valid_loader:close()
save_model(g_model, g_args.rundir, g_args.it, config)
