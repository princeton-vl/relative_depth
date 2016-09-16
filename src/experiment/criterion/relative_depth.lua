---------------------------------------GPU version
require 'cutorch'

local relative_depth_crit, parent = torch.class('nn.relative_depth_crit', 'nn.Criterion')

function relative_depth_crit:__init()
    parent.__init(self)
    self.buffer = torch.Tensor()
end



function relative_depth_crit:_loss_func( z_A, z_B, ground_truth )
    --print('checkpoint in relative_depth_crit  _loss_func')
    --io.read()
    
    local val = 0
    if ground_truth == 0 then
        val = math.max( 0, (z_A - z_B) * (z_A - z_B) )
    else
        -- this is the hinge loss function
        val = math.log( 1 + math.exp( - ground_truth * (z_A - z_B) ) )             
    end
    return val;
end

function relative_depth_crit:_loss_func_arr(z_A, z_B, ground_truth)
    local mask = torch.abs(ground_truth)
    local z_A_z_B = torch.add(z_A, -z_B)
    local fix_length_cuda_tensor = torch.zeros(#z_A_z_B):cuda()
    local val1 = torch.cmax(fix_length_cuda_tensor, torch.cmul(z_A_z_B, z_A_z_B))
    local val2 = torch.log( torch.add(fix_length_cuda_tensor:fill(1), torch.exp(torch.cmul(z_A_z_B, ground_truth) * -1.0)) )
    return torch.add(torch.cmul(val2, mask), torch.cmul(val1, torch.add(torch.ones(#mask):cuda(), -mask)))
end

function relative_depth_crit:_grad_loss_func( z_A, z_B, ground_truth )
    --print('checkpoint in relative_depth_crit  _grad_loss_func')
    --io.read() 

    if ground_truth == 0 then
        local d = (z_A - z_B) * (z_A - z_B)
        if d > 0 then
            grad_A = 2 * (z_A - z_B)
            grad_B = - grad_A
        else
            grad_A = 0
            grad_B = 0
        end
    else
        -- gradient of the hinge loss function
        denominator = math.exp( ground_truth * (z_A - z_B) ) + 1;    
        grad_A = -ground_truth / denominator
        grad_B = ground_truth / denominator    
    end   

    return grad_A, grad_B
end

function relative_depth_crit:_grad_loss_func_arr(z_A, z_B, ground_truth)
    local mask = torch.abs(ground_truth)
    local z_A_z_B = torch.add(z_A, -z_B)
    local d = torch.cmul(z_A_z_B, z_A_z_B)
    local mask_d = torch.le(d, 0)
    local grad_A1 = z_A_z_B * 2
    local grad_B1 = - grad_A1
    grad_A1:maskedFill(mask_d, 0)
    grad_B1:maskedFill(mask_d, 0)

    local denom = torch.exp(torch.cmul(z_A_z_B, ground_truth)) + 1
    local grad_A2 = -torch.cdiv(ground_truth, denom)
    local grad_B2 = torch.cdiv(ground_truth, denom)

    local grad_A = torch.add(torch.cmul(mask, grad_A2), torch.cmul(-mask + 1, grad_A1))
    local grad_B = torch.add(torch.cmul(mask, grad_B2), torch.cmul(-mask + 1, grad_B1))

    return grad_A, grad_B
end

function relative_depth_crit:updateOutput2(input, target)
    -- the input is a 4D tensor : batch, channel(=1), height, width??????????????? Is the order correct?
    -- the target is an structure with these fields : the coordinates of point A and B, and their depth order( z_A > z_B or z_A < z_B)
    -- have to make sure that the x and y order match the tensor!
    
    self.output = 0
    
    local n_point_total = 0

    for batch_idx = 1 , input:size(1) do

        n_point_total = n_point_total + target[batch_idx].n_point

        -- print(target[batch_idx].n_point)
        for point_idx = 1, target[batch_idx].n_point do
            local x_A = target[batch_idx].x_A[point_idx]
            local y_A = target[batch_idx].y_A[point_idx]
            local x_B = target[batch_idx].x_B[point_idx]
            local y_B = target[batch_idx].y_B[point_idx]

            local z_A = input[{batch_idx, 1, y_A, x_A}]
            local z_B = input[{batch_idx, 1, y_B, x_B}]

            assert(x_A ~= x_B or y_A ~= y_B)

            local ground_truth = target[batch_idx].ordianl_relation[point_idx];    -- the ordianl_relation is in the form of 1 and -1?
            --[[
            ----------------------------
                        Debug

            print(string.format('%d,%d,%d,%d,%d',y_A,x_A,y_B,x_B,ground_truth))
            io.read()
            ]]
            -- print(z_A, z_B, ground_truth)

            self.output = self.output + self:_loss_func(z_A, z_B, ground_truth);
        end

    end

  
    
    return self.output / n_point_total
end

function relative_depth_crit:updateOutput(input, target)
    -- the input is a 4D tensor : batch, channel(=1), height, width??????????????? Is the order correct?
    -- the target is an structure with these fields : the coordinates of point A and B, and their depth order( z_A > z_B or z_A < z_B)
    -- have to make sure that the x and y order match the tensor!

    
    self.output = 0
    
    local n_point_total = 0

    local cpu_input = input

    for batch_idx = 1 , cpu_input:size(1) do

        n_point_total = n_point_total + target[batch_idx].n_point

        local x_A_arr = target[batch_idx].x_A
        local y_A_arr = target[batch_idx].y_A
        local x_B_arr = target[batch_idx].x_B
        local y_B_arr = target[batch_idx].y_B

        local batch_input = cpu_input[{batch_idx, 1, {}}]
        local z_A_arr = batch_input:index(2, x_A_arr):gather(1, y_A_arr:view(1, -1))
        local z_B_arr = batch_input:index(2, x_B_arr):gather(1, y_B_arr:view(1, -1))

        local ground_truth_arr = target[batch_idx].ordianl_relation
        self.output = self.output +  torch.sum(self:_loss_func_arr(z_A_arr, z_B_arr, ground_truth_arr))

    end
       
    return self.output / n_point_total
end


function relative_depth_crit:updateGradInput2(input, target)    
    
    -- pre-allocate memory and reset gradient to 0
    if self.gradInput then
        local nElement = self.gradInput:nElement()        
        if self.gradInput:type() ~= input:type() then
            self.gradInput = self.gradInput:typeAs(input);
        end
        self.gradInput:resizeAs(input)
        if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
        end
    end

    self.gradInput:zero()

    local n_point_total = 0

    -- calculate the gradients 
    for batch_idx = 1, input:size(1) do
        
        n_point_total = n_point_total + target[batch_idx].n_point

        for point_idx = 1, target[batch_idx].n_point do
            x_A = target[batch_idx].x_A[point_idx]
            x_B = target[batch_idx].x_B[point_idx]
            y_A = target[batch_idx].y_A[point_idx]
            y_B = target[batch_idx].y_B[point_idx]

            z_A = input[{batch_idx, 1, y_A, x_A}]
            z_B = input[{batch_idx, 1, y_B, x_B}]

            assert(x_A ~= x_B or y_A ~= y_B)
            
            ground_truth = target[batch_idx].ordianl_relation[point_idx];    -- the ordianl_relation is in the form of 1 and -1?

            grad_A, grad_B = self:_grad_loss_func( z_A, z_B, ground_truth )
            
            --print('check point in relative_depth_crit:updateGradInput')
            --io.read()
            self.gradInput[{batch_idx, 1, y_A, x_A}] = self.gradInput[{batch_idx, 1, y_A, x_A}] + grad_A       -- pay special attention here!!
            self.gradInput[{batch_idx, 1, y_B, x_B}] = self.gradInput[{batch_idx, 1, y_B, x_B}] + grad_B
        end

 
    end
       
    return self.gradInput:div( n_point_total )
end

function relative_depth_crit:updateGradInput(input, target)    
    
    -- pre-allocate memory and reset gradient to 0
    if self.gradInput then
        local nElement = self.gradInput:nElement()        
        if self.gradInput:type() ~= input:type() then
            self.gradInput = self.gradInput:typeAs(input);
        end
        self.gradInput:resizeAs(input)
        if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
        end
    end



    self.gradInput:zero()

    local n_point_total = 0

    local cpu_input = input        -- might cause some memory issue?  
    if self.buffer:type() ~= cpu_input:type() then
        self.buffer = self.buffer:typeAs(cpu_input);
    end  
    self.buffer:resizeAs(cpu_input)
    self.buffer:zero()

    -- calculate the gradients 
    for batch_idx = 1, input:size(1) do
        
        n_point_total = n_point_total + target[batch_idx].n_point

        -- for point_idx = 1, target[batch_idx].n_point do
        local x_A_arr = target[batch_idx].x_A
        local y_A_arr = target[batch_idx].y_A
        local x_B_arr = target[batch_idx].x_B
        local y_B_arr = target[batch_idx].y_B

        local batch_input = cpu_input[{batch_idx, 1, {}}]
        local z_A_arr = batch_input:index(2, x_A_arr):gather(1, y_A_arr:view(1, -1))
        local z_B_arr = batch_input:index(2, x_B_arr):gather(1, y_B_arr:view(1, -1))

        local ground_truth_arr = target[batch_idx].ordianl_relation

        grad_A, grad_B = self:_grad_loss_func_arr( z_A_arr, z_B_arr, ground_truth_arr )
        -- print(grad_A, grad_B)
        
        --print('check point in relative_depth_crit:updateGradInput')
        local p2 = torch.Tensor(cpu_input:size()[3], target[batch_idx].n_point):cuda()
        local p1 = torch.Tensor(cpu_input:size()[3], cpu_input:size()[4]):cuda()
        
        p1:zero()
        p2:zero()
        p2:scatter(1, y_A_arr:view(1, -1), grad_A:view(1, -1))
        p1:indexAdd(2, x_A_arr, p2)

        self.buffer[{batch_idx, 1, {}}] = self.buffer[{batch_idx, 1, {}}] + p1

        p1:zero()
        p2:zero()
        p2:scatter(1, y_B_arr:view(1, - 1), grad_B:view(1, -1))
        p1:indexAdd(2, x_B_arr, p2)

        self.buffer[{batch_idx, 1, {}}] = self.buffer[{batch_idx, 1, {}}] + p1
    end
       
    self.gradInput:copy(self.buffer)

    return self.gradInput:div( n_point_total )
end