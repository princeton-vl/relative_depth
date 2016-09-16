local function _is_correct(z_A, z_B, ground_truth)
    assert( ground_truth ~= 0, 'Warining: The ground_truth is not supposed to be 0.')
    local _classify_res = 1;
    if z_A > z_B then
        _classify_res = 1
    elseif z_A < z_B then
        _classify_res = -1    
    end

    if _classify_res == ground_truth then
        return true
    else
        return false
    end
end

local function _count_correct(output, target)
    assert(output:size(1) == 1)
    y_A = target.y_A[1]
    x_A = target.x_A[1]
    y_B = target.y_B[1]
    x_B = target.x_B[1]    

    z_A = output[{1, 1, y_A, x_A}]
    z_B = output[{1, 1, y_B, x_B}]

    assert(x_A ~= x_B or y_A ~= y_B)

    ground_truth = target.ordianl_relation[1];    -- the ordinal_relation is in the form of 1 and -1


    if _is_correct(z_A, z_B, ground_truth) then
        return 1
    else
        return 0
    end
end

function evaluate( data_loader, model, criterion, max_n_sample )  
    print('>>>>>>>>>>>>>>>>>>>>>>>>> Valid Crit DIW: Evaluating on validation set...');

    print("Evaluate() Switch  On!!!")
    model:evaluate()    -- this is necessary
    
    local total_validation_loss = 0;
    local n_iters = 200;
    local n_total_validate_samples = 0;
    local correct_count = 0;

    print(string.format("Number of samples we are going to examine: %d", n_iters))

    for iter = 1, n_iters do
         -- Get sample one by one
        local batch_input, batch_target = data_loader:load_indices(torch.Tensor({iter}), nil)
        -- The relative depth target. Since there is only one sample, we just take its first component.
        local relative_depth_target = batch_target[1]

        -- forward
        local batch_output = model:forward(batch_input)    
        local batch_loss = criterion:forward(batch_output, batch_target);   

        local output_depth = get_depth_from_model_output(batch_output)

        -- check this output
        local _n_point_correct = _count_correct(output_depth, relative_depth_target)
        
        -- get validation loss and correct ratio
        total_validation_loss = total_validation_loss + batch_loss 
        correct_count = correct_count + _n_point_correct
        n_total_validate_samples = n_total_validate_samples + 1

        collectgarbage()
    end   

    print("Evaluate() Switch Off!!!")
    model:training()

    local WHDR = 1 - correct_count / n_total_validate_samples
    print("Evaluation result: WHDR = ", WHDR)
    --Return the loss per point pair, and ERROR ratio
    return total_validation_loss / n_total_validate_samples, WHDR
end
