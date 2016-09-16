local function _classify(z_A, z_B, ground_truth, thresh)
    local _classify_res = 1;
    if z_A - z_B > thresh then
        _classify_res = 1
    elseif z_A - z_B < -thresh then
        _classify_res = -1
    elseif z_A - z_B <= thresh and z_A - z_B >= -thresh then
        _classify_res = 0;
    end

    if _classify_res == ground_truth then
        return true
    else
        return false
    end
end



local function _count_correct(output, target, record)
    
    for point_idx = 1, target.n_point do

        x_A = target.x_A[point_idx]
        y_A = target.y_A[point_idx]
        x_B = target.x_B[point_idx]
        y_B = target.y_B[point_idx]

        z_A = output[{1, 1, y_A, x_A}]
        z_B = output[{1, 1, y_B, x_B}]

        assert(x_A ~= x_B or y_A ~= y_B)

        ground_truth = target.ordianl_relation[point_idx];    -- the ordinal_relation is in the form of 1 and -1 and 0

        for tau_idx = 1 , record.n_thresh do
            if _classify(z_A, z_B, ground_truth, record.thresh[tau_idx]) then
                
                if ground_truth == 0 then
                    record.eq_correct_count[tau_idx] = record.eq_correct_count[tau_idx] + 1;
                elseif ground_truth == 1 or ground_truth == -1 then
                    record.not_eq_correct_count[tau_idx] = record.not_eq_correct_count[tau_idx] + 1;
                end

            end
        end

        if ground_truth == 0 then
            record.eq_count = record.eq_count + 1
        elseif ground_truth == 1 or ground_truth == -1 then
            record.not_eq_count = record.not_eq_count + 1
        end

    end
    
end

local _eval_record = {}
_eval_record.n_thresh = 15;
_eval_record.eq_correct_count = torch.Tensor(_eval_record.n_thresh )
_eval_record.not_eq_correct_count = torch.Tensor(_eval_record.n_thresh )
_eval_record.not_eq_count = 0;
_eval_record.eq_count = 0;
_eval_record.thresh = torch.Tensor(_eval_record.n_thresh )
_eval_record.WKDR = torch.Tensor(_eval_record.n_thresh , 4)
for i = 1, _eval_record.n_thresh  do
    _eval_record.thresh[i] = i * 0.1;
end


local function reset_record(record)
    record.eq_correct_count:fill(0)
    record.not_eq_correct_count:fill(0)
    record.WKDR:fill(0)
    record.not_eq_count = 0
    record.eq_count = 0
    -- print(record.eq_correct_count)
    -- print(record.not_eq_correct_count)
    -- print(record.not_eq_count)
    -- print(record.eq_count)
    -- io.read()
end


function evaluate( data_loader, model, criterion, max_n_sample )  
    print('>>>>>>>>>>>>>>>>>>>>>>>>> Valid Crit Threshed: Evaluating on validation set...');

    print("Evaluate() Switch  On!!!")
    model:evaluate()    -- this is necessary
    
    -- reset the record
    reset_record(_eval_record)


    local total_validation_loss = 0;
    local n_iters = math.min(data_loader.n_relative_depth_sample, max_n_sample);                
    local n_total_point_pair = 0;    

    print(string.format("Number of samples we are going to examine: %d", n_iters))

    for iter = 1, n_iters do
        -- Get sample one by one
        local batch_input, batch_target = data_loader:load_indices(torch.Tensor({iter}), nil)   
        -- The the relative depth target. Since there is only one sample, we just take its first component.
        local relative_depth_target = batch_target[1]

        -- forward
        local batch_output = model:forward(batch_input)    
        local batch_loss = criterion:forward(batch_output, batch_target);   

        local output_depth = get_depth_from_model_output(batch_output)
        -- count the number of correct point pairs.
        _count_correct(output_depth, relative_depth_target, _eval_record)        

        -- get relative depth loss
        total_validation_loss = total_validation_loss + batch_loss * relative_depth_target.n_point        -- 

        -- update the number of point pair
        n_total_point_pair = n_total_point_pair + relative_depth_target.n_point

        collectgarbage()        
    end   


    print("Evaluate() Switch Off!!!")
    model:training()



    local max_min = 0;
    local max_min_i = 1;
    for tau_idx = 1 , _eval_record.n_thresh do
        _eval_record.WKDR[{tau_idx, 1}] = _eval_record.thresh[tau_idx]
        _eval_record.WKDR[{tau_idx, 2}] = (_eval_record.eq_correct_count[tau_idx] + _eval_record.not_eq_correct_count[tau_idx]) / (_eval_record.eq_count + _eval_record.not_eq_count)
        _eval_record.WKDR[{tau_idx, 3}] = _eval_record.eq_correct_count[tau_idx] / _eval_record.eq_count
        _eval_record.WKDR[{tau_idx, 4}] = _eval_record.not_eq_correct_count[tau_idx] / _eval_record.not_eq_count
        
        if math.min(_eval_record.WKDR[{tau_idx,3}], _eval_record.WKDR[{tau_idx,4}]) > max_min then
            max_min = math.min(_eval_record.WKDR[{tau_idx,3}], _eval_record.WKDR[{tau_idx,4}])
            max_min_i = tau_idx;
        end
    end

    print(_eval_record.WKDR)
    print(_eval_record.WKDR[{{max_min_i}, {}}])        
    print(string.format("\tEvaluation Completed. Loss = %f, WKDR = %f", total_validation_loss, 1 - max_min))

    --Return the loss per point pair, and ERROR ratio
    return total_validation_loss / n_total_point_pair, 1 - max_min
end