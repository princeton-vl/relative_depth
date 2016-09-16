require 'nn'
require 'image'
require 'csvigo'
require 'hdf5'
require 'cudnn'
require 'cunn'
require 'cutorch'

local function _read_data_handle( _filename )        
    -- the file is a csv file
    local csv_file_handle = csvigo.load({path = _filename, mode = 'large'})
    
    local _n_lines = #csv_file_handle;


    local _data ={}
    local _line_idx = 2 --skip the first line
    local _sample_idx = 0
    while _line_idx <= _n_lines do

        _sample_idx = _sample_idx + 1
        

        _data[_sample_idx] = {};
        _data[_sample_idx].img_filename = csv_file_handle[ _line_idx ][ 1 ]    
        _data[_sample_idx].n_point = tonumber(csv_file_handle[ _line_idx ][ 3 ])

        _data[_sample_idx].y_A = {}
        _data[_sample_idx].y_B = {}
        _data[_sample_idx].x_A = {}
        _data[_sample_idx].x_B = {}
        _data[_sample_idx].ordianl_relation = {}       
        
        _line_idx = _line_idx + 1 
        for point_idx = 1 , _data[_sample_idx].n_point do
            
            _data[_sample_idx].y_A[point_idx] = tonumber(csv_file_handle[ _line_idx ][ 1 ])
            _data[_sample_idx].x_A[point_idx] = tonumber(csv_file_handle[ _line_idx ][ 2 ])
            _data[_sample_idx].y_B[point_idx] = tonumber(csv_file_handle[ _line_idx ][ 3 ])
            _data[_sample_idx].x_B[point_idx] = tonumber(csv_file_handle[ _line_idx ][ 4 ])

             -- Important!
            if csv_file_handle[ _line_idx ][ 5 ] == '>' then
                _data[_sample_idx].ordianl_relation[point_idx] = 1;
            elseif csv_file_handle[ _line_idx ][ 5 ] == '<' then
                _data[_sample_idx].ordianl_relation[point_idx] = -1;    
            elseif csv_file_handle[_line_idx][ 5 ] == '=' then        -- important!
                _data[_sample_idx].ordianl_relation[point_idx] = 0;
            end

            _line_idx = _line_idx + 1 
        end                 

    end

    return _sample_idx, _data;
end

local function _evaluate_correctness_our(_batch_output, _batch_target, WKDR, WKDR_eq, WKDR_neq)
    local n_gt_correct = torch.Tensor(n_thresh):fill(0);
    local n_gt = 0;

    local n_lt_correct = torch.Tensor(n_thresh):fill(0);
    local n_lt = 0;

    local n_eq_correct = torch.Tensor(n_thresh):fill(0);
    local n_eq = 0;

    for point_idx = 1, _batch_target.n_point do

        x_A = _batch_target.x_A[point_idx]
        y_A = _batch_target.y_A[point_idx]
        x_B = _batch_target.x_B[point_idx]
        y_B = _batch_target.y_B[point_idx]

        z_A = _batch_output[{1, 1, y_A, x_A}]
        z_B = _batch_output[{1, 1, y_B, x_B}]
        

        ground_truth = _batch_target.ordianl_relation[point_idx];    -- the ordianl_relation is in the form of 1 and -1

        for thresh_idx = 1, n_thresh do

            local _classify_res = 1;
            if z_A - z_B > thresh[thresh_idx] then
                _classify_res = 1
            elseif z_A - z_B < -thresh[thresh_idx] then
                _classify_res = -1
            elseif z_A - z_B <= thresh[thresh_idx] and z_A - z_B >= -thresh[thresh_idx] then
                _classify_res = 0;
            end

            if _classify_res == 0 and ground_truth == 0 then
                n_eq_correct[thresh_idx] = n_eq_correct[thresh_idx] + 1;
            elseif _classify_res == 1 and ground_truth == 1 then
                n_gt_correct[thresh_idx] = n_gt_correct[thresh_idx] + 1;
            elseif _classify_res == -1 and ground_truth == -1 then
                n_lt_correct[thresh_idx] = n_lt_correct[thresh_idx] + 1;
            end      
        end

        if ground_truth > 0 then
            n_gt = n_gt + 1;
        elseif ground_truth < 0 then
            n_lt = n_lt + 1;
        elseif ground_truth == 0 then
            n_eq = n_eq + 1;
        end
    end



    for i = 1 , n_thresh do        
        WKDR[{i}] = 1 - (n_eq_correct[i] + n_lt_correct[i] + n_gt_correct[i]) / (n_eq + n_lt + n_gt)
        WKDR_eq[{i}] = 1 - n_eq_correct[i]  / n_eq
        WKDR_neq[{i}] = 1 - (n_lt_correct[i] + n_gt_correct[i]) / (n_lt + n_gt)
    end
end



function crop_resize_input(img)
    local crop = cmd_params.crop
    local img_original_height = img:size(2)
    local img_original_width = img:size(3)

    local cropped_input = img[{{},{crop, img_original_height-crop}, {crop, img_original_width - crop}}]
    return image.scale(cropped_input,network_input_width ,network_input_height)
end

function inpaint_pad_output_our(output, img_original_width, img_original_height)
    local crop = cmd_params.crop
    local resize_height = img_original_height - 2*crop + 1
    local resize_width = img_original_width - 2*crop + 1
    local resize_output = image.scale( output[{1,1,{}}]:double(), resize_width, resize_height)
    local padded_output = torch.Tensor(1, img_original_height, img_original_width);


    padded_output[{{},{crop, img_original_height-crop}, {crop, img_original_width - crop}}]:copy(resize_output)
    -- pad left and right
    for i = 1 , crop do
        padded_output[{1,{crop, img_original_height - crop}, i}]:copy(resize_output[{{},1}])
        padded_output[{1,{crop, img_original_height - crop}, img_original_width - i + 1}]:copy(resize_output[{{},resize_width}])
    end

    -- pad top and down    
    for i = 1 , crop do
        padded_output[{1,i, {}}]:copy(padded_output[{1,crop,{}}])
        padded_output[{1,img_original_height - i + 1, {}}]:copy(padded_output[{1,resize_height + crop - 1,{}}])
    end

    return padded_output
end

function metric_error(gtz, z)
    local fmse = torch.mean(torch.pow(gtz - z, 2))
    local fmselog = torch.mean(torch.pow(torch.log(gtz) - torch.log(z), 2))
    local flsi = torch.mean(  torch.pow(torch.log(z)  - torch.log(gtz) + torch.mean(torch.log(gtz) - torch.log(z)) , 2 )  )
    local fabsrel = torch.mean( torch.cdiv(torch.abs( z - gtz ), gtz ))
    local fsqrrel = torch.mean( torch.cdiv(torch.pow( z - gtz ,2), gtz ))

    return fmse, fmselog, flsi, fabsrel, fsqrrel
end

function normalize_output_depth_with_NYU_mean_std( input )
    local std_of_NYU_training = 0.6148231626
    local mean_of_NYU_training = 2.8424594402
    
    local transformed_weifeng_z = input:clone()
    transformed_weifeng_z = transformed_weifeng_z - torch.mean(transformed_weifeng_z);
    transformed_weifeng_z = transformed_weifeng_z / torch.std(transformed_weifeng_z);
    transformed_weifeng_z = transformed_weifeng_z * std_of_NYU_training;
    transformed_weifeng_z = transformed_weifeng_z + mean_of_NYU_training;
    
    -- remove and replace the depth value that are negative
    if torch.sum(transformed_weifeng_z:lt(0)) > 0 then
        -- fill it with the minimum of the non-negative plus a eps so that it won't be 0
        transformed_weifeng_z[transformed_weifeng_z:lt(0)] = torch.min(transformed_weifeng_z[transformed_weifeng_z:gt(0)]) + 0.00001
    end

    return transformed_weifeng_z
end

----------------------------------------------[[
--[[

Main Entry

]]--
------------------------------------------------



cmd = torch.CmdLine()
cmd:text('Options')

cmd:option('-num_iter',1,'number of training iteration')
cmd:option('-prev_model_file','','Absolute / relative path to the previous model file. Resume training from this file')
cmd:option('-vis', false, 'visualize output')
cmd:option('-output_folder','./output_imgs','image output folder')
cmd:option('-mode','validate','mode: test or validate')
cmd:option('-valid_set', '45_NYU_validate_imgs_points_resize_240_320.csv', 'validation file name');
cmd:option('-test_set','654_NYU_MITpaper_test_imgs_orig_size_points.csv', 'test file name');
cmd:option('-crop',10, 'cropping size')
cmd:option('-thresh',-1, 'threhold for determing WKDR. Obtained from validations set.')
cmd_params = cmd:parse(arg)



if cmd_params.mode == 'test' then
    csv_file_name = '../../data/' .. cmd_params.test_set       -- test set
elseif cmd_params.mode == 'validate' then
    csv_file_name = '../../data/' .. cmd_params.valid_set        -- validation set
end
preload_t7_filename = string.gsub(csv_file_name, "csv", "t7")




f=io.open(preload_t7_filename,"r")
if f == nil then
    print('loading csv file...')
    n_sample, data_handle = _read_data_handle( csv_file_name )
    torch.save(preload_t7_filename, data_handle)
else
    io.close(f)
    print('loading pre load t7 file...')
    data_handle = torch.load(preload_t7_filename)
    n_sample = #data_handle
end



print("Hyper params: ")
print("csv_file_name:", csv_file_name);
print("N test samples:", n_sample);
n_iter = math.min( n_sample, cmd_params.num_iter )
print(string.format('n_iter = %d',n_iter))


-- Load the model
prev_model_file = cmd_params.prev_model_file
model = torch.load(prev_model_file)
model:evaluate()
print("Model file:", prev_model_file)


network_input_height = 240
network_input_width = 320
_batch_input_cpu = torch.Tensor(1,3,network_input_height,network_input_width)


n_thresh = 140;
thresh = torch.Tensor(n_thresh);
for i = 1, n_thresh do
    thresh[i] = 0.1 + i * 0.01;
end


local WKDR = torch.Tensor(n_iter, n_thresh):fill(0)
local WKDR_eq = torch.Tensor(n_iter, n_thresh):fill(0)
local WKDR_neq = torch.Tensor(n_iter, n_thresh):fill(0)
local fmse = torch.Tensor(n_iter):fill(0)
local fmselog = torch.Tensor(n_iter):fill(0) 
local flsi = torch.Tensor(n_iter):fill(0) 
local fabsrel = torch.Tensor(n_iter):fill(0) 
local fsqrrel = torch.Tensor(n_iter):fill(0)

-------------------------------
--[[
The validation is done without cropping

The test is done with cropping
]]
-------------------------------
for i = 1, n_iter do   

    -- read image, scale it to the input size
    local img = image.load(data_handle[i].img_filename)
    local img_original_height = img:size(2)
    local img_original_width = img:size(3)


    if cmd_params.mode == 'test' then
        -- only crop it when testing, because there is a white boundary, also need to resize it to network input size
        _batch_input_cpu[{1,{}}]:copy( crop_resize_input(img) )
    elseif cmd_params.mode == 'validate' then
        -- no need to crop it, just resize it, because there is no white boundary in the validation image!
        _batch_input_cpu[{1,{}}]:copy( image.scale(img,network_input_width ,network_input_height))
    end
    
    
    local _single_data = {};
    _single_data[1] = data_handle[i]


    -- forward
    local batch_output = model:forward(_batch_input_cpu:cuda());  
    cutorch.synchronize()
    local temp = batch_output
    if torch.type(batch_output) == 'table' then
        batch_output = batch_output[1]
    end


    local original_size_output = torch.Tensor(1,1,img_original_height, img_original_width)

    if cmd_params.mode == 'test' then                    
        --image.scale(src, width, height, [mode])    Scale it to the original size!
        original_size_output[{1,1,{}}]:copy( inpaint_pad_output_our(batch_output, img_original_width, img_original_height) ) 
        
        -- evaluate on the original size!
        _evaluate_correctness_our(original_size_output, _single_data[1], WKDR[{i,{}}], WKDR_eq[{i,{}}], WKDR_neq[{i,{}}]);


        local gtz_h5_handle = hdf5.open(paths.dirname(data_handle[i].img_filename) .. '/' .. i ..'_depth.h5', 'r')
        local gtz = gtz_h5_handle:read('/depth'):all()
        gtz_h5_handle:close()
        assert(gtz:size(1) == 480)
        assert(gtz:size(2) == 640)
        
        -- transform the output depth with training mean and std
        transformed_weifeng_z_orig_size = normalize_output_depth_with_NYU_mean_std( original_size_output[{1,1,{}}] )

        -- evaluate the data at the cropped area
        local metric_test_crop = 16
        transformed_weifeng_z_orig_size = transformed_weifeng_z_orig_size:sub(metric_test_crop,img_original_height-metric_test_crop,metric_test_crop,img_original_width-metric_test_crop)
        gtz = gtz:sub(metric_test_crop,img_original_height-metric_test_crop,metric_test_crop,img_original_width-metric_test_crop)
        
        -- metric error
        fmse[i], fmselog[i], flsi[i], fabsrel[i], fsqrrel[i] = metric_error(gtz, transformed_weifeng_z_orig_size)

    elseif cmd_params.mode == 'validate' then
        -- resize it to the original input size
        original_size_output[{1,1,{}}]:copy( image.scale(batch_output[{1,1,{}}]:double(), img_original_width ,img_original_height) )
        -- no need to perform padding because the input is not cropped!            
        _evaluate_correctness_our(original_size_output, _single_data[1], WKDR[{i,{}}], WKDR_eq[{i,{}}], WKDR_neq[{i,{}}]);
    end


    collectgarbage()
    collectgarbage()
    collectgarbage()
    collectgarbage()
    collectgarbage()


    if cmd_params.vis then
        local local_image = torch.Tensor(1,img_original_height,img_original_width)
        local local_image2 = torch.Tensor(3,img_original_height,img_original_width)
        local output_image = torch.Tensor(3, img_original_height,img_original_width * 2)


        local_image:copy(original_size_output:double())
        local_image = local_image:add( - torch.min(local_image) )
        local_image = local_image:div( torch.max(local_image:sub(1,-1, 20, img_original_height - 20, 20, img_original_width - 20)) )
        

        output_image[{1,{1,img_original_height},{img_original_width + 1,img_original_width * 2}}]:copy(local_image)
        output_image[{2,{1,img_original_height},{img_original_width + 1,img_original_width * 2}}]:copy(local_image)
        output_image[{3,{1,img_original_height},{img_original_width + 1,img_original_width * 2}}]:copy(local_image)


        local_image2:copy(image.load(data_handle[i].img_filename)) 

        output_image[{{1},{1,img_original_height},{1,img_original_width}}]:copy(local_image2[{1,{}}])
        output_image[{{2},{1,img_original_height},{1,img_original_width}}]:copy(local_image2[{2,{}}])
        output_image[{{3},{1,img_original_height},{1,img_original_width}}]:copy(local_image2[{3,{}}])

        image.save(cmd_params.output_folder.. '/' .. i .. '.png', output_image)   
    end
end


-- get averaged WKDR and so on 
WKDR = torch.mean(WKDR,1)
WKDR_eq = torch.mean(WKDR_eq,1)
WKDR_neq = torch.mean(WKDR_neq,1)
overall_summary = torch.Tensor(n_thresh, 4)


-- find the best threshold on the validation set according to our criteria, and use it as the threshold on the test set.
min_max = 100;
min_max_i = 1;
for i = 1 , n_thresh do
    overall_summary[{i,1}] = thresh[i]
    overall_summary[{i,2}] = WKDR[{1,i}]
    overall_summary[{i,3}] = WKDR_eq[{1,i}]
    overall_summary[{i,4}] = WKDR_neq[{1,i}]
    if math.max(WKDR_eq[{1,i}], WKDR_neq[{1,i}]) < min_max then
        min_max = math.max(WKDR_eq[{1,i}], WKDR_neq[{1,i}])
        min_max_i = i;
    end
end


-- print the final output
if cmd_params.thresh < 0 then
    print(overall_summary)
    print("====================================================================")
    if min_max_i > 1 then
        if min_max_i < n_thresh then
            print(overall_summary[{{min_max_i-1,min_max_i+1},{}}])
        end
    end
else
    print("Result:\n")
    for i = 1 , n_thresh do
        if overall_summary[{i,1}] == cmd_params.thresh then
            print(" Thresh\tWKDR\tWKDR_eq\tWKDR_neq")
            print(overall_summary[{{i},{}}])
        end
    end
end

if cmd_params.mode == 'test' then 
    print("====================================================================")
    print(string.format('rmse:\t%f',math.sqrt(torch.mean(fmse))))
    print(string.format('rmselog:%f',math.sqrt(torch.mean(fmselog))))
    print(string.format('lsi:\t%f',math.sqrt(torch.mean(flsi))))
    print(string.format('absrel:\t%f',torch.mean(fabsrel)))
    print(string.format('sqrrel:\t%f',torch.mean(fsqrrel)))
end

