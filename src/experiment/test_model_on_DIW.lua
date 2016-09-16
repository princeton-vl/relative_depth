require 'nn'
require 'image'
require 'csvigo'
require 'hdf5'






local function _read_data_handle( _filename )        
    -- the file is a csv file
    local csv_file_handle = csvigo.load({path = _filename, mode = 'large'})
    
    local _n_lines = #csv_file_handle;


    
    local _data ={}
    local _line_idx = 1
    local _sample_idx = 0
    while _line_idx <= _n_lines do

        _sample_idx = _sample_idx + 1

        _data[_sample_idx] = {};
        _data[_sample_idx].img_filename = csv_file_handle[ _line_idx ][ 1 ]    
        _data[_sample_idx].n_point = 1
        _data[_sample_idx].img_filename_line_idx = _line_idx;


        _line_idx = _line_idx + _data[_sample_idx].n_point
        _line_idx = _line_idx + 1
    end
    print("number of sample =", _sample_idx);
   
    _data.csv_file_handle = csv_file_handle

    return _sample_idx, _data;
end

local function _read_one_sample(_sample_idx, handle)

    local _data ={}

    local n_point = handle[_sample_idx].n_point
    
    _data.img_filename = handle[_sample_idx].img_filename
    _data.n_point = handle[_sample_idx].n_point
    

    _data.y_A = torch.Tensor(n_point)
    _data.y_B = torch.Tensor(n_point)
    _data.x_A = torch.Tensor(n_point)
    _data.x_B = torch.Tensor(n_point)
    _data.ordianl_relation = torch.Tensor(n_point)   

    
    local _line_idx = handle[_sample_idx].img_filename_line_idx + 1


    for point_idx = 1 , handle[_sample_idx].n_point do
                           
        _data.y_A[point_idx] = tonumber(handle.csv_file_handle[ _line_idx ][ 1 ])
        _data.x_A[point_idx] = tonumber(handle.csv_file_handle[ _line_idx ][ 2 ])
        _data.y_B[point_idx] = tonumber(handle.csv_file_handle[ _line_idx ][ 3 ])
        _data.x_B[point_idx] = tonumber(handle.csv_file_handle[ _line_idx ][ 4 ])

        if _data.y_A[point_idx] == _data.y_B[point_idx] and _data.x_A[point_idx] == _data.x_B[point_idx] then
            assert(false, 'The coordinates shouldn not be equal!!!!');
        end

        local ord = string.sub(handle.csv_file_handle[ _line_idx ][ 5 ],1,1)
         -- Important!
        if ord == '>' then
            _data.ordianl_relation[point_idx] = 1;
        elseif ord == '<' then
            _data.ordianl_relation[point_idx] = -1;    
        elseif ord == '=' then        -- important!
            assert(false, 'Error in _read_one_sample()! The ordinal_relationship should not be equal!!!');
        else
            assert(false, 'Error in _read_one_sample()! The ordinal_relationship does not read correctly!!!!');
        end

        -- print(_data.img_filename)
        -- print(string.format("Original:%d, %d, %d, %d", tonumber(handle.csv_file_handle[ _line_idx ][ 1 ]), tonumber(handle.csv_file_handle[ _line_idx ][ 2 ]), tonumber(handle.csv_file_handle[ _line_idx ][ 3 ]), tonumber(handle.csv_file_handle[ _line_idx ][ 4 ])))
        -- print(string.format("Read : %d, %d, %d, %d", _data.y_A[point_idx], _data.x_A[point_idx], _data.y_B[point_idx], _data.x_B[point_idx]))
        -- print(string.format("relationship: %d", _data.ordianl_relation[point_idx]))
        -- io.read()



        _line_idx = _line_idx + 1            

    end
    
    return _data
end



function file_exist(filename)
    f=io.open(filename,"r")
    if f == nil then
        -- skip it
        return false
    else
        io.close(f)
        return true
    end
end



local function inpaint_pad_output_eigen(output)
    assert(output:size(2) == 109)
    assert(output:size(3) == 147)
    

    -- input height = 240
    -- input width = 320
    -- [12 .. 227] height
    -- [14 .. 305] width

    local resize_height = 227 - 12 + 1
    local resize_width = 305 - 14 + 1
    local resize_output = image.scale( output[{1,{}}]:double(), resize_width, resize_height)
    
    local dst_out_height = 240
    local dst_out_width = 320

    local padded_output1 = torch.Tensor(1, resize_height, dst_out_width);
    padded_output1[{{},{}, {14, 305}}]:copy(resize_output)
    -- pad left 
    for i = 1 , 13 do
        padded_output1[{1,{}, i}]:copy(padded_output1[{1,{},14}])                
    end
    -- pad right
    for i = 306 , 320 do
        padded_output1[{1,{}, i}]:copy(padded_output1[{1,{},305}])                
    end




    local padded_output2 = torch.Tensor(1, dst_out_height, dst_out_width);
    padded_output2[{{},{12, 227}, {}}]:copy(padded_output1)
    -- pad top and down    
    for i = 1 , 11 do
        padded_output2[{1, i, {}}]:copy(padded_output1[{1, 1,{}}])        
    end
    for i = 228, 240 do
        padded_output2[{1, i, {}}]:copy(padded_output1[{1, resize_height,{}}])
    end

    return padded_output2
end




local function _evaluate_correctness(_batch_output, _batch_target, record)

    assert(_batch_target.n_point == 1)
    for point_idx = 1, _batch_target.n_point do

        x_A = _batch_target.x_A[point_idx]
        y_A = _batch_target.y_A[point_idx]
        x_B = _batch_target.x_B[point_idx]
        y_B = _batch_target.y_B[point_idx]

        z_A = _batch_output[{1, 1, y_A, x_A}]
        z_B = _batch_output[{1, 1, y_B, x_B}]
        

        ground_truth = _batch_target.ordianl_relation[point_idx];    -- the ordianl_relation is in the form of 1 and -1

                
        if (z_A - z_B) * ground_truth > 0 then
            if ground_truth > 0 then
                record.n_gt_correct = record.n_gt_correct + 1;
            else 
                record.n_lt_correct = record.n_lt_correct + 1;
            end       
        end

        
        if ground_truth > 0 then
            record.n_gt = record.n_gt + 1;
        elseif ground_truth < 0 then
            record.n_lt = record.n_lt + 1;
        elseif ground_truth == 0 then
            assert(false, 'The input should not contain equal terms!');
        end

        
        if cmd_params.test_model == 'debug' then
            print(string.format('x_A= %d, y_A = %d, z_A = %f',x_A, y_A, z_A))
            print(string.format('x_B= %d, y_B = %d, z_B = %f',x_B, y_B, z_B))
            if ground_truth > 0 then
                print(string.format('ground truth is z_A > z_B. predict is z_A - z_B = %f', z_A - z_B));
            else
                print(string.format('ground truth is z_A < z_B. predict is z_A - z_B = %f', z_A - z_B));
            end
        end
    end        

end



function print_result(record)
    print(string.format('Less_than correct ratio = %f, n_lt_correct = %d, n_lt = %d', record.n_lt_correct / record.n_lt, record.n_lt_correct, record.n_lt))
    print(string.format('Greater_than correct ratio = %f, n_gt_correct = %d, n_gt = %d', record.n_gt_correct / record.n_gt, record.n_gt_correct, record.n_gt))
    print(string.format('Overall correct ratio = %f', (record.n_lt_correct + record.n_gt_correct) / (record.n_gt + record.n_lt)))
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
cmd:option('-test_model','our','eigen  ,our or debug')
cmd:option('-vis', false, 'visualize output')
cmd:option('-output_folder','/scratch/jiadeng_flux/wfchen/depthordereval/relative_depth/script_test_AMT/AMT_qual_result','image output folder')



cmd_params = cmd:parse(arg)




---CSV
csv_file_name = '../../data/DIW_test.csv'       -- test set
print('loading csv file...')
n_sample, data_handle = _read_data_handle( csv_file_name )









print("Hyper params: ")
print("csv_file_name:", csv_file_name);
print("N test samples:", n_sample);


num_iter = math.min(n_sample, cmd_params.num_iter)
print("num_iter:", cmd_params.num_iter);

if cmd_params.test_model == 'debug' then
    print('Debugging..........')
    -- reset debug result
    debug_result = {}
    debug_result.n_gt_correct = 0;
    debug_result.n_gt = 0;

    debug_result.n_lt_correct = 0;
    debug_result.n_lt = 0;

    debug_result.n_eq_correct = 0;
    debug_result.n_eq = 0;


    for i = 1, num_iter do   
        print(i)    
        -- read the input image
        local png_filename = string.gsub(data_handle[i].img_filename, '.thumb', '.png')
        local orig_img = image.load(png_filename)
        local orig_height = orig_img:size(2)
        local orig_width = orig_img:size(3)

        -- read target data
        local _single_data = {};
        _single_data[1] = _read_one_sample(i, data_handle, orig_height, orig_width) --_read_one_sample(_sample_idx, handle, orig_height, orig_width)


        local orig_size_output = torch.Tensor(1, 1, orig_height, orig_width);
        


        x_A = _single_data[1].x_A[1]
        y_A = _single_data[1].y_A[1]
        x_B = _single_data[1].x_B[1]
        y_B = _single_data[1].y_B[1]        
        ground_truth = _single_data[1].ordianl_relation[1];    -- the ordianl_relation is in the form of 1 and -1               
        if ground_truth > 0 then
            orig_size_output[{1,1,y_A,x_A}] = 0;
            orig_size_output[{1,1,y_B,x_B}] = 1;
        elseif ground_truth < 0  then
            orig_size_output[{1,1,y_A,x_A}] = 0;
            orig_size_output[{1,1,y_B,x_B}] = 1;
        end
                       
        -- evaluate
        _evaluate_correctness(orig_size_output, _single_data[1], debug_result);
        

        if math.fmod(i, 100) == 0 then
            print_result(debug_result)
        end
    end
end

if cmd_params.test_model == 'eigen' then
    print('Testing on Eigen......................')
    -- reset eigen result
    eigen_result = {}
    eigen_result.n_gt_correct = 0;
    eigen_result.n_gt = 0;

    eigen_result.n_lt_correct = 0;
    eigen_result.n_lt = 0;

    eigen_result.n_eq_correct = 0;
    eigen_result.n_eq = 0;


    for i = 1, num_iter do   
        print(i)    
        -- read the input image
        local thumb_filename = data_handle[i].img_filename
        local orig_img = image.load(thumb_filename)
        local orig_height = orig_img:size(2)
        local orig_width = orig_img:size(3)

        -- read target data
        local _single_data = {};
        _single_data[1] = _read_one_sample(i, data_handle, orig_height, orig_width) --_read_one_sample(_sample_idx, handle, orig_height, orig_width)


        local orig_size_output = torch.Tensor(1, 1, orig_height, orig_width);
        local eigen_result_filename = string.gsub(data_handle[i].img_filename, '.thumb', '_eigen15_result.h5')
        print('reading ', eigen_result_filename);
        if file_exist(eigen_result_filename) then
            local myFile = hdf5.open( eigen_result_filename, 'r')
            local recovered_depth = myFile:read('/result'):all()
            myFile:close()

            -- pad and resize back to 240 x 320
            local padded_320_240_eigen_depth = inpaint_pad_output_eigen(recovered_depth)

            -- resize to the original scale
            recovered_depth = image.scale(padded_320_240_eigen_depth, orig_width, orig_height)     --image.scale(src, width, height, [mode])
            orig_size_output[{1,1,{}}]:copy(recovered_depth)

            -- -- the input should be a two dimensional tensor
            -- local vis_img = recovered_depth:clone()
            -- vis_img = vis_img:add( - torch.min(vis_img) )
            -- vis_img = vis_img:div( torch.max(vis_img) )   
            -- image.save( i .. '_debug_depth.png', vis_img) 
            -- io.read()
            -- image.save( i .. '_debug_orig.png', orig_img)
            
            -- evaluate
            _evaluate_correctness(orig_size_output, _single_data[1], eigen_result);
        end

        if math.fmod(i, 100) == 0 then
            print_result(eigen_result)
        end
    end
end



if cmd_params.test_model == 'our' then

    require 'cudnn'
    require 'cunn'
    require 'cutorch'

    -- hyper params
    _network_input_width = 320
    _network_input_height = 240

    --Load Model
    prev_model_file = cmd_params.prev_model_file
    model = torch.load(prev_model_file)
    model:evaluate()


    print("Model file:", prev_model_file)


    -- reset our result
    our_result = {}
    our_result.n_gt_correct = 0;
    our_result.n_gt = 0;

    our_result.n_lt_correct = 0;
    our_result.n_lt = 0;

    our_result.n_eq_correct = 0;
    our_result.n_eq = 0;

    --buffer
    _batch_input_cpu = torch.Tensor(1,3,_network_input_height,_network_input_width)

    --main loop
    for i = 1, num_iter do   
        print(i)
        
        -- read the input image
        local thumb_filename = data_handle[i].img_filename
        local orig_img = image.load(thumb_filename)
        local orig_height = orig_img:size(2)
        local orig_width = orig_img:size(3)


        

        print('Processing sample ' .. thumb_filename);

        -- scale the image to the network input size!!!!
        local img = image.scale(orig_img, _network_input_width, _network_input_height)        --image.scale(src, width, height, [mode])
        

        -- check if it's gray scale, if so, make it multi channel zero
        if img:size(1) == 1 then
            print(data_handle[i].img_filename, ' is gray')
            _batch_input_cpu[{1,1,{}}]:copy(img);    -- Note that the image read is in the range of 0~1
            _batch_input_cpu[{1,2,{}}]:copy(img);    
            _batch_input_cpu[{1,3,{}}]:copy(img);    
        else
            _batch_input_cpu[{1,{}}]:copy(img);    -- Note that the image read is in the range of 0~1    
        end
        img = nil


        -- read target data        
        local _single_data = {};
        _single_data[1] = _read_one_sample(i, data_handle, orig_height, orig_width) --_read_one_sample(_sample_idx, handle, orig_height, orig_width)


        


        -- forward
        _processed_input = _batch_input_cpu
        if torch.type(_processed_input) ~= 'table' then
            batch_output = model:forward(_processed_input:cuda());  
        else
            batch_output = model:forward(_processed_input);  
        end
        cutorch.synchronize()


        -- test the correctness on the original size!!!!
        local orig_size_output = torch.Tensor(1, 1, orig_height, orig_width);
        orig_size_output[{1,1,{}}]:copy(image.scale(batch_output[{1,1,{}}]:double(), orig_width, orig_height))   --image.scale(src, width, height, [mode])

        -- evaluate
        _evaluate_correctness(orig_size_output, _single_data[1], our_result);

        if math.fmod(i, 100) == 0 then
            print_result(our_result)
        end
        

        collectgarbage()
        collectgarbage()
        collectgarbage()
        collectgarbage()
        collectgarbage()


        if cmd_params.vis then

            local_image = torch.Tensor(1, orig_height, orig_width)
            local_image2 = torch.Tensor(3, orig_height, orig_width)
            output_image = torch.Tensor(3, orig_height, orig_width * 3)

            
            local_image:copy(orig_size_output:double())
            local_image = local_image:add( - torch.min(local_image) )
            local_image = local_image:div( torch.max(local_image) )
            

            output_image[{1,{1,orig_height},{orig_width + 1, 2 * orig_width}}]:copy(local_image)
            output_image[{2,{1,orig_height},{orig_width + 1, 2 * orig_width}}]:copy(local_image)
            output_image[{3,{1,orig_height},{orig_width + 1, 2 * orig_width}}]:copy(local_image)


            img = image.load(thumb_filename) 
            if img:size(1) == 1 then
                local_image2[{1,{}}]:copy(img);    -- Note that the image read is in the range of 0~1
                local_image2[{1,{}}]:copy(img);    
                local_image2[{1,{}}]:copy(img);    
            else
                local_image2:copy(img)
            end    

            output_image[{{1},{1,orig_height},{1,orig_width}}]:copy(local_image2[{1,{}}])
            output_image[{{2},{1,orig_height},{1,orig_width}}]:copy(local_image2[{2,{}}])
            output_image[{{3},{1,orig_height},{1,orig_width}}]:copy(local_image2[{3,{}}])

            
            -- eigen result
            eigen_result_filename = string.gsub(data_handle[i].img_filename, '.thumb', '_eigen15_result.h5')
            if file_exist(eigen_result_filename) then
                local myFile = hdf5.open( eigen_result_filename, 'r')
                local recovered_depth = myFile:read('/result'):all()
                myFile:close()
                
                -- pad and resize back to 240 x 320
                recovered_depth = inpaint_pad_output_eigen(recovered_depth)

                -- resize it to original size                
                recovered_depth = image.scale(recovered_depth, orig_width, orig_height)     --image.scale(src, width, height, [mode])

                -- visualize it 
                recovered_depth = recovered_depth:add( - torch.min(recovered_depth) )
                recovered_depth = recovered_depth:div( torch.max(recovered_depth) )              
                

                output_image[{{1},{1, orig_height},{2*orig_width + 1, orig_width*3}}]:copy(recovered_depth)
                output_image[{{2},{1, orig_height},{2*orig_width + 1, orig_width*3}}]:copy(recovered_depth)
                output_image[{{3},{1, orig_height},{2*orig_width + 1, orig_width*3}}]:copy(recovered_depth)       
            end

            image.save( cmd_params.output_folder .. '/' .. i .. '.png', output_image) 
        elseif cmd_params.vis and cmd_params.test_model == 'eigen' then
            print("Not visualizing anything because the test model is not our.");
        end
    end
end




print(string.format("Summary:========================================================================="))
if cmd_params.test_model == 'our' then
    print_result(our_result)
elseif cmd_params.test_model == 'eigen' then
    print_result(eigen_result)
elseif cmd_params.test_model == 'debug' then
    print_result(debug_result)
end
