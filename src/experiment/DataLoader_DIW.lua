require('../common/NYU_params')
require('./DataPointer')
require 'image'
require 'xlua'
require 'csvigo'





local DataLoader = torch.class('DataLoader')

function DataLoader:__init(relative_depth_filename)     
    print(">>>>>>>>>>>>>>>>> Using DataLoader DIW")   
    self:parse_depth(relative_depth_filename)    
    
    self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)
    


    print(string.format('DataLoader init: \n \t%d relative depth samples', self.n_relative_depth_sample))
end


local function parse_DIW_csv(_filename)
    -- the file is a csv file
    local csv_file_handle = csvigo.load({path = _filename, mode = 'large'})
    
    local _n_lines = #csv_file_handle

    local _handle ={}
    local _line_idx = 1
    local _sample_idx = 0
    while _line_idx <= _n_lines do

        _sample_idx = _sample_idx + 1

        _handle[_sample_idx] = {};
        _handle[_sample_idx].img_filename = csv_file_handle[ _line_idx ][ 1 ]    
        _handle[_sample_idx].n_point = 1
        _handle[_sample_idx].img_filename_line_idx = _line_idx;


        _line_idx = _line_idx + _handle[_sample_idx].n_point
        _line_idx = _line_idx + 1
    end
    
    -- this variable keeps the handle to the csv file
    _handle.csv_file_handle = csv_file_handle

    -- print(_handle[_sample_idx].img_filename, _handle[_sample_idx].n_point, _handle[_sample_idx].img_filename_line_idx)
    -- io.read()
    print(string.format("%s: number of sample = %d", _filename, _sample_idx))
    

    return _handle;
end

local function parse_one_coordinate_line(csv_file_handle, _line_idx)
    -- parse the coordinate line
    local orig_img_width  = tonumber(csv_file_handle[ _line_idx ][ 6 ])
    local orig_img_height = tonumber(csv_file_handle[ _line_idx ][ 7 ])

    local y_A_float_orig = tonumber(csv_file_handle[ _line_idx ][ 1 ]) / orig_img_height 
    local x_A_float_orig = tonumber(csv_file_handle[ _line_idx ][ 2 ]) / orig_img_width
    local y_B_float_orig = tonumber(csv_file_handle[ _line_idx ][ 3 ]) / orig_img_height
    local x_B_float_orig = tonumber(csv_file_handle[ _line_idx ][ 4 ]) / orig_img_width

    local y_A = math.min(g_input_height, math.max(1,math.floor( y_A_float_orig * g_input_height )))
    local x_A = math.min(g_input_width,math.max(1,math.floor( x_A_float_orig * g_input_width)))
    local y_B = math.min(g_input_height,math.max(1,math.floor( y_B_float_orig * g_input_height)))
    local x_B = math.min(g_input_width,math.max(1,math.floor( x_B_float_orig * g_input_width)))

    -- avoid the situation where both points are at the same location after rescaling
    if (y_A == y_B) and (x_A == x_B) then
        -- squeeze it a little bit
        if y_A > 1 then
            y_A = y_A - 1;
        else
            y_A = y_A + 1;
        end
    end

    local ord = string.sub(csv_file_handle[ _line_idx ][ 5 ],1,1)
     -- Important!
    if ord == '>' then
        ord = 1;
    elseif ord == '<' then
        ord = -1;    
    elseif ord == '=' then        -- important!
        assert(false, 'Error in _read_one_sample()! The ordinal_relationship should never be = !!!!');
    else
        assert(false, 'Error in _read_one_sample()! The ordinal_relationship does not read correctly!!!!');
    end
    
    -- print(string.format("Original:%d, %d, %d, %d", tonumber(csv_file_handle[ _line_idx ][ 1 ]), tonumber(csv_file_handle[ _line_idx ][ 2 ]), tonumber(csv_file_handle[ _line_idx ][ 3 ]), tonumber(csv_file_handle[ _line_idx ][ 4 ])))
    -- print(string.format("Size    :%d, %d", orig_img_width, orig_img_height))
    -- print(string.format("Float  : %.3f, %.3f, %.3f, %.3f", y_A_float_orig, x_A_float_orig, y_B_float_orig, x_B_float_orig))
    -- print(string.format("Scaled : %d, %d, %d, %d", y_A, x_A, y_B, x_B, ord))
    -- print(string.format("relationship: %d", ord))
    -- io.read()

    return y_A, x_A, y_B, x_B, ord
end


function DataLoader:parse_depth(relative_depth_filename)
    if relative_depth_filename ~= nil then
        -- parse csv file
        self.relative_depth_handle = parse_DIW_csv(relative_depth_filename)
    else
        self.relative_depth_handle = {}        
    end

    -- update the number of samples
    self.n_relative_depth_sample = #self.relative_depth_handle
end


function DataLoader:close()
end

local function mixed_sample_strategy1(batch_size)       -- to do
    local n_depth =  torch.random(0,batch_size)     

    return n_depth, batch_size - n_depth
end

local function mixed_sample_strategy2(batch_size)
    local n_depth =  math.ceil(batch_size / 2)

    return n_depth, batch_size - n_depth
end


_batch_target_relative_depth_gpu = {};

for i = 1 , g_args.bs  do                                 -- to test
    _batch_target_relative_depth_gpu[i] = {}
    _batch_target_relative_depth_gpu[i].y_A = torch.CudaTensor()
    _batch_target_relative_depth_gpu[i].x_A = torch.CudaTensor()
    _batch_target_relative_depth_gpu[i].y_B = torch.CudaTensor()
    _batch_target_relative_depth_gpu[i].x_B = torch.CudaTensor()
    _batch_target_relative_depth_gpu[i].ordianl_relation = torch.CudaTensor()

end


function DataLoader:load_indices( depth_indices )
    local n_depth
    if depth_indices ~= nil then
        n_depth = depth_indices:size(1)
    else
        n_depth = 0
    end

    local batch_size = n_depth

    local color = torch.Tensor();    
    color:resize(batch_size, 3, g_input_height, g_input_width); 


    _batch_target_relative_depth_gpu.n_sample = n_depth


    local csv_file_handle = self.relative_depth_handle.csv_file_handle
    -- Read the relative depth data
    for i = 1, n_depth do    
        
        local chosen_idx = depth_indices[i]
        local img_name = self.relative_depth_handle[chosen_idx].img_filename
        local n_point = self.relative_depth_handle[chosen_idx].n_point

        -- print(string.format("Loading %s", img_name))
        
        -- read the input image
        local img = image.scale(image.load(img_name), g_input_width, g_input_height)        --image.scale(src, width, height, [mode])       
        if img:size(1) == 1 then
            print(img_name, ' is gray')
            color[{i,1,{}}]:copy(img);    -- Note that the image read is in the range of 0~1
            color[{i,2,{}}]:copy(img);    
            color[{i,3,{}}]:copy(img);    
        else
            color[{i,{}}]:copy(img);    -- Note that the image read is in the range of 0~1    
        end


        -- Read in the relative depth annotation. Pay attention to the order!!!!        
        _batch_target_relative_depth_gpu[i].y_A:resize(n_point)
        _batch_target_relative_depth_gpu[i].x_A:resize(n_point)
        _batch_target_relative_depth_gpu[i].y_B:resize(n_point)
        _batch_target_relative_depth_gpu[i].x_B:resize(n_point)
        _batch_target_relative_depth_gpu[i].ordianl_relation:resize(n_point)
        _batch_target_relative_depth_gpu[i].n_point = n_point
        

        local _line_idx = self.relative_depth_handle[chosen_idx].img_filename_line_idx + 1
        -- print(string.format("Loading: %s", img_name))
        -- print(string.format("n_point: %d", _batch_target_relative_depth_gpu[i].n_point))
        local y_A, x_A, y_B, x_B, ord = parse_one_coordinate_line(csv_file_handle, _line_idx)

        _batch_target_relative_depth_gpu[i].y_A[1] = y_A
        _batch_target_relative_depth_gpu[i].x_A[1] = x_A
        _batch_target_relative_depth_gpu[i].y_B[1] = y_B
        _batch_target_relative_depth_gpu[i].x_B[1] = x_B
        _batch_target_relative_depth_gpu[i].ordianl_relation[1] = ord

        -- -- for debug        
        -- print(img_name)
        -- for k = 1, n_point do    
        --     print(string.format("%d,%d,%d,%d,%d", _batch_target_relative_depth_gpu[i].y_A[k], _batch_target_relative_depth_gpu[i].x_A[k], _batch_target_relative_depth_gpu[i].y_B[k], _batch_target_relative_depth_gpu[i].x_B[k], _batch_target_relative_depth_gpu[i].ordianl_relation[k]))
        -- end
        -- io.read()
    end

    return color:cuda(), _batch_target_relative_depth_gpu
end


function DataLoader:load_next_batch(batch_size)

    -- Obtain the indices for each group of data
    local depth_indices = self.data_ptr_relative_depth:load_next_batch(batch_size)

    return self:load_indices( depth_indices )
end



function DataLoader:reset()
    self.current_pos = 1
end
