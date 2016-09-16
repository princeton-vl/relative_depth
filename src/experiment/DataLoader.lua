require('../common/NYU_params')
require('./DataPointer')
require 'image'
require 'xlua'






local DataLoader = torch.class('DataLoader')

function DataLoader:__init(relative_depth_filename)    
    print(">>>>>>>>>>>>>>>>> Using DataLoader")       
    self:parse_depth(relative_depth_filename)    
    
    self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)


    print(string.format('DataLoader init: \n \t%d relative depth samples \n ', self.n_relative_depth_sample))
end

local function parse_relative_depth_line(line)    
    local splits = line:split(',')
    local sample = {};
    sample.img_filename = splits[ 1 ]    
    sample.n_point = tonumber(splits[ 3 ])

    return sample
end


local function parse_csv(filename, parsing_func)
    local _handle = {}

    -- take care of the case where filename is a nil, i.e., no training data
    if filename == nil then
        return _handle
    end

    -- read the number of lines
    local _n_lines = 0
    for _ in io.lines(filename) do
      _n_lines = _n_lines + 1
    end

    -- read in the image name address
    local csv_file_handle = io.open(filename, 'r');    
    local _sample_idx = 0
    while _sample_idx < _n_lines do

        local this_line = csv_file_handle:read()
        _sample_idx = _sample_idx + 1
        
        _handle[_sample_idx] = parsing_func(this_line)    
        
    end    
    csv_file_handle:close();

    return _handle
end

function DataLoader:parse_depth(relative_depth_filename)
    if relative_depth_filename ~= nil then
        -- the file is a csv file    
        local _simplified_relative_depth_filename = string.gsub(relative_depth_filename, ".csv", "_name.csv");

        -- simplify the csv file into just address lines
        if not paths.filep(_simplified_relative_depth_filename) then
            local command = 'grep \'.png\' ' .. relative_depth_filename .. ' > ' .. _simplified_relative_depth_filename
            print(string.format("executing: %s", command))
            
            os.execute(command)
        else
            print(_simplified_relative_depth_filename , " already exists.")
        end

        self.relative_depth_handle = parse_csv(_simplified_relative_depth_filename, parse_relative_depth_line)    

        -- the handle for the relative depth point pairs                to do 
        local hdf5_filename = string.gsub(relative_depth_filename, ".csv", ".h5");
        self.relative_depth_handle.hdf5_handle = hdf5.open(hdf5_filename,'r')  
    else
        self.relative_depth_handle = {}        
    end
    

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


function DataLoader:load_indices( depth_indices)
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

    -- Read the relative depth data
    for i = 1, n_depth do    
        
        local idx = depth_indices[i]
        local img_name = self.relative_depth_handle[idx].img_filename
        local n_point = self.relative_depth_handle[idx].n_point

        -- print(string.format("Loading %s", img_name))
        
        -- read the input image
        color[{i,{}}]:copy(image.load(img_name));    -- Note that the image read is in the range of 0~1


        -- relative depth
        local _hdf5_offset = 5 * (idx - 1) + 1        
        local _this_sample_hdf5  = self.relative_depth_handle.hdf5_handle:read('/data'):partial({_hdf5_offset, _hdf5_offset + 4}, {1, n_point})

        assert(_this_sample_hdf5:size(1) == 5)
        assert(_this_sample_hdf5:size(2) == n_point)

        -- Pay attention to the order!!!!        
        _batch_target_relative_depth_gpu[i].y_A:resize(n_point):copy(_this_sample_hdf5[{1,{}}])         -- to check if is correct
        _batch_target_relative_depth_gpu[i].x_A:resize(n_point):copy(_this_sample_hdf5[{2,{}}])
        _batch_target_relative_depth_gpu[i].y_B:resize(n_point):copy(_this_sample_hdf5[{3,{}}])
        _batch_target_relative_depth_gpu[i].x_B:resize(n_point):copy(_this_sample_hdf5[{4,{}}])
        _batch_target_relative_depth_gpu[i].ordianl_relation:resize(n_point):copy(_this_sample_hdf5[{5,{}}])
        _batch_target_relative_depth_gpu[i].n_point = n_point

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

    
    local depth_indices = self.data_ptr_relative_depth:load_next_batch(batch_size)

    return self:load_indices( depth_indices )
end



function DataLoader:reset()
    self.current_pos = 1
end
