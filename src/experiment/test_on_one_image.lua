require 'nn'
require 'image'
require 'csvigo'
require 'hdf5'



require 'cudnn'
require 'cunn'
require 'cutorch'


cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-prev_model_file','','Absolute / relative path to the previous model file. Resume training from this file')
cmd:option('-input_image','','path to the input image')
cmd:option('-output_image','', 'path to the output image')
cmd_params = cmd:parse(arg)



-- hyper params
_network_input_width = 320
_network_input_height = 240

--Load Model
prev_model_file = cmd_params.prev_model_file
model = torch.load(prev_model_file)
model:evaluate()


print("Model file:", prev_model_file)


--buffer
_batch_input_cpu = torch.Tensor(1,3,_network_input_height,_network_input_width)



-------------------------------------------------------------------------------    
-- read the input image
local thumb_filename = cmd_params.input_image
local orig_img = image.load(thumb_filename)
local orig_height = orig_img:size(2)
local orig_width = orig_img:size(3)


print('Processing sample ' .. thumb_filename);
-- scale the image to the network input size!!!!
local img = image.scale(orig_img, _network_input_width, _network_input_height)        --image.scale(src, width, height, [mode])

-- check if it's gray scale, if so, make it multi channel zero
if img:size(1) == 1 then
    print(thumb_filename, ' is gray')
    _batch_input_cpu[{1,1,{}}]:copy(img);    -- Note that the image read is in the range of 0~1
    _batch_input_cpu[{1,2,{}}]:copy(img);    
    _batch_input_cpu[{1,3,{}}]:copy(img);    
else
    _batch_input_cpu[{1,{}}]:copy(img);    -- Note that the image read is in the range of 0~1    
end
img = nil


-- forward
_processed_input = _batch_input_cpu
if torch.type(_processed_input) ~= 'table' then
    batch_output = model:forward(_processed_input:cuda());  
else
    batch_output = model:forward(_processed_input);  
end
cutorch.synchronize()

-- resize back to original size
local orig_size_output = torch.Tensor(1, 1, orig_height, orig_width);
orig_size_output[{1,1,{}}]:copy(image.scale(batch_output[{1,1,{}}]:double(), orig_width, orig_height))   --image.scale(src, width, height, [mode])


collectgarbage()


---------------------------------------------------
-- visualize the result
    local_image = torch.Tensor(1, orig_height, orig_width)
    local_image2 = torch.Tensor(3, orig_height, orig_width)
    output_image = torch.Tensor(3, orig_height, orig_width * 2)

    
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
    
    image.save( cmd_params.output_image, output_image) 




