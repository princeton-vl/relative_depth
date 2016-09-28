# Single-Image Depth Perception in the Wild

Code for reproducting the results in the follwing paper:

**Single-Image Depth Perception in the Wild,**
Weifeng Chen, Zhao Fu, Dawei Yang, Jia Deng
Neural Information Processing Systems (NIPS), 2016.

Please check out the [project site](http://www-personal.umich.edu/~wfchen/depth-in-the-wild/)  for more details.


# Setup

1. Install the Torch 7 framework as described in http://torch.ch/docs/getting-started.html#_. Please make sure that you have the `cudnn`, `hdf5` and `csvigo` modules installed.

2. Clone this repo.

		git clone https://github.com/wfchen-umich/relative_depth.git

3. Download and extract the DIW dataset from the [project site](http://www-personal.umich.edu/~wfchen/depth-in-the-wild/). Download and extract `DIW_test.tar.gz` and `DIW_train_val.tar.gz` into 2 folders. Run the following command to download and extract `DIW_Annotations.tar.gz`. Then modify the filepath to images in `DIW_test.csv`, `DIW_train.csv` and `DIW_val.csv` to be the absolute file path where you extracted `DIW_test.tar.gz` and `DIW_train_val.tar.gz`. 

		cd relative_depth
		mkdir data
		cd data
		wget https://vl-lab.eecs.umich.edu/data/nips2016/DIW_Annotations_splitted.tar.gz
		tar -xzf DIW_Annotations_splitted.tar.gz
		rm DIW_Annotations_splitted.tar.gz
		

# Training and evaluating the networks

## Testing on pre-trained models 

Please first run the following commands to download the test data from our processed NYU dataset and the pre-trained models:

	cd relative_depth
	wget https://vl-lab.eecs.umich.edu/data/nips2016/data.tar.gz
	tar -xzf data.tar.gz
	rm data.tar.gz
	cd data
	python convert_csv_2_h5.py -i 750_train_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv
	python convert_csv_2_h5.py -i 45_validate_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv
	
	cd ../src
	mkdir results
	cd results
	wget https://vl-lab.eecs.umich.edu/data/nips2016/hourglass3.tar.gz
	tar -xzf hourglass3.tar.gz
	rm hourglass3.tar.gz

	

Then change directory into `/relative_depth/src/experiment`.

1. To evaluate the pre-trained model ***Ours***(model trained on the NYU labeled training subset) on the NYU dataset, run the following command:

		th test_model_on_NYU.lua -num_iter 1000 -prev_model_file ../results/hourglass3/NYU_795_800_c9_1e-3/Best_model_period1.t7 -test_set 654_NYU_MITpaper_test_imgs_orig_size_points.csv -mode test -thresh 0.9
			
2. To evaluate the pre-trained model ***Ours_Full***(model trained on the full NYU training set) on the NYU dataset, run the following command:

		th test_model_on_NYU.lua -num_iter 1000 -prev_model_file ../results/hourglass3/1e-3_Drop_205315_NYU_fs_c9/Best_model_period1.t7 -test_set 654_NYU_MITpaper_test_imgs_orig_size_points.csv -mode test -thresh 0.32

3. To evaluate the pre-trained model ***Ours_DIW***(our network trained from scratch on DIW) on the DIW dataset, run the following script:

		th test_model_on_DIW.lua -num_iter 90000 -prev_model_file ../results/hourglass3/AMT_from_scratch_1e-4_release/Best_model_period1.t7 -test_model our

4. To evaluate the trained model ***Ours_NYU_DIW***(our network pre-trained on NYU and fine-tuned on DIW) on the DIW dataset, run the following script:

		th test_model_on_DIW.lua -num_iter 90000 -prev_model_file ../results/hourglass3/AMT_from_205315_1e-4_release/Best_model_period2.t7 -test_model our


## Training 

Please first change directory into `/relative_depth/src/experiment`.

To train the model ***Ours***(model trained on the NYU labeled training subset), please run the following command:

	th main.lua -lr 0.001 -bs 4 -m hourglass3 -it 100000 -t_depth_file 750_train_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv -v_depth_file 45_validate_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv -rundir ../results/hourglass3/Ours


To train the model ***Ours_DIW***(our network trained from scratch on DIW), please run the following command:

	th main.lua -diw -lr 0.000100 -bs 4 -m hourglass3 -it 200000 -t_depth_file DIW_train.csv -v_depth_file DIW_val.csv -rundir ../results/hourglass3/Ours_DIW

 
To train the model ***Ours_NYU_DIW***(our network pre-trained on NYU and fine-tuned on DIW), please run the following command:

	cd relative_depth/src/results/hourglass3/
	mkdir Ours_NYU_DIW
	cp 1e-3_Drop_205315_NYU_fs_c9/Best_model_period1.t7 Ours_NYU_DIW/205315_Best_model_period1.t7
	cd ../../experiment/
	th main.lua -diw -lr 0.000100 -bs 4 -m hourglass3 -it 200000 -t_depth_file DIW_train.csv -v_depth_file DIW_val.csv -start_from 205315_Best_model_period1.t7 -rundir ../results/hourglass3/Ours_NYU_DIW/




