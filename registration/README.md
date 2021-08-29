# Code for experiments

The goal was to train the voxelmorph model for intra-modal volumetric T1-weighted MR brain images, intra-modal 
volumetric T2-weighted MR brain images and inter-modal volumetric T1-weighted and T2-weighted MR brain images. 
Additionally, the corresponding performance was evaluated.

## [main.py](main.py)

This is the main script that was used for the experiments. It assumes a paths.json2 file that consist of a dictionary
which contains the directory for the original data {brain_t1, brain_t2} and a 'home' directory where the results 
get stored. T1 weighted brain MRI and T2 weighted brain MRI are supported. The original data folder is expected to 
contain *nii.gz files.

For registration there are three possible cases.

- t1: &nbsp;&nbsp;&nbsp;&nbsp;  intra-modal T1w MRI
- t2:  &nbsp;&nbsp;&nbsp;&nbsp; intra-modal T2w MRI
- t1t2: &nbsp; inter-modal T1w-T2w MRI

The base usage is:

- python3 main.py -c --case -pp True

This will preprocess the original data and stores the result in 'home/case/results'. Afterwards a default voxelmorph 
model is trained. In 'home/case/results' a folder for the weights of the voxelmorph model gets stored. For all three 
cases the preprocessing applies skull stripping with [HD-BET](https://github.com/MIC-DKFZ/HD-BET) and then affine 
alignment to a mean image and nyul intensity normalization with 
[intensity-normalization](https://github.com/jcreinhold/intensity-normalization). The main script assumes that 
this two frameworks are installed. Inter-modal T1w-T2w MRI registration will merge the corresponding T1w and T2w 
dataset into one dataset. If the data does not need to be preprocessed, then the following can be used:

- python3 main.py -c --case 

This expects that the data is stored in 'home/case/results'. 

If you want to train a previously trained voxelmorph model further than you can give the path to the folder with the 
corresponding weights by using the -rp flag:

- python3 main.py -c --case -rp path/to/folder/with/weights

It takes the latest weights file if there are multiple weights in the folder. 

Also, you can specify if L2 or L1 regularization
should be applied by the use of the -l flag {'l1', 'l2'}.

- python3 main.py -c --case -l l2

For evaluation the following command can be used:

- python3 main.py -c --case -e True

This creates various plots which gets stored in 'home/case/plots'. Note that if the -e flag is set to True, 
preprocessing and training is always skipped. Therefore

- python3 main.py -c --case -pp True -e True

will NOT preprocess the data, train a model and then evaluate it. Rather it just evaluates the latest model in 
'home/case/results' and if there are multiple subsequently trained models it creates a plot of the training and testing
loss over all models. If you want to evaluate a specific model you can specify the path to the model with the -ep flag

- python3 main.py c --case -e True -ep path/to/folder/with/weights

## [main.sh](main.sh)

This script can be used to submit a job to a slurm cluster. The path for the conda environment needs to be adapted to 
your system. Additionally, check that the correct conda environment (seminar) is loaded. Creating the environment is 
described below. Also, do not forget to create a log folder.

## Setup

A conda environment is provided in the file 'environment.yml'. You can activate it by using

- conda env create -f environment.yml
 
This setups all dependencies except [HD-BET](https://github.com/MIC-DKFZ/HD-BET) which should be installed like 
described in the corresponding linked repository.

## [final_weights](final_weights)

This folder contains the final weights of the voxelmorph model. The default architecture was used.

    nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

where 'vol_shape' should be compatible with the first layer.