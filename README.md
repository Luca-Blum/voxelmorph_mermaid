# A review of the voxelmorph and mermaid libraries for medical image registration

This repository shows the work done in a Seminar in FS 2021 at ETH Zürich under the supervision of Professor Ender Konukoglu.
Two popular learning-based medical image registration frameworks ([voxelmorph](https://github.com/voxelmorph/voxelmorph)
and [mermaid](https://github.com/uncbiag/registration)) were reviewed. Further, [voxelmorph](https://github.com/voxelmorph/voxelmorph)
was trained on intra-modal and inter-modal MR brain images and the corresponding registration performances were evaluated.

## [Report](report.pdf)

This is the final report that summarizes the two frameworks and analysis the results from the experiments

## [main.py](main.py)

This is the main script that was used for the experiments. It assumes a paths.json2 file that consist of a dictionary
which contains the directory for the original data {brain_t1, brain_t2} and a 'home' directory where the results get stored. 
T1 weighted brain MRI and T2 weighted brain MRI are supported. The original data folder is expected to contain *nii.gz files
There are three possible cases.

1. t1: &nbsp;&nbsp;&nbsp;&nbsp;  intra-modal T1w MRI
2. t2:  &nbsp;&nbsp;&nbsp;&nbsp; intra-modal T2w MRI
3. t1t2: &nbsp; inter-modal T1w-T2w MRI

The base usage is:

python main.py -c case -pp True

This will preprocess the original data and stores the result in 'home/case/results'. Afterwards a default voxelmorph model is
trained. In 'home/case/results' a folder for the weights of the voxelmorph model get stored. For all three cases the preprocessing applies
skull stripping with [HD-BET](https://github.com/MIC-DKFZ/HD-BET) and then affine alignment to a mean image and nyul intensity
normalization with [intensity-normalization](https://github.com/jcreinhold/intensity-normalization). The main script
assumes that this two frameworks are installed. If the data does not need to be preprocessed,
than the following can be used:

python main.py -c case 

This expects that the original data is stored in 'home/case/results'. If you want to train a previously trained
voxelmorph model further than you can give the path to the folder with the corresponding weights by using the -rp flag:

python main.py -c case -rp path/to/folder/with/weights

It takes the latest weights file if there are multiple weights in the folder. Also you can specify if L2 or L1 regularization
should be applied by the use of the -l flag {l1,l2}

python main.py -c case -l l2

## [main.sh](main.sh)

This script can be used to submit a job to a slurm cluster. The path for the conda environment needs to be adapted to 
your system.

## Setup

A conda environment is provided in the file environment.yml. You can activate it by using

conda env create -f environment.yml
 
This should setup all dependencies except [HD-BET](https://github.com/MIC-DKFZ/HD-BET) and 
[intensity-normalization](https://github.com/jcreinhold/intensity-normalization) which should be installed like 
described in the corresponding linked repositories.
