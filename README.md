# IntelligentCarpet

## Dataset
Dataset can be found here: https://www.dropbox.com/sh/rzuzasgf6dntnrb/AACsWKeXpCOzZyYlan1cmNiha?dl=0

* __tactile_keypoint_data.zip__ contains the normalized tactile frames from the carpet and the triangulated and optimized 3D keypoint position in real-world frame (unit: cm)
* __singlePerson_test.zip__ contains the test set for single person pose estimation 
* __singlePerson_test_diffTask.zip__ contains the test set for single person pose estimation, which is arranged by individual tasks

## Quick Start
````
git clone https://github.com/yiyueluo/intelligentCarpet  
cd IntelligentCarpet   
conda env create -f environment.yml   
conda activate p36   
````

## Demo
Checkpoints can be found here: https://www.dropbox.com/sh/9t13gp11wgw8vhd/AACG-woYPVSr_L4UBof5FZnoa?dl=0 

To visualize predictions, set ''--exp_image''


## Contact
If you have any questions about the paper or the codebase, please feel free to contact yiyueluo@mit.edu.
