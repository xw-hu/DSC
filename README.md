# Direction-aware Spatial Context Features for Shadow Detection (and Removal)

by Xiaowei Hu, Chi-Wing Fu, Lei Zhu, Jing Qin and Pheng-Ann Heng

This implementation is written by Xiaowei Hu at the Chinese University of Hong Kong.

***

## Citation

@inproceedings{hu2018direction,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-aware Spatial Context Features for Shadow Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {CVPR},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}

@article{hu2018direction,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Fu, Chi-Wing and Zhu, Lei and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-aware Spatial Context Features for Shadow Detection and Removal},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={arXiv preprint arXiv},   
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}


## Results

The results of shadow detection on SBU and UCF can be found at [Google Drive](https://drive.google.com/open?id=1DCTqEnYJ8ADBqShBzXFYKa_yD-YZKEo7).

## Installation

1. Clone the DSC repository, and we'll call the directory that you cloned DSC into `DSC`.

    ```shell
    git clone https://github.com/xw-hu/DSC.git
    ```

2. Build DSC (based on Caffe)

   *This model is tested on Ubuntu 16.04, CUDA 8.0, cuDNN 5.0   
    
   Follow the Caffe installation instructions here: [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html)   
   
   ```shell
   make all -j XX
   make matcaffe
   make pycaffe
   ```

## Test   

# Shadow Detection   
1. Please download our pretrained model at [Google Drive](https://drive.google.com/open?id=1RAdblaOEZaH8fAeqJ-8G2Cro4Crp1NdJ).   
   Put this model in `DSC/examples/DSC_detection/snapshot/`.

2. (Matlab User) Enter the `DSC/examples/` and run `test_detection.m` in Matlab. 
 
2. (Python User) Enter the `DSC/examples/DSC/` and export PYTHONPATH in the command window such as:

   ```shell
   export PYTHONPATH='../../python'
   ```  
   
   Run the test model (please modify the path of images) and resize the results to the size of original images:
     
   ```shell
   ipython notebook DSC_test.ipynb
   ``` 

3. Apply CRF to do the post-processing for each image.   
   The code for CRF can be found in [https://github.com/Andrew-Qibin/dss_crf](https://github.com/Andrew-Qibin/dss_crf)   
   *Note that please provide a link to the original code as a footnote or a citation if you plan to use it.

# Shadow Removal   
Enter the `DSC/examples/` and run `test_removal.m` in Matlab. 
  
## Train

Download the pre-trained VGG16 model at [http://www.robots.ox.ac.uk/~vgg/research/very_deep/](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).   
   Put this model in `DSC/models/`
   
# Shadow Detection   
1. Enter the `DSC/examples/DSC_detection/`   
   Modify the image path in `DSC.prototxt`.

2. Run   
   ```shell
   sh train.sh
   ```

# Shadow Removal   
1. Color compensation mechanism:   
   Enter the `DSC/data/SRD/` or `DSC/data/ISTD/`.   
   Run `color_transfer_function.m` in Matlab.   

2. Transfer the images into the `LAB` color sapce and do the data argumentation:
   Enter the `DSC/data/SRD/` or `DSC/data/ISTD/`.   
   Run `ToLab.m` and `data_argument.m` in Matlab.   
   
3. Enter the `DSC/examples/DSC_removal_SRD/` or `DSC/examples/DSC_removal_ISTD/`.
   Modify the image path in `DSC.prototxt`.

4. Run   
   ```shell
   sh train.sh
   ```
   
