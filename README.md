# Direction-aware Spatial Context Features for Shadow Detection

by Xiaowei Hu, Lei Zhu, Chi-Wing Fu, Jing Qin and Pheng-Ann Heng

This implementation is written by Xiaowei Hu at the Chinese University of Hong Kong.

***

## Citation
@inproceedings{hu2018direction,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-aware Spatial Context Features for Shadow Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {CVPR},    
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
   make pycaffe
   ```

## Test
1. Please download our pretrained model at [Google Drive](https://drive.google.com/open?id=1RAdblaOEZaH8fAeqJ-8G2Cro4Crp1NdJ).   
   Put this model in `DSC/examples/snapshot/`.

2. Export PYTHONPATH in the command window such as:

   ```shell
   export PYTHONPATH='/home/xwhu/DSC/python'
   ```
 
3. Run the test model (please modify the path of images):
   
   ```shell
   ipython notebook DSC_test.ipynb
   ``` 

4. Resize the results to the size of original images and apply CRF to do the post-processing for each image.   
   The code for CRF can be found in [https://github.com/Andrew-Qibin/dss_crf](https://github.com/Andrew-Qibin/dss_crf)   
   *Note that please provide a link to the original code as a footnote or a citation if you plan to use it.
  
## Train
1. Download the pre-trained VGG16 model at [http://www.robots.ox.ac.uk/~vgg/research/very_deep/](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).   
   Put this model in `DSC/models/`

2. Enter the `DSC/examples/`   
   Modify the image path in `train_val.prototxt`.

3. Run   
   ```shell
   sh train.sh
   ```
