# Direction-Aware Spatial Context Features for Shadow Detection (and Removal)

by Xiaowei Hu, Chi-Wing Fu, Lei Zhu, Jing Qin and Pheng-Ann Heng

This implementation is written by Xiaowei Hu at the Chinese University of Hong Kong.

***

## Citation

@InProceedings{Hu_2018_CVPR,      
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},      
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-Aware Spatial Context Features for Shadow Detection},      
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},      
&nbsp;&nbsp;&nbsp;&nbsp;  pages={7454--7462},        
&nbsp;&nbsp;&nbsp;&nbsp;  year = {2018}
}

@article{hu2019direction,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Fu, Chi-Wing and Zhu, Lei and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-Aware Spatial Context Features for Shadow Detection and Removal},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2019},          
&nbsp;&nbsp;&nbsp;&nbsp;  note={to appear}                  
}


## Results

The shadow detection results on the SBU and UCF datasets can be found at [Google Drive](https://drive.google.com/open?id=1DCTqEnYJ8ADBqShBzXFYKa_yD-YZKEo7).           
The shadow detection results on the new split of UCF (used by some works) can be found at [Google Drive](https://drive.google.com/file/d/1AL78O1Vkdb0gCUWS57lv2wcQM0gDFa0L); BER: 10.38, accuracy: 0.95.          

The shadow removal results on the SRD and ISTD datasets can be found at [Google Drive](https://drive.google.com/open?id=1QzsaNn35PE4OORj4yemKxMwSh-Azcf3Y).     


## Pytorch Version
A PyTorch version is available at [https://github.com/stevewongv/DSC-PyTorch](https://github.com/stevewongv/DSC-PyTorch) implemented by [Tianyu Wang](https://github.com/stevewongv).


## Installation

1. Please download and compile our [CF-Caffe](https://github.com/xw-hu/CF-Caffe).

2. Clone the DSC repository, and we'll call the directory that you cloned as `DSC-master`.

    ```shell
    git clone https://github.com/xw-hu/DSC.git
    ```

3. Replace `CF-Caffe/examples/` by `DSC-master/examples/`.
   Replace `CF-Caffe/data/` by `DSC-master/data/`.


## Test   

### Shadow Detection   
1. Please download our pretrained model at [Google Drive](https://drive.google.com/open?id=1RAdblaOEZaH8fAeqJ-8G2Cro4Crp1NdJ).   
   Put this model in `examples/DSC/DSC_detection/snapshot/`.

2. (Matlab User) Enter the `examples/DSC/` and run `test_detection.m` in Matlab. 
 
2. (Python User) Enter the `examples/DSC/DSC_detection/` and export PYTHONPATH in the command window such as:

   ```shell
   export PYTHONPATH='../../../python'
   ```  
   
   Run the test model and resize the results to the size of original images:
     
   ```shell
   ipython notebook DSC_test.ipynb
   ``` 

3. Apply CRF to do the post-processing for each image.   
   The code for CRF can be found in [https://github.com/Andrew-Qibin/dss_crf](https://github.com/Andrew-Qibin/dss_crf)   
   *Note that please provide a link to the original code as a footnote or a citation if you plan to use it.

### Shadow Removal   
Enter the `examples/DSC/` and run `test_removal.m` in Matlab.    
  
## Train

Download the pre-trained VGG16 model at [http://www.robots.ox.ac.uk/~vgg/research/very_deep/](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).   
   Put this model in `CF-Caffe/models/`
   
### Shadow Detection   
1. Enter the `examples/DSC/DSC_detection/`   
   Modify the image path in `DSC.prototxt`.

2. Run   
   ```shell
   sh train.sh
   ```

### Shadow Removal   
1. Color compensation mechanism:     
   Enter the `/data/SRD/` or `/data/ISTD/`.      
   Run `color_transfer_function.m` in Matlab.     

2. Transfer the images into the `LAB` color sapce and do the data argumentation:     
   Enter the `/data/SRD/` or `/data/ISTD/`.       
   Run `ToLab.m` and `data_argument.m` in Matlab.       
   
3. Enter the `examples/DSC/DSC_removal_SRD/` or `examples/DSC/DSC_removal_ISTD/`.     
   Modify the image path in `DSC.prototxt`.     

4. Run     
   ```shell
   sh train.sh
   ```    
   
