// ------------------------------------------------------------------
// Copyright (c) 2017
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// Spatial RNN with identity matrix at four directions 
// ------------------------------------------------------------------
#include <vector>

#include "caffe/layers/irnn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>  //left up right down
__global__ void IRNNForward(const int nthreads, const Dtype* bottom_data, Dtype* top_left, Dtype* top_up, Dtype* top_right, Dtype* top_down, const int channel, const int height, const int width, const Dtype* weight_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     Dtype temp = 0; 

     //left
     top_left[index] = 0; 
     
     for (int i=width-1; i>=w; i--)
     {
        temp = top_left[index] * weight_data[c] + bottom_data[((n*channel + c)*height + h)*width + i]; //0*channel+c
        top_left[index] = (temp > 0) ? temp : 0; //ReLU: max(x,0)
     }

      
     //up
     top_up[index] = 0;
 
     for (int i=height-1; i>=h; i--)
     {
        temp = top_up[index] * weight_data[channel + c] + bottom_data[((n*channel + c)*height + i)*width + w]; //1*channel+c
        top_up[index] = (temp > 0) ? temp : 0; //ReLU: max(x,0)
     }
     
     
     //right
     top_right[index] = 0;
     
     for (int i=0; i<=w; i++)
     {
        temp = top_right[index] * weight_data[2*channel + c] + bottom_data[((n*channel + c)*height + h)*width + i]; //2*channel+c
        top_right[index] = (temp > 0) ? temp : 0; //ReLU: max(x,0)
     }
     
     
     //down
     top_down[index] = 0; 

     for (int i=0; i<=h; i++)
     {
        temp = top_down[index] * weight_data[3*channel + c] + bottom_data[((n*channel + c)*height + i)*width + w]; //3*channel+c
        top_down[index] = (temp > 0) ? temp : 0; //ReLU: max(x,0)
    
     }

     //printf("ori: %f, left: %f, up: %f, right: %f, down: %f, n: %d, c: %d, h:%d, w:%d\n",bottom_data[index], top_left[index], top_up[index], top_right[index], top_down[index],n,c,h,w);
  }
}

template <typename Dtype>
__global__ void IRNNBackward(const int nthreads, Dtype* bottom_diff, const Dtype* top_left, const Dtype* top_up, const Dtype* top_right, const Dtype* top_down, const Dtype* top_left_data, const Dtype* top_up_data, const Dtype* top_right_data, const Dtype* top_down_data, const int channel, const int height, const int width, const Dtype* weight_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     
     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     Dtype diff_right = 0;
     Dtype diff_left = 0;
     Dtype diff_down = 0;
     Dtype diff_up = 0;


     //right 
     for (int i=width-1; i>=w; i--)
     {
        diff_right *= weight_data[2*channel + c];
        diff_right += (top_right_data[((n*channel + c)*height + h)*width + i]==0)? 0 : top_right[((n*channel + c)*height + h)*width + i];
     }
 

     //left 
     for (int i=0; i<=w; i++)
     {  
        diff_left *= weight_data[c];
        diff_left += (top_left_data[((n*channel + c)*height + h)*width + i]==0)? 0 : top_left[((n*channel + c)*height + h)*width + i];
     }
     

     //down 
     for (int i=height-1; i>=h; i--)
     {
        diff_down *= weight_data[3*channel + c];
        diff_down += (top_down_data[((n*channel + c)*height + i)*width + w]==0)? 0 : top_down[((n*channel + c)*height + i)*width + w];
     }
     
     
     //up
     for (int i=0; i<=h; i++)
     {  
        diff_up *= weight_data[channel + c];
        diff_up += (top_up_data[((n*channel + c)*height + i)*width + w]==0)? 0 : top_up[((n*channel + c)*height + i)*width + w];
     }

     bottom_diff[index] = diff_right + diff_left + diff_down + diff_up;

     //printf("ori_diff: %f, left: %f, up: %f, right: %f, down: %f, n: %d, c: %d, h:%d, w:%d\n",bottom_diff[index], top_left[index], top_up[index], top_right[index], top_down[index],n,c,h,w);
  }
}

template <typename Dtype>
__global__ void IRNNParamBackward(const int nthreads, const Dtype* top_left, const Dtype* top_up, const Dtype* top_right, const Dtype* top_down, const Dtype* top_left_data, const Dtype* top_up_data, const Dtype* top_right_data, const Dtype* top_down_data, const int channel, const int height, const int width, const Dtype* weight_data, const Dtype* bottom_data, Dtype* weight_map) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  
  int w = index % width;
  int h = (index / width) % height;
  int c = (index / width / height) % channel;
  int n = index / width / height / channel;

  //left
  Dtype diff_temp = 0;
  float acc_weight = 1;
  for (int m=w+1; m<width; m++)
  {  
      if (top_left_data[((n*channel + c)*height + h)*width + m]==0)
      { 
         break;
      }
      diff_temp += (m-w)*acc_weight*top_left[((n*channel + c)*height + h)*width + m];
      acc_weight *= weight_data[c];
  }

  weight_map[((n*channel + c*4)*height + h)*width + w] = diff_temp*bottom_data[((n*channel + c)*height + h)*width + w];

  
  // up
  diff_temp = 0;
  acc_weight = 1;
  for (int m=h+1; m<height; m++)
  {  
      if (top_up_data[((n*channel + c)*height + m)*width + w]==0)
      { 
          break;
      }
      diff_temp += (m-h)*acc_weight*top_up[((n*channel + c)*height + m)*width + w];
      acc_weight *= weight_data[channel + c];               
  }
  weight_map[((n*channel + c*4+1)*height + h)*width + w] = diff_temp*bottom_data[((n*channel + c)*height + h)*width + w];

  
  //right
  diff_temp = 0;
  acc_weight = 1;
  for (int m=w-1; m>=0; m--)
  {  
       if (top_right_data[((n*channel + c)*height + h)*width + m]==0)
       { 
          break;
       }
       diff_temp += (w-m)*acc_weight*top_right[((n*channel + c)*height + h)*width + m];
       acc_weight *= weight_data[2*channel + c];
                     
  }
  weight_map[((n*channel + c*4+2)*height + h)*width + w] = diff_temp*bottom_data[((n*channel + c)*height + h)*width + w];
      

  //down
  diff_temp = 0;
  acc_weight = 1;
  for (int m=h-1; m>=0; m--)
  {  
      if (top_down_data[((n*channel + c)*height + m)*width + w]==0)
      { 
          break;
      }
      diff_temp += (h-m)*acc_weight*top_down[((n*channel + c)*height + m)*width + w];
      acc_weight *= weight_data[3*channel + c];
                    
   }
   weight_map[((n*channel + c*4+3)*height + h)*width + w] = diff_temp*bottom_data[((n*channel + c)*height + h)*width + w];
 }  
}


template <typename Dtype>
__global__ void IRNNParamSetValueBackward(const int ndex, int batch_num, const int channel, const int height, const int width,  Dtype* weight_diff, Dtype* weight_map) {
  CUDA_KERNEL_LOOP(index, ndex) {
  
  int MAX_GRADIENT = 5000;
  int dir = index / channel;
  int c = index % channel;

  switch (dir)
  {
    float diff;
    case 0: // left 
      diff = 0;
      for (int i=0; i<batch_num; i++)
      {
          for (int j=0; j<height; j++)
          {
              for (int k=0; k<width; k++)
              {
                 
                 diff += weight_map[((i*channel + c*4)*height + j)*width + k];
              }
          }
      }
  
      // gradient clip
      if (diff>MAX_GRADIENT || diff<-MAX_GRADIENT)
      { diff = 0; }
      weight_diff[c] = diff;
      
    break;

    case 1: // up
      diff = 0;
      for (int i=0; i<batch_num; i++)
      {
          for (int j=0; j<height; j++)
          {
              for (int k=0; k<width; k++)
              {
                 diff += weight_map[((i*channel + c*4+1)*height + j)*width + k];
              }
          }
      }

      if (diff>MAX_GRADIENT || diff<-MAX_GRADIENT)
      { diff = 0; }
      weight_diff[channel + c] = diff;
    break;

    case 2: // right
      diff = 0;
      for (int i=0; i<batch_num; i++)
      {
          for (int j=0; j<height; j++)
          {
              for (int k=0; k<width; k++)
              {
                 diff += weight_map[((i*channel + c*4+2)*height + j)*width + k];
              }
          }
      }

      if (diff>MAX_GRADIENT || diff<-MAX_GRADIENT)
      { diff = 0; }
      weight_diff[2*channel + c] = diff;
    break;

    case 3: // down
      diff = 0;
      for (int i=0; i<batch_num; i++)
      {
          for (int j=0; j<height; j++)
          {
              for (int k=0; k<width; k++)
              {
                 diff += weight_map[((i*channel + c*4+3)*height + j)*width + k];
              }
          }
      }

      if (diff>MAX_GRADIENT || diff<-MAX_GRADIENT)
      { diff = 0; }
      weight_diff[3*channel + c] = diff;
    break;
   
    default: break;
  }  
 }
}


template <typename Dtype>
void IRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int count = bottom[0]->count();
 
  Dtype* top_data[4];

  for (int i=0; i<4; i++)
  {
      top_data[i] = top[i]->mutable_gpu_data();
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();

  IRNNForward<Dtype>  
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data[0], top_data[1], top_data[2], top_data[3], channels_, height_, width_, weight_data);
  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void IRNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();

  const Dtype* top_left = top[0]->gpu_diff();
  const Dtype* top_up = top[1]->gpu_diff();
  const Dtype* top_right = top[2]->gpu_diff();
  const Dtype* top_down = top[3]->gpu_diff();

  const Dtype* top_left_data = top[0]->gpu_data();
  const Dtype* top_up_data = top[1]->gpu_data();
  const Dtype* top_right_data = top[2]->gpu_data();
  const Dtype* top_down_data = top[3]->gpu_data();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();

  IRNNBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, top_left, top_up, top_right, top_down, top_left_data, top_up_data, top_right_data, top_down_data, channels_, height_, width_, weight_data);
  CUDA_POST_KERNEL_CHECK;

  
  
  if (!weight_fixed)
  {
    Dtype* weight_map = weight_diff_map.mutable_gpu_data();

    int cdim = 4*channels_;
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    int batch_num = bottom[0]->shape(0);


    IRNNParamBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_left, top_up, top_right, top_down, top_left_data, top_up_data, top_right_data, top_down_data, channels_, height_, width_, weight_data, bottom_data, weight_map);
    CUDA_POST_KERNEL_CHECK;

    IRNNParamSetValueBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim), CAFFE_CUDA_NUM_THREADS>>>(cdim, batch_num, channels_, height_, width_, weight_diff, weight_map);
    CUDA_POST_KERNEL_CHECK;

    
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(IRNNLayer);

}  // namespace caffe
