// ------------------------------------------------------------------
// Copyright (c) 2017
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// ShiftLayer shift the feature map at four directions 
// ------------------------------------------------------------------
#include <vector>

#include "caffe/layers/shift_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>  //left up right down
__global__ void ShiftForward(const int nthreads, const int stride, const Dtype* bottom_data, Dtype* top_left, Dtype* top_up, Dtype* top_right, Dtype* top_down, const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     if ((w+stride)<width)
     {
        top_left[index] = bottom_data[((n*channel + c)*height + h)*width + w + stride];
     }
     else
     {
        top_left[index] = 0; //bottom_data[((n*channel + c)*height + h)*width + w];
     }
     
     if ((h+stride)<height)
     {
        top_up[index] = bottom_data[((n*channel + c)*height + h + stride)*width + w];
     }
     else
     {
        top_up[index] = 0; //bottom_data[((n*channel + c)*height + h)*width + w];
     }

     if ((w-stride)>=0)
     {
        top_right[index] = bottom_data[((n*channel + c)*height + h)*width + w - stride];
     }
     else
     {
        top_right[index] = 0; //bottom_data[((n*channel + c)*height + h)*width + w];
     }

     if ((h-stride)>=0)
     {
        top_down[index] = bottom_data[((n*channel + c)*height + h - stride)*width + w];
     }
     else
     {
        top_down[index] = 0; //bottom_data[((n*channel + c)*height + h)*width + w];
     }
     //printf("ori: %f, left: %f, up: %f, right: %f, down: %f, n: %d, c: %d, h:%d, w:%d\n",bottom_data[index], top_left[index], top_up[index], top_right[index], top_down[index],n,c,h,w);
  }
}

template <typename Dtype>
__global__ void ShiftBackward(const int nthreads, const int stride, Dtype* bottom_diff, const Dtype* top_left, const Dtype* top_up, const Dtype* top_right, const Dtype* top_down, const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     
     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     Dtype diff = 0;
 
     if ((w+stride)<width)
     {
        diff += top_right[((n*channel + c)*height + h)*width + w + stride];
     }
     else
     {
        diff += 0; //top_right[((n*channel + c)*height + h)*width + w];
     }
     
     if ((h+stride)<height)
     {
        diff += top_down[((n*channel + c)*height + h + stride)*width + w];
     }
     else
     {
        diff += 0; //top_down[((n*channel + c)*height + h)*width + w];
     }

     if ((w-stride)>=0)
     {
        diff += top_left[((n*channel + c)*height + h)*width + w - stride];
     }
     else
     {
        diff += 0; //top_left[((n*channel + c)*height + h)*width + w];
     }

     if ((h-stride)>=0)
     {
        diff += top_up[((n*channel + c)*height + h - stride)*width + w];
     }
     else
     {
        diff += 0; //top_up[((n*channel + c)*height + h)*width + w];
     }

     bottom_diff[index] = diff;

     //printf("ori_diff: %f, left: %f, up: %f, right: %f, down: %f, n: %d, c: %d, h:%d, w:%d\n",bottom_diff[index], top_left[index], top_up[index], top_right[index], top_down[index],n,c,h,w);
  }
}

template <typename Dtype>
void ShiftLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int count = bottom[0]->count();
 
  Dtype* top_data[4];

  for (int i=0; i<4; i++)
  {
      top_data[i] = top[i]->mutable_gpu_data();
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();

  ShiftForward<Dtype>  
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, stride, bottom_data, top_data[0], top_data[1], top_data[2], top_data[3], channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void ShiftLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();

  const Dtype* top_left = top[0]->gpu_diff();
  const Dtype* top_up = top[1]->gpu_diff();
  const Dtype* top_right = top[2]->gpu_diff();
  const Dtype* top_down = top[3]->gpu_diff();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  ShiftBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, stride, bottom_diff, top_left, top_up, top_right, top_down, channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(ShiftLayer);

}  // namespace caffe
