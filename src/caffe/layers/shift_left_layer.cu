// ------------------------------------------------------------------
// Copyright (c) 2017
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// ShiftLeftLayer shift the feature map 
// ------------------------------------------------------------------
#include <vector>

#include "caffe/layers/shift_left_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>  //left up right down
__global__ void ShiftLeftForward(const int nthreads, const int stride, const Dtype* bottom_data, Dtype* top_left, const int channel, const int height, const int width) {
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
     
  }
}

template <typename Dtype>
__global__ void ShiftLeftBackward(const int nthreads, const int stride, Dtype* bottom_diff, const Dtype* top_left, const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     
     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     Dtype diff = 0;
 

     if ((w-stride)>=0)
     {
        diff += top_left[((n*channel + c)*height + h)*width + w - stride];
     }
     else
     {
        diff += 0; //top_left[((n*channel + c)*height + h)*width + w];
     }

     bottom_diff[index] = diff;

  }
}

template <typename Dtype>
void ShiftLeftLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int count = bottom[0]->count();
 
  Dtype* top_data = top[0]->mutable_gpu_data();

  const Dtype* bottom_data = bottom[0]->gpu_data();

  ShiftLeftForward<Dtype>
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, stride, bottom_data, top_data, channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void ShiftLeftLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();

  const Dtype* top_left = top[0]->gpu_diff();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  ShiftLeftBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, stride, bottom_diff, top_left, channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(ShiftLeftLayer);

}  // namespace caffe
