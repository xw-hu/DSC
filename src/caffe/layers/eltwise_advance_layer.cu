// ------------------------------------------------------------------
// Copyright (c) 2017
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// EltwiseAdvLayer the bottom[1] has lower dims than bottom[0].
// Only the Production operation was implemented.
// ------------------------------------------------------------------

#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_advance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype> 
__global__ void MultiForward(const int nthreads, const int replicate_times, const Dtype* bottom_data, const Dtype* bottom_producter, Dtype* top_data, const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     top_data[index] = bottom_data[index] * bottom_producter[((n*channel + (c/replicate_times))*height + h)*width + w];
    
  }
}


template <typename Dtype> 
__global__ void MultiBackward(const int nthreads, const int replicate_times, const Dtype* top_diff, const Dtype* bottom_data0, const Dtype* bottom_data1, Dtype* bottom_diff0, Dtype* bottom_diff1, const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     bottom_diff1[index] = 0;

     for (int i=0; i<replicate_times; i++)
     {
         bottom_diff0[((n*channel + (replicate_times*c+i))*height + h)*width + w] = bottom_data1[index] * top_diff[((n*channel + (replicate_times*c+i))*height + h)*width + w]; //another_bottom_data*top_diff

        bottom_diff1[index] += bottom_data0[((n*channel + (replicate_times*c+i))*height + h)*width + w] * top_diff[((n*channel + (replicate_times*c+i))*height + h)*width + w];
    
     }
    
  }
}

template <typename Dtype>
void EltwiseAdvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_producter = bottom[1]->gpu_data();
  const int replicate_times = int(bottom[0]->count()) / int(bottom[1]->count());

  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    MultiForward<Dtype>  
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, replicate_times, bottom_data, bottom_producter, top_data, channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    NOT_IMPLEMENTED;
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void EltwiseAdvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 
  const int count = bottom[1]->count();
  const Dtype* top_diff = top[0]->gpu_diff();

  Dtype* bottom_diff0 = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff1 = bottom[1]->mutable_gpu_diff();

  const Dtype* bottom_data0 = bottom[0]->gpu_data();
  const Dtype* bottom_data1 = bottom[1]->gpu_data(); 

  const int replicate_times = int(bottom[0]->count()) / int(bottom[1]->count());

  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    if (stable_prod_grad_) {

       MultiBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
           <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, replicate_times, top_diff, bottom_data0, bottom_data1, bottom_diff0, bottom_diff1, channels_, height_, width_);
      CUDA_POST_KERNEL_CHECK;
    
    } else {
      NOT_IMPLEMENTED;
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    NOT_IMPLEMENTED;
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseAdvLayer);

}  // namespace caffe
