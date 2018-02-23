// ------------------------------------------------------------------
// Copyright (c) 2017
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// Spatial RNN with identity matrix at four directions 
// ------------------------------------------------------------------

#include <vector>
#include <iostream>

#include "caffe/filler.hpp"

#include "caffe/layers/irnn_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void IRNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  IRNNParameter irnn_parameter = this->layer_param_.irnn_param();
  weight_fixed = irnn_parameter.weight_fixed();
   
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else { 
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels_*4)));
  }
  
  shared_ptr<Filler<Dtype> > filler;
 
  FillerParameter filler_param;
  filler_param.set_type("constant");
  filler_param.set_value(1.0);
  filler.reset(GetFiller<Dtype>(filler_param));
  filler->Fill(this->blobs_[0].get());  

  // Propagate gradients to the parameters (as directed by backward pass).
  if (!weight_fixed)
     this->param_propagate_down_.resize(this->blobs_.size(), true);
 
}

template <typename Dtype>
void IRNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  for (int i=0;i<4;i++) //left up right down
  {
      top[i]->Reshape(bottom[0]->shape(0), channels_, height_, width_);
  }

  if (!weight_fixed)
     weight_diff_map.Reshape(bottom[0]->shape(0), channels_*4, height_, width_);
}

template <typename Dtype>
void IRNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  NOT_IMPLEMENTED;
  
  //Dtype* top_data[4];
  //const Dtype* bottom_data[1];

  //for (int i=0; i<4; i++)
  //{
  //    top_data[i] = top[i]->mutable_cpu_data();
  //}
  //bottom_data[0] = bottom[0]->cpu_data();
 
}

template <typename Dtype>
void IRNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(IRNNLayer);
#endif

INSTANTIATE_CLASS(IRNNLayer);
REGISTER_LAYER_CLASS(IRNN);

}  // namespace caffe
