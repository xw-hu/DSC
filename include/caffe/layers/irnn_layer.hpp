// ------------------------------------------------------------------
// Copyright (c) 2017
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// Spatial RNN with identity matrix at four directions 
// ------------------------------------------------------------------

#ifndef CAFFE_IRNN_LAYER_HPP_
#define CAFFE_IRNN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bias_layer.hpp"

namespace caffe {

template <typename Dtype>
class IRNNLayer : public Layer<Dtype> {
 public:
  explicit IRNNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IRNN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }  
  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool weight_fixed;
  int channels_;
  int height_;
  int width_;
  Blob<Dtype> weight_diff_map;

};


}  // namespace caffe

#endif  // CAFFE_IRNN_LAYER_HPP_
