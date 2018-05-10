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
void EltwiseAdvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "Eltwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
}

template <typename Dtype>
void EltwiseAdvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  top[0]->ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void EltwiseAdvLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void EltwiseAdvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseAdvLayer);
#endif

INSTANTIATE_CLASS(EltwiseAdvLayer);
REGISTER_LAYER_CLASS(EltwiseAdv);

}  // namespace caffe
