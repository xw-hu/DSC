#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer_BER_weight.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts, float weight0_, float weight1_, float weight_a0_, float weight_a1_, int num_pixel_) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const float target_value = static_cast<float>(target[i]);

    if (has_ignore_label_ && static_cast<int>(target_value) == ignore_label_) {
      loss[i] = 0;
      counts[i] = 0;
    } else {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] *
          (input_data[i] >= 0)));
      counts[i] = 1;

      // weighted
      if (target_value>0.5) //positive
      { loss[i] *= (weight_a0_ + weight0_); }
      else
      { loss[i] *= (weight_a1_ + weight1_); }

      //printf("(weight_a0_ + weight0_): %f, (weight_a1_ + weight1_): %f, After_loss[i]:%f, target_value:%f\n",(weight_a0_ + weight0_), (weight_a1_ + weight1_), loss[i], target_value);
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossIgnoreDiffGPU(const int count,
    const int ignore_label, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}


template <typename Dtype>
void SigmoidCrossEntropyLossBERweightLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* count_data = bottom[1]->mutable_gpu_diff();
  Dtype valid_count;


  // Hu Xiaowei 2017
  // BER = 1-1/2*(TP/(TP+FN)+TN/(TN+FP)) 
  // Calculate the BER error and cross_entropy_loss with weight (area) for each image and use it as a weight
  // 1/2 + 1/2
  //int h = bottom[0]->shape(1);
  //int w = bottom[0]->shape(2);

  const Dtype* input_data_cpu = bottom[0]->cpu_data();
  const Dtype* target_cpu = bottom[1]->cpu_data();
  
  const int num_image = bottom[0]->shape(0);
  int num_pixel = bottom[0]->count(1);

  float weight0, weight1; // [batch_size=1] [weight_p, weight_n];
  float weight_a0, weight_a1;

  for (int i=0; i<num_image; i++)
  {
      int countPos = 0;
      int countNeg = 0;
      int countTP = 0; //true positive;
      int countTN = 0; //true negative;

      for (int j=0; j<num_pixel; j++)
      {   
          float t_data = target_cpu[i*num_pixel+j];
          float i_data = input_data_cpu[i*num_pixel+j];


         if (t_data>0.5) //positive
          {  
             countPos++;
             if (i_data>0.5) //positive
             {
                countTP++;
             }
          }
          else
          {  
             countNeg++;
             if (i_data<=0.5) //negative
             {
                countTN++;
             }
          }
      } 

      weight_a0 = (float)countNeg / (float)(countPos+countNeg);
      weight_a1 = (float)countPos / (float)(countPos+countNeg);

      if (countPos==0)
      { weight0 = 1; }
      else
      { weight0 = 1-((float)countTP/(float)countPos); }//positive weight
      if (countNeg==0)
      { weight1 = 1; }
      else
      { weight1 = 1-((float)countTN/(float)countNeg); }//negative weight
  }  

  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data,
      has_ignore_label_, ignore_label_, count_data, weight0, weight1, weight_a0, weight_a1, num_pixel);
  // Only launch another CUDA kernel if we actually need the valid count.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(count, count_data, &valid_count);
  } else {
    valid_count = count;
  }
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void SigmoidCrossEntropyLossBERweightLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SigmoidCrossEntropyLossIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, ignore_label_, target, bottom_diff);
    }
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossBERweightLayer);

}  // namespace caffe
