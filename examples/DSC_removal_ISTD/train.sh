GLOG_logtostderr=1 ../../build/tools/caffe train \
  --solver=solver.prototxt \
  --weights=../../models/VGG_ILSVRC_16_layers.caffemodel \
  --gpu=0  2>&1 | tee log.log

