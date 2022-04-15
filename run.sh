#!/bin/bash

g++ yolov4.c yolov4_det_head.c -o yolov4 -I. -I../onnxruntime-linux-x64-1.11.0/include/ -fPIC ../onnxruntime-linux-x64-1.11.0/lib/libonnxruntime.so -lm `pkg-config --libs opencv`

if [ $# -eq 0 ] ; then
	./yolov4 yolov4.onnx dog.jpg cpu
else
	./yolov4 yolov4.onnx $1 cpu
fi
