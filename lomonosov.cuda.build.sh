#!/bin/bash

module add impi/5.0.1
module add cuda/5.5

nvcc -rdc=true -arch=sm_20 -ccbin mpicxx -Xcompiler -Wall -O2 -I./src src/MpiSupport.cpp src/CudaSupport.cpp src/MathObjects.cpp src/CudaObjects.cpp src/main.cpp src/kernel.cu -o cudirch 2> nvcc_errors.txt

module rm cuda/5.5
module rm impi/5.0.1
