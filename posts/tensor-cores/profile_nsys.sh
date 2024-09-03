#!/bin/bash
# ------------------------------------------------------------------
# [Author] Robert Alexander Limas Sierra
#          Basic scripting code for profiling CUDA applications
# ------------------

dir=`pwd`
shared_folder="$dir/shared"
echo "Folder for storing results: $shared_folder"
echo "Starting to profile the applications"

# Outer loop goes through the sequence 1, 2, 4, 8, 16, 32
for memory in 16 32; do
  echo "Profiling the apps using $i number elements inside the shared memory"
  for size in $(seq 256 256 16384); do
    echo "Matrix size $size Performing iteration $iteration"
    mkdir -p $shared_folder/$size/$memory
    profile_file="$shared_folder/$size/$memory/report.nsys-rep"
    command="nsys profile \
	    --output=$profile_file --trace=cuda,osrt,nvtx \
	    --cuda-memory-usage=true \
	    --gpu-metrics-device=all \
	    --capture-range=cudaProfilerApi \
	    --cudabacktrace=true \
	    /home/r.limas/Documents/code-samples/posts/tensor-cores/tensorExample $size $memory"
     
    sudo /opt/nvidia/nsight-systems/2022.5.2/target-linux-tegra-armv8/nsys profile \
      --output=~/Documents/report.nsys-rep --force-overwrite=true \
      --trace=cuda,osrt,nvtx \
      --cuda-memory-usage=true \
      --gpu-metrics-device=all \
      --capture-range=cudaProfilerApi \
      --cudabacktrace=true \
      --sampling-trigger=perf --sampling-period=137600 \
      /home/r.limas/Documents/code-samples/posts/tensor-cores/tensorExample $size $memory
  done
done
