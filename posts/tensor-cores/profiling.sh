#!/bin/bash
# ------------------------------------------------------------------
# [Author] Robert Alexander Limas Sierra
#          Basic scripting code for profiling CUDA applications
# ------------------------------------------------------------------

dir=`pwd`
echo "Working on $dir"

echo "method,size,time,shared,flops,tflops" > $dir/profile.csv

echo "Starting to profile the applications"

# Outer loop goes through the sequence 1, 2, 4, 8, 16, 32
for i in 1 2 4 8 16 32; do
  echo "Profiling the apps using $i number elements inside the shared memory"
  for size in $(seq 256 256 16384); do
    for iteration in $(seq 1 1 10); do
        echo "Matrix size $size Performing iteration $iteration"
        ./tensorExample $size $i >> $dir/profile.csv
    done
  done
done
