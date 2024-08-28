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
    profile_file="$shared_folder/$size/$memory/report.ncu-proj"
    command="ncu --export $profile_file --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Nvlink_Tables --section Nvlink_Topology --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source no --check-exit-code yes /home/jdguest/Documents/code/posts/tensor-cores/tensorExample $size $memory"
    echo "$command"
    sudo /usr/local/cuda-11.6/nsight-compute-2022.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --export $profile_file --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Nvlink_Tables --section Nvlink_Topology --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source no --check-exit-code yes /home/jdguest/Documents/code/posts/tensor-cores/tensorExample $size $memory
  done
done
