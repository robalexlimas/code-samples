/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <curand.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#include <mma.h>
using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;

   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
   
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);
   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);

#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_row_major);
   }
}

__device__ void expand_device(half* out, half* in, int rows, int cols, int index, int stride) {
   // Safe mode
   for (int row = 0; row < rows; row++)
   {
      for (int col = 0; col < WMMA_K / 2; col++)
      {
         printf("Index %d row %d col %d\n", index, row, col);
         out[(row * cols) + col] = in[index + (row * cols) + col];
         out[(row * cols) + (WMMA_K / 2) + col] = in[index + (row * cols) + col];
      }
   }
}

__global__ void wmma_example_mod(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
   int sublaneid;
   asm("mov.u32 %0, %laneid;" :"=r"(sublaneid));
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag_0; 
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag_1;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_0;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_1;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_0;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_1;
   wmma::fill_fragment(acc_frag_0, 0.0f);
   wmma::fill_fragment(acc_frag_1, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);

         // wmma::load_matrix_sync(b_frag_0, b_0, 16);
         wmma::load_matrix_sync(b_frag_0, b + bRow + bCol * ldb, ldb);
         wmma::load_matrix_sync(b_frag_1, b + bRow + bCol * ldb, ldb);

         // if (sublaneid == 7) {
         //    for(int i=0; i < 8; i++) { // * IMPORTANT: this value must be between 0 and 7, otherwise, it affects the other operations
         //       if (i < 4)        // ! TCU 0
         //          b_frag_0.x[i] = 0.0f;
         //       else              // ! TCU 1
         //          b_frag_0.x[i] = 10.0f;
         //    }
         // }

         // * SAFE MODE TCU 0 <- TCU 1
         for(int i=0; i < b_frag_0.num_elements / 4; i++) {
            b_frag_0.x[i] = b_frag_0.x[i + 4];
         }

         // * SAFE MODE TCU 1 <- TCU 0
         for(int i=0; i < b_frag_1.num_elements / 4; i++) {
            b_frag_1.x[i + 4] = b_frag_1.x[i];
         }

         // !emulate a faul inside the TCU 0 on only one column
         // if (sublaneid == 0) { // * fault in the column 0
         //    for(int i=0; i < 4; i++) {
         //       if (i == 1) {
         //          b_frag_0.x[i] = 0.0f;
         //       }
         //    }
         // }

         // * DIAGNOSIS MODE
         // for(int i=0; i < b_frag_0.num_elements; i++) {
         //    if (i < 2) {
         //       b_frag_0.x[i] = 0.0f;
         //    }
         // }

         __syncthreads();

         wmma::mma_sync(acc_frag_0, a_frag, b_frag_0, acc_frag_0);
         wmma::mma_sync(acc_frag_1, a_frag, b_frag_1, acc_frag_1);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag_0, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);
      wmma::load_matrix_sync(c_frag_1, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);


      for(int i=0; i < c_frag_0.num_elements; i++) {
         c_frag_0.x[i] = alpha * acc_frag_0.x[i] + beta * c_frag_0.x[i];
      }

      for(int i=0; i < c_frag_1.num_elements; i++) {
         c_frag_1.x[i] = alpha * acc_frag_1.x[i] + beta * c_frag_1.x[i];
      }

      __syncthreads();

      for(int i=0; i < 4; i++) {
         if (c_frag_0.x[i] != c_frag_0.x[i + 4]) {
            printf("0Dif %d frag %2.2f 2frag %2.2f\n", i, c_frag_0.x[i], c_frag_0.x[i + 4]);
         }
      }

      for(int i=0; i < 4; i++) {
         if (c_frag_1.x[i] != c_frag_1.x[i + 4]) {
            printf("1Dif %d frag %2.2f 2frag %2.2f\n", i, c_frag_1.x[i], c_frag_1.x[i + 4]);
         }
      }

      __syncthreads();
      
      // Store the output

      for(int i=0; i < 4; i++) {
         c_frag_0.x[i] = c_frag_1.x[i + 4];
      }

      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag_0, ldc, wmma::mem_row_major);
   }
}

__global__ void wmma_safe_mode(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta, int *fault) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag_0; 
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag_1;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_0;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_1;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_0;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_1;
   wmma::fill_fragment(acc_frag_0, 0.0f);
   wmma::fill_fragment(acc_frag_1, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);

         wmma::load_matrix_sync(b_frag_0, b + bRow + bCol * ldb, ldb);
         wmma::load_matrix_sync(b_frag_1, b + bRow + bCol * ldb, ldb);

         // * SAFE MODE TCU 0 <- TCU 1
         for(int i=0; i < b_frag_0.num_elements / 4; i++) {
            b_frag_0.x[i] = b_frag_0.x[i + 4];
         }

         // * SAFE MODE TCU 1 <- TCU 0
         for(int i=0; i < b_frag_1.num_elements / 4; i++) {
            b_frag_1.x[i + 4] = b_frag_1.x[i];
         }

         wmma::mma_sync(acc_frag_0, a_frag, b_frag_0, acc_frag_0);
         wmma::mma_sync(acc_frag_1, a_frag, b_frag_1, acc_frag_1);
      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag_0, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);
      wmma::load_matrix_sync(c_frag_1, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);

      for(int i=0; i < c_frag_0.num_elements; i++) {
         c_frag_0.x[i] = alpha * acc_frag_0.x[i] + beta * c_frag_0.x[i];
      }

      for(int i=0; i < c_frag_1.num_elements; i++) {
         c_frag_1.x[i] = alpha * acc_frag_1.x[i] + beta * c_frag_1.x[i];
      }

      __syncthreads();

      for(int i=0; i < 4; i++) {
         if (c_frag_0.x[i] != c_frag_0.x[i + 4]) {
            printf("Something happened frag0!!!\n");
            fault[0] = 1;
         }
         if (c_frag_1.x[i] != c_frag_1.x[i + 4]) {
            printf("Something happened frag1!!!\n");
            fault[0] = 1;
         }
      }

      // Store the output
      __syncthreads();
      for(int i=0; i < 4; i++) {
         c_frag_0.x[i] = c_frag_1.x[i + 4];
      }

      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag_0, ldc, wmma::mem_row_major);
   }
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

enum MatrixInitializationType{
   ZERO,
   ONE,
   RANDOM,
   IDENTITY,
   LINEAR
};

__host__ int get_value(MatrixInitializationType init_type,int randomRange=3,int RESET=false){
   static int val=0;
   switch(init_type){
      case ZERO:
         break;
      case ONE:
         val=1;
         break;
      case RANDOM:
         val=rand()%randomRange;
         break;
      case LINEAR:
         val++;
         break;
      default :
         printf("illegal MatrixInitializationType\n");
         abort();
         break;
   }
   if(RESET)
      val=0;
   return val;
}

template <typename T>
__host__ void initialize_matrix(T *matrix, int size, MatrixInitializationType init_type, int RESET=false)  {
   for (int idx=0; idx<size; idx++) {
      matrix[idx] = (T)get_value(init_type, 3, RESET);
   }
}

template <typename T>
void print_matrix(T *matrix, int rows, int cols){
   for (int row=0; row<rows; row++) {
      for (int col=0; col<cols; col++) {
         T val;
         val=matrix[row*cols+col];
         printf("%2.2f ",static_cast<float>(val));
      }
      printf(";\n");
   }
}

template<typename T>
void expand(T* out, T* in, int rows, int cols) {
   // Safe mode
   for (int row = 0; row < rows; row++)
   {
      for (int col = 0; col < WMMA_K / 2; col++)
      {
         out[(row * cols) + col] = in[(row * cols) + col];
         out[(row * cols) + (WMMA_K / 2) + col] = in[(row * cols) + col];
      }
   }
}


int main(int argc, char* argv[]) {
   float *a_fp32;
   float *b_fp32;
   float *a_fp32_device;
   float *b_fp32_device;
   half *a_fp16;
   half *b_fp16;

   // Must be multiples of 16 for wmma code to work
   if (argv[1] == "") {
      printf("Matrix size must be initialized, and it must be multiples of 16");
      return 0;
   }
   const int MATRIX_M = atoi(argv[1]);
   const int MATRIX_N = atoi(argv[1]);
   const int MATRIX_K = atoi(argv[1]);

   float *c;
   float *c_wmma;
   float *c_wmma_1;

   float *c_wmma_safe;

   float *c_host_wmma;
   float *c_host_wmma_1;

   float *c_host_wmma_safe;

   int *fault;
   int *fault_device;

   printf("Starting...\n");

   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaEventCreate(&startWMMA);
   cudaEventCreate(&stopWMMA);

   cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half));
   cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half));
   cudaMalloc((void**)&a_fp32_device, MATRIX_M * MATRIX_K * sizeof(float));
   cudaMalloc((void**)&b_fp32_device, MATRIX_K * MATRIX_N * sizeof(float));
   cudaMalloc((void**)&fault_device, sizeof(int));

   a_fp32 = (float*)malloc(MATRIX_M * MATRIX_K * sizeof(float));
   b_fp32 = (float*)malloc(MATRIX_K * MATRIX_N * sizeof(float));
   fault = (int*)malloc(sizeof(int));
   fault[0] = 0;

   c = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_wmma_1 = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   c_host_wmma_safe = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float));
   cudaMalloc((void**)&c_wmma_1, MATRIX_M * MATRIX_N * sizeof(float));

   cudaMalloc((void**)&c_wmma_safe, MATRIX_M * MATRIX_N * sizeof(float));

   printf("b_host\n");
   initialize_matrix<float>(b_fp32, MATRIX_K * MATRIX_N, LINEAR);
   print_matrix<float>(b_fp32, MATRIX_K, MATRIX_N);
   
   printf("a_host\n");
   initialize_matrix<float>(a_fp32, MATRIX_M * MATRIX_K, ONE);
   print_matrix<float>(a_fp32, MATRIX_M, MATRIX_K);

   initialize_matrix<float>(c, MATRIX_M * MATRIX_N, ZERO, true);

   cudaMemcpy(a_fp32_device, a_fp32, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(b_fp32_device, b_fp32, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice);

   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32_device, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32_device, MATRIX_K * MATRIX_N);

   cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice);

   float alpha = 2.0f;
   float beta = 2.0f;

   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
      // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;

   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
   printf("Running with wmma...\n");
   cudaEventRecord(startWMMA);
   wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   printf("Running safe...\n");
   wmma_safe_mode <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma_safe, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, fault_device);
   cudaEventRecord(stopWMMA);
   cudaEventSynchronize(stopWMMA);
  
   printf("\nChecking results...\n");
   cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(c_host_wmma_safe, c_wmma_safe, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(fault, fault_device, sizeof(int), cudaMemcpyDeviceToHost);

   printf("c_result\n");
   print_matrix<float>(c_host_wmma, MATRIX_M, MATRIX_N);

   printf("\nc_result secondary\n");
   print_matrix<float>(c_host_wmma_safe, MATRIX_M, MATRIX_N);

   printf("Fault detected %d\n", fault[0]);
 
   float wmmaTime;
   cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA);
   printf("wmma took %fms\n", wmmaTime);

   printf("\nFor a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was written as a demo only!\n\n");
   
   cudaEventDestroy(startWMMA);
   cudaEventDestroy(stopWMMA);

   cudaFree(a_fp16);
   cudaFree(b_fp16);
   cudaFree(a_fp32_device);
   cudaFree(b_fp32_device);

   cudaFree(c_wmma);
   cudaFree(c_wmma_1);
   free(c_host_wmma);
   free(c_host_wmma_1);

   cudaFree(c_wmma_safe);
   free(c_host_wmma_safe);

   free(a_fp32);
   free(b_fp32);
   free(c);

   return 0;
}

