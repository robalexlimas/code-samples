#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>

#define TILE_SIZE 16  // WMMA supports 16x16 tiles

using namespace nvcuda;

template <typename T>
void printMatrix(T* matrix, int N, int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%2.0f ", float( matrix[i * M + j] ));
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void matrixMulAddWMMA(half* A, half* B, float* C, float* D, int N) {
    extern __shared__ half sharedMem[];
    half* sharedA = sharedMem;
    half* sharedB = sharedMem + TILE_SIZE * TILE_SIZE;
    float* sharedC = (float*)(sharedMem + 2 * TILE_SIZE * TILE_SIZE);

    int tileRow = blockIdx.y * TILE_SIZE;
    int tileCol = blockIdx.x * TILE_SIZE;
    int row = tileRow + threadIdx.y;
    int col = tileCol + threadIdx.x;

    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> acc_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int tileIdx = 0; tileIdx < (N + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        int tiledRow = tileRow + threadIdx.y;
        int tiledCol = tileCol + threadIdx.x;

        if (tiledRow < N && tileIdx * TILE_SIZE + threadIdx.x < N) {
            sharedA[threadIdx.y * TILE_SIZE + threadIdx.x] = A[tiledRow * N + tileIdx * TILE_SIZE + threadIdx.x];
        } else {
            sharedA[threadIdx.y * TILE_SIZE + threadIdx.x] = __float2half(0.0f);
        }

        if (tiledCol < N && tileIdx * TILE_SIZE + threadIdx.y < N) {
            sharedB[threadIdx.y * TILE_SIZE + threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * N + tiledCol];
        } else {
            sharedB[threadIdx.y * TILE_SIZE + threadIdx.x] = __float2half(0.0f);
        }

        if (tiledRow < N && tiledCol < N) {
            sharedC[threadIdx.y * TILE_SIZE + threadIdx.x] = C[tiledRow * N + tiledCol];
        } else {
            sharedC[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, sharedA, TILE_SIZE);
        wmma::load_matrix_sync(b_frag, sharedB, TILE_SIZE);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    if (row < N && col < N) {
        wmma::load_matrix_sync(c_frag, sharedC, TILE_SIZE, wmma::mem_row_major);
    
#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
      }

        // Store the output
        wmma::store_matrix_sync(sharedC, c_frag, TILE_SIZE, wmma::mem_row_major);

        D[row * N + col] = sharedC[threadIdx.y * TILE_SIZE + threadIdx.x];
    }
}

int main() {
    int N = 128;
    size_t half_bytes = N * N * sizeof(half);
    size_t float_bytes = N * N * sizeof(float);

    half *h_A = (half*)malloc(half_bytes);
    half *h_B = (half*)malloc(half_bytes);
    float *h_C = (float*)malloc(float_bytes);
    float *h_D = (float*)malloc(float_bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = __float2half(static_cast<float>(1.0));
        h_B[i] = __float2half(static_cast<float>(1.0));
        h_C[i] = static_cast<float>(1.0);
    }

    half *d_A, *d_B;
    float *d_C, *d_D;
    cudaMalloc(&d_A, half_bytes);
    cudaMalloc(&d_B, half_bytes);
    cudaMalloc(&d_C, float_bytes);
    cudaMalloc(&d_D, float_bytes);

    cudaMemcpy(d_A, h_A, half_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, half_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, float_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    size_t sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(half) + TILE_SIZE * TILE_SIZE * sizeof(float);
    matrixMulAddWMMA<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, d_D, N);

    cudaMemcpy(h_D, d_D, float_bytes, cudaMemcpyDeviceToHost);

    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, N, N);

    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, N, N);

    std::cout << "Matrix C:" << std::endl;
    printMatrix(h_C, N, N);

    std::cout << "Matrix D (Result A*B+C):" << std::endl;
    printMatrix(h_D, N, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}

