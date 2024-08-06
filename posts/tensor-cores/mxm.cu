#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WMMA_TILE 16  // Size of a tile (sub-matrix)
#define TILE 32

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

__global__ void matrixMulAddCUDA(half* A, half* B, float* C, float* D, int N) {
    __shared__ half sharedA[TILE][TILE];
    __shared__ half sharedB[TILE][TILE];
    __shared__ float sharedC[TILE][TILE];

    int row = blockIdx.y * WMMA_TILE + threadIdx.y;
    int col = blockIdx.x * WMMA_TILE + threadIdx.x;

    printf("Row %d Col %d\n", row, col);

    float DValue = 0.0f;

    for (int tileIdx = 0; tileIdx < (N + WMMA_TILE - 1) / WMMA_TILE; ++tileIdx) {
        // Load tiles into shared memory
        if (row < N && tileIdx * WMMA_TILE + threadIdx.x < N) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * N + tileIdx * WMMA_TILE + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = __float2half(static_cast<float>(0.0f));3
        }

        if (col < N && tileIdx * WMMA_TILE + threadIdx.y < N) {
            sharedB[threadIdx.y][threadIdx.x] = B[(tileIdx * WMMA_TILE + threadIdx.y) * N + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = __float2half(static_cast<float>(0.0f));
        }

        if (row < N && col < N) {
            sharedC[threadIdx.y][threadIdx.x] = C[row * N + col];
        } else {
            sharedC[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // debuging how the shared memory is loaded
        // printf("sharedA[%d][%d]: %2.2f\n", threadIdx.y, threadIdx.x,  __half2float( sharedA[threadIdx.y][threadIdx.x] ));

        // Compute partial product for this tile
        for (int k = 0; k < WMMA_TILE; ++k) {
            DValue += __half2float(sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x]);
        }

        __syncthreads();
    }

    // Add the corresponding tile from matrix C
    if (row < N && col < N) {
        D[row * N + col] = DValue + sharedC[threadIdx.y][threadIdx.x];
    }
}

int main() {
    // Matrix size
    int N = 32;
    size_t bytes = N * N * sizeof(half);
    size_t acc_bytes = N * N * sizeof(float);

    // Allocate host memory
    half *h_A = (half*)malloc(bytes);
    half *h_B = (half*)malloc(bytes);
    float *h_C = (float*)malloc(acc_bytes);
    float *h_D = (float*)malloc(acc_bytes);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = __float2half(static_cast<float>(i));
        h_B[i] = __float2half(static_cast<float>(i));
        h_C[i] = float(i);
    }

    // Allocate device memory
    half *d_A, *d_B;
    float *d_C, *d_D;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, acc_bytes);
    cudaMalloc(&d_D, acc_bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, acc_bytes, cudaMemcpyHostToDevice);

    // Set block and grid dimensions
    dim3 blockDim(WMMA_TILE, WMMA_TILE);
    dim3 gridDim((N + WMMA_TILE - 1) / WMMA_TILE, (N + WMMA_TILE - 1) / WMMA_TILE);

    // Launch kernel
    matrixMulAddCUDA<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_D, N);

    // Copy result back to host
    cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost);

    // Print matrices
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, N, N);

    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, N, N);

    std::cout << "Matrix C:" << std::endl;
    printMatrix(h_C, N, N);

    std::cout << "Matrix D (Result A*B+C):" << std::endl;
    printMatrix(h_D, N, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return 0;
}
