#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Size of a tile (sub-matrix)

template <typename T>
void printMatrix(T* matrix, int N, int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%2.2f ", float( matrix[i * M + j] ));
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void matrixMulAddCUDA(float* A, float* B, float* C, float* D, int N) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedC[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float DValue = 0.0f;

    for (int tileIdx = 0; tileIdx < (N + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        // Load tiles into shared memory
        if (row < N && tileIdx * TILE_SIZE + threadIdx.x < N) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * N + tileIdx * TILE_SIZE + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tileIdx * TILE_SIZE + threadIdx.y < N) {
            sharedB[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (row < N && col < N) {
            sharedC[threadIdx.y][threadIdx.x] = C[row * N + col];
        } else {
            sharedC[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            DValue += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
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
    int N = 1024;
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = float(1.0);
        h_B[i] = float(1.0);
        h_C[i] = float(1.0);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_D, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);

    // Set block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

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
