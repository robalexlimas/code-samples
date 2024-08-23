#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>

#define WMMA_TILE   16  // WMMA supports 16x16 tiles
#define TILE_BLOCKS 2   // How many tiles are loaded inside the shared memory

using namespace nvcuda;

template <typename T>
void printMatrix(T* matrix, int N, int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%2.0f ", float(matrix[i * M + j]));
        }
        printf("\n");
    }
    printf("\n");
}

__device__ void loadSharedMemory(half* A, half* B, float* C, half* sharedA, half* sharedB, float* sharedC, int N, int M, int tileRow, int tileCol, int sharedTile) {
    // * tile at the warp level for loading shared memory
    for (int wmmaTile = 0; wmmaTile < TILE_BLOCKS * WMMA_TILE; wmmaTile+=WMMA_TILE) {
        // * these values can get values between 0 - TILE_BLOCK * WMMA_TILE
        // * e.g., when TILE_BLOCK is equal to 2, these values can get 0 to 31

        // * Compute the indices for loading the data inside the shared memories
        int sharedAxId = threadIdx.x * TILE_BLOCKS * WMMA_TILE;
        int sharedAyId = threadIdx.y + wmmaTile;
        int sharedAId = sharedAxId + sharedAyId;

        int sharedBxId = (threadIdx.x + wmmaTile) * WMMA_TILE;
        int sharedById = threadIdx.y;
        int sharedBId = sharedBxId + sharedById;

        // * Compute the indices for reading the data from the global memory
        int globalAx = (tileRow + threadIdx.x) * N;
        int globalAy = sharedTile + wmmaTile + threadIdx.y;
        int globalA = globalAx + globalAy;

        int globalBx = (threadIdx.x + sharedTile + wmmaTile) * N;
        int globalBy = tileCol + threadIdx.y;
        int globalB = globalBx + globalBy;

        // * Load the input values from global memory to shared memory
        sharedA[sharedAId] = A[globalA];
        sharedB[sharedBId] = B[globalB];

        // * Since C must be only loaded once
        if (sharedTile == 0 && wmmaTile == 0) {
            int sharedCxId = threadIdx.x * WMMA_TILE;
            int sharedCyId = threadIdx.y;
            int sharedCId = sharedCxId + sharedCyId;

            int globalCx = (tileRow + threadIdx.x) * N;
            int globalCy = tileCol + threadIdx.y;
            int globalC = globalCx + globalCy;

            sharedC[sharedCId] = C[globalC];
        }
    }
}

__device__ void storeGlobalMemory(float* D, float* sharedC, int N, int tileRow, int tileCol) {
    int globalCxStore = (tileRow + threadIdx.x) * N;
    int globalCyStore = tileCol + threadIdx.y;
    int globalCStore = globalCxStore + globalCyStore;

    int cxStore = threadIdx.x * WMMA_TILE;
    int cyStore = threadIdx.y;
    int cStore = cxStore + cyStore;

    D[globalCStore] = sharedC[cStore];
}

__global__ void matrixMulAddWMMA(half* A, half* B, float* C, float* D, int N, int M) {
    extern __shared__ half sharedMem[];
    half* sharedA = sharedMem;
    half* sharedB = sharedMem + TILE_BLOCKS * WMMA_TILE * WMMA_TILE;
    float* sharedC = (float*)(sharedMem + 2 * TILE_BLOCKS * WMMA_TILE * WMMA_TILE);

    int tileRow = blockIdx.y * WMMA_TILE;
    int tileCol = blockIdx.x * WMMA_TILE;

    wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // if (tileRow == 0 && tileCol == 0) {
    // * validate that the indices are inside the matrices
    if (tileRow < M && tileCol < N) {

        // * tile at the device level
        for (int sharedTile = 0; sharedTile < M; sharedTile += TILE_BLOCKS * WMMA_TILE) {

            loadSharedMemory(A, B, C, sharedA, sharedB, sharedC, N, M, tileRow, tileCol, sharedTile);

            __syncthreads();

            // * tile at the warp level for performing matrix multiplciation A * B for each segment
            for (int wmmaTile = 0; wmmaTile < TILE_BLOCKS * WMMA_TILE; wmmaTile+=WMMA_TILE) {
                wmma::load_matrix_sync(a_frag, sharedA + wmmaTile, WMMA_TILE);
                wmma::load_matrix_sync(b_frag, sharedB + wmmaTile, WMMA_TILE);

                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

        }
        // * add the C segment
        wmma::load_matrix_sync(c_frag, sharedC, WMMA_TILE, wmma::mem_row_major);

#pragma unroll
        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
        }

        __syncthreads();

        // * store the output segment from the fragment into the shared memory
        wmma::store_matrix_sync(sharedC, c_frag, WMMA_TILE, wmma::mem_row_major);

        storeGlobalMemory(D, sharedC, N, tileRow, tileCol);
    }
}

int main() {
    int N = 64;
    int M = 64;
    size_t half_bytes = N * M * sizeof(half);
    size_t float_bytes = N * M * sizeof(float);

    half* h_A = (half*)malloc(half_bytes);
    half* h_B = (half*)malloc(half_bytes);
    float* h_C = (float*)malloc(float_bytes);
    float* h_D = (float*)malloc(float_bytes);

    for (int i = 0; i < N * M; i++) {
        h_A[i] = __float2half(static_cast<float>(1));
        h_B[i] = __float2half(static_cast<float>(1));
        h_C[i] = static_cast<float>(1);
    }

    half* d_A, * d_B;
    float* d_C, * d_D;
    cudaMalloc(&d_A, half_bytes);
    cudaMalloc(&d_B, half_bytes);
    cudaMalloc(&d_C, float_bytes);
    cudaMalloc(&d_D, float_bytes);

    cudaMemcpy(d_A, h_A, half_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, half_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, float_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(WMMA_TILE, WMMA_TILE);
    dim3 gridDim(N / WMMA_TILE, M / WMMA_TILE);

    size_t sharedMemSize = 2 * TILE_BLOCKS * WMMA_TILE * WMMA_TILE * sizeof(half) + WMMA_TILE * WMMA_TILE * sizeof(float);
    matrixMulAddWMMA<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, d_D, N, M);

    cudaMemcpy(h_D, d_D, float_bytes, cudaMemcpyDeviceToHost);

    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, N, M);

    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, N, M);

    std::cout << "Matrix C:" << std::endl;
    printMatrix(h_C, N, M);

    std::cout << "Matrix D (Result A*B+C):" << std::endl;
    printMatrix(h_D, N, M);

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
