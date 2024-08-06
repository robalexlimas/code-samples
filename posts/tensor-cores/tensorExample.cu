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

if (tileRow == 0 && tileCol == 0) {

    for (int sharedTile = 0; sharedTile < (M / (TILE_BLOCKS * WMMA_TILE)); sharedTile++) {
        int sharedCol = sharedTile * TILE_BLOCKS * WMMA_TILE;
        int sharedRow = sharedTile * TILE_BLOCKS * WMMA_TILE;

        for (int wmmaTile = 0; wmmaTile < TILE_BLOCKS; wmmaTile++) {
            int wmmaRow = threadIdx.y + wmmaTile * WMMA_TILE;
            int wmmaCol = threadIdx.x + wmmaTile * WMMA_TILE;

            int aElement = (tileRow + threadIdx.y) * M + (sharedCol + wmmaCol);
            int bElement = ((sharedRow + wmmaRow) * M) + tileCol + threadIdx.x;
            
            // printf("tileRow %d tileCol %d sharedRow %d sharedCol %d wmmaRow %d wmmaCol %d elementA %d elementB %d sharedTile %d wmmaTile %d\n",tileRow, tileCol, sharedRow, sharedCol, wmmaRow, wmmaCol, aElement, bElement, sharedTile, wmmaTile);

            if (tileCol + sharedCol + wmmaCol < N) {
                int sharedAx = (threadIdx.y * TILE_BLOCKS * WMMA_TILE) + wmmaCol;
                sharedA[sharedAx] = A[aElement];
                // if (sharedTile == 1) {
                //     printf("y %d wmmaTile %d sharedRow %d sharedCol %d wmmaRow %d wmmaCol %d sharedA[%d]: %2.f\n", threadIdx.y, wmmaTile, sharedRow, sharedCol, wmmaRow, wmmaCol, sharedAx, __half2float(A[aElement])); 
                // }
            }

            if (tileRow + sharedRow + wmmaRow < N) {
                int sharedBx = (threadIdx.y * WMMA_TILE) + wmmaCol;
                sharedB[sharedBx] = B[bElement];
                if (sharedTile == 0) {
                    printf("y %d wmmaTile %d sharedRow %d sharedCol %d wmmaRow %d wmmaCol %d sharedB[%d]: %2.f\n", threadIdx.y, wmmaTile, sharedRow, sharedCol, wmmaRow, wmmaCol, sharedBx, __half2float(B[bElement])); 
                }
            }

            if (sharedTile == 0 && tileRow + sharedRow < M && tileCol + sharedCol < M) {
                sharedC[sharedRow * WMMA_TILE + sharedCol] = C[(tileCol + sharedCol) * N + tileRow + sharedRow];
                // printf("tileRow %d tileCol %d sharedRow %d sharedCol %d sharedC[%d]: %2.f\n", tileRow,tileCol, sharedRow, sharedCol, sharedRow * WMMA_TILE + sharedCol, C[(tileRow + sharedRow) * N + tileCol + sharedCol]);
            }
        }



        __syncthreads();
    }
}

    // for (int tileIdx = 0; tileIdx < (N + (TILE_BLOCKS * WMMA_TILE) - 1) / (TILE_BLOCKS * WMMA_TILE); ++tileIdx) {
    //     int tiledRow = tileRow + threadIdx.y;
    //     int tiledCol = tileCol + threadIdx.x;

        // if (tiledRow < N && tileIdx * (TILE_BLOCKS * WMMA_TILE) + threadIdx.x < N) {
        //     sharedA[threadIdx.y * (TILE_BLOCKS * WMMA_TILE) + threadIdx.x] = A[tiledRow * N + tileIdx * (TILE_BLOCKS * WMMA_TILE) + threadIdx.x];
        // } else {
        //     sharedA[threadIdx.y * (TILE_BLOCKS * WMMA_TILE) + threadIdx.x] = __float2half(0.0f);
        // }

    //     if (tiledCol < N && tileIdx * (TILE_BLOCKS * WMMA_TILE) + threadIdx.y < N) {
    //         sharedB[threadIdx.y * (TILE_BLOCKS * WMMA_TILE) + threadIdx.x] = B[(tileIdx * (TILE_BLOCKS * WMMA_TILE) + threadIdx.y) * N + tiledCol];
    //     } else {
    //         sharedB[threadIdx.y * (TILE_BLOCKS * WMMA_TILE) + threadIdx.x] = __float2half(0.0f);
    //     }

    //     if (tileIdx == 0 && tiledRow < N && tiledCol < N) {
    //         sharedC[threadIdx.y * WMMA_TILE + threadIdx.x] = C[tiledRow * N + tiledCol];
    //     } else {
    //         sharedC[threadIdx.y * WMMA_TILE + threadIdx.x] = 0.0f;
    //     }

        

    //     for (int subTileIdx = 0; subTileIdx < TILE_BLOCKS; ++subTileIdx) {
    //         wmma::load_matrix_sync(a_frag, sharedA + subTileIdx * WMMA_TILE, WMMA_TILE);
    //         wmma::load_matrix_sync(b_frag, sharedB + subTileIdx * WMMA_TILE, WMMA_TILE);
    //         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    //         for (int i = 0; i < a_frag.num_elements; i++) {
    //             printf("a_frag[%d]: %2.f\n", i, __half2float(a_frag.x[i]));
    //         }

    //         __syncthreads();
    //     }

    //     __syncthreads();
    // }

    // if (row < N && col < N) {
    //     wmma::load_matrix_sync(c_frag, sharedC, WMMA_TILE, wmma::mem_row_major);

    //     for (int i = 0; i < c_frag.num_elements; i++) {
    //         c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
    //     }

    //     wmma::store_matrix_sync(sharedC, c_frag, WMMA_TILE, wmma::mem_row_major);

    //     D[row * N + col] = sharedC[threadIdx.y * WMMA_TILE + threadIdx.x];
    // }
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
        h_A[i] = __float2half(static_cast<float>(i));
        h_B[i] = __float2half(static_cast<float>(i));
        h_C[i] = static_cast<float>(i);
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
