#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <string>
#include <nvml.h>

#define WMMA_TILE   16  // WMMA supports 16x16 tiles
// #define DEBUG 0

using namespace nvcuda;

void checkNvmlError(nvmlReturn_t result, const char* msg) {
    if (result != NVML_SUCCESS) {
        std::cerr << "Error: " << msg << " - " << nvmlErrorString(result) << std::endl;
        exit(1);
    }
}

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

__device__ void loadSharedMemory(half* A, half* B, float* C, half* sharedA, half* sharedB, float* sharedC, int N, int M, int tileRow, int tileCol, int sharedTile, int TILE_BLOCKS) {
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

__device__ void updateBFrag(
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag_0, 
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag_1) {
        // * SAFE MODE TCU 0 <- TCU 1
        for(int i=0; i < b_frag_0.num_elements / 4; i++) {
            b_frag_0.x[i] = b_frag_0.x[i + 4];
        }

        // * SAFE MODE TCU 1 <- TCU 0
        for(int i=0; i < b_frag_1.num_elements / 4; i++) {
            b_frag_1.x[i + 4] = b_frag_1.x[i];
        }
}

__device__ void checkCFrag(
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> c_frag_0, 
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> c_frag_1,
    int *fault) {
        for(int i=0; i < 4; i++) {
            if (c_frag_0.x[i] != c_frag_0.x[i + 4]) {
                printf("Something happened frag0!!!\n");
                fault[0] = -1;
            }
            if (c_frag_1.x[i] != c_frag_1.x[i + 4]) {
                printf("Something happened frag1!!!\n");
                fault[0] = 1;
            }
        }
}

__device__ int wmma_diagnosis(
   wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> fragA,
   wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> fragB,
   const float*  fragC,
   int N, int M) {

    int laneid;
    asm("mov.u32 %0, %laneid;" :"=r"(laneid));
    int bCol = (int)(laneid / 4);
    int bRow = (int)(laneid % 4);

    // Declare the fragments
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ half diagnosis[256];
    __shared__ float Cdiagnosis[256];

    // * fill the b diagnosis within the fragment data
    for (int i = 0; i < 2; i++) {
        diagnosis[((bRow * 2) + i) * WMMA_TILE + bCol] = fragB.x[i];
        diagnosis[((bRow * 2) + i  + 8) * WMMA_TILE + bCol] = fragB.x[i + 2];

        diagnosis[((bRow * 2) + i) * WMMA_TILE + bCol + 8] = fragB.x[i + 4];
        diagnosis[((bRow * 2) + i  + 8) * WMMA_TILE + bCol + 8] = fragB.x[i + 6];
    }

    __syncthreads();

    // * Copy the columns into the following ones
    if (bCol < 4) {
        for (int i = 0; i < 16; i++) {
            diagnosis[i * WMMA_TILE + bCol + 4] = diagnosis[i * WMMA_TILE + bCol];
            diagnosis[i * WMMA_TILE + bCol + 12] = diagnosis[i * WMMA_TILE + bCol + 8];
        }
    }

    wmma::load_matrix_sync(b_frag, diagnosis, WMMA_TILE);
    wmma::mma_sync(acc_frag, fragA, b_frag, acc_frag);

    __syncthreads();

    // * identification
    int cRow = (int)(laneid / 4);
    int cCol = (int)(laneid % 4);

    // * fill the diagnosis matrix with the c fragment data
    for (int i = 0; i < 2; i++) {
        Cdiagnosis[((cRow) * WMMA_TILE) + (cCol * 2) + i] = acc_frag.x[i];
        Cdiagnosis[((cRow + 8) * WMMA_TILE) + (cCol * 2) + i] = acc_frag.x[i + 2];
        Cdiagnosis[((cRow) * WMMA_TILE) + (cCol * 2) + 8 + i] = acc_frag.x[i + 4];
        Cdiagnosis[((cRow + 8) * WMMA_TILE) + (cCol * 2) + 8 + i] = acc_frag.x[i + 6];
    }

    __syncthreads();

    // * diagnosis
    if (laneid == 0) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 4; j++) {
                if (Cdiagnosis[i * WMMA_TILE + j] != Cdiagnosis[i * WMMA_TILE + j + 4]) {
                    // * faulty TCU0
                    return 0;
                }
                if (Cdiagnosis[i * WMMA_TILE + j + 8] != Cdiagnosis[i * WMMA_TILE + j + 12]) { 
                    // * faulty TCU1
                    return 1;
                }
            }
        }
    }
    // * the fault has not been detected
    return -1;
}


__global__ void matrixMulAddWMMACorrection(half* A, half* B, float* C, float* D, int N, int M, int *fault, int* tcu, int TILE_BLOCKS) {
    extern __shared__ half sharedMem[];
    half* sharedA = sharedMem;
    half* sharedB = sharedMem + TILE_BLOCKS * WMMA_TILE * WMMA_TILE;
    float* sharedC = (float*)(sharedMem + 2 * TILE_BLOCKS * WMMA_TILE * WMMA_TILE);

    int tileRow = blockIdx.y * WMMA_TILE;
    int tileCol = blockIdx.x * WMMA_TILE;

    wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> a_frag;

    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag_0; 
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag_1;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag_0;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag_1;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> c_frag_0;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> c_frag_1;

    wmma::fill_fragment(acc_frag_0, 0.0f);
    wmma::fill_fragment(acc_frag_1, 0.0f);

    // if (tileRow == 0 && tileCol == 0) {
    // * validate that the indices are inside the matrices
    if (tileRow < M && tileCol < N) {

        // * tile at the device level
        for (int sharedTile = 0; sharedTile < M; sharedTile += TILE_BLOCKS * WMMA_TILE) {

            loadSharedMemory(A, B, C, sharedA, sharedB, sharedC, N, M, tileRow, tileCol, sharedTile, TILE_BLOCKS);

            __syncthreads();

            // * tile at the warp level for performing matrix multiplciation A * B for each segment
            for (int wmmaTile = 0; wmmaTile < TILE_BLOCKS * WMMA_TILE; wmmaTile+=WMMA_TILE) {
                wmma::load_matrix_sync(a_frag, sharedA + wmmaTile, WMMA_TILE);

                wmma::load_matrix_sync(b_frag_0, sharedB, WMMA_TILE);
                wmma::load_matrix_sync(b_frag_1, sharedB, WMMA_TILE);

                updateBFrag(b_frag_0, b_frag_1);

                wmma::mma_sync(acc_frag_0, a_frag, b_frag_0, acc_frag_0);
                wmma::mma_sync(acc_frag_1, a_frag, b_frag_1, acc_frag_1);
            }
        }
        // * add the C segment
        wmma::load_matrix_sync(c_frag_0, sharedC, WMMA_TILE, wmma::mem_row_major);
        wmma::load_matrix_sync(c_frag_1, sharedC, WMMA_TILE, wmma::mem_row_major);

#pragma unroll
        for(int i=0; i < c_frag_0.num_elements; i++) {
            c_frag_0.x[i] = acc_frag_0.x[i] + c_frag_0.x[i];
        }
#pragma unroll
        for(int i=0; i < c_frag_1.num_elements; i++) {
            c_frag_1.x[i] = acc_frag_1.x[i] + c_frag_1.x[i];
        }

        __syncthreads();

        checkCFrag(c_frag_0, c_frag_1, fault);

        __syncthreads();

        if (fault[0] == -1) {
            tcu[0] = wmma_diagnosis(a_frag, b_frag_0, sharedC, N, M);
        } else if  (fault[0] == 1) {
            tcu[0] = wmma_diagnosis(a_frag, b_frag_1, sharedC, N, M);
        }

        // * store the output segment from the fragment into the shared memory
        __syncthreads();

        if (tcu[0] == 0){
            // * faulty TCU 0, means consider only the TCU 1 data
            for(int i=0; i < 4; i++) {
                c_frag_0.x[i] = c_frag_1.x[i + 4];
            }
            wmma::store_matrix_sync(sharedC, c_frag_0, WMMA_TILE, wmma::mem_row_major);
        } else {
            // * faulty TCU 1, means consider only the TCU 0 data
            for(int i=0; i < 4; i++) {
                c_frag_1.x[i + 4] = c_frag_0.x[i];
            }
            wmma::store_matrix_sync(sharedC, c_frag_1, WMMA_TILE, wmma::mem_row_major);
        }

        storeGlobalMemory(D, sharedC, N, tileRow, tileCol);
    }
}

__global__ void matrixMulAddWMMADetection(half* A, half* B, float* C, float* D, int N, int M, int *fault, int TILE_BLOCKS) {
    extern __shared__ half sharedMem[];
    half* sharedA = sharedMem;
    half* sharedB = sharedMem + TILE_BLOCKS * WMMA_TILE * WMMA_TILE;
    float* sharedC = (float*)(sharedMem + 2 * TILE_BLOCKS * WMMA_TILE * WMMA_TILE);

    int tileRow = blockIdx.y * WMMA_TILE;
    int tileCol = blockIdx.x * WMMA_TILE;

    wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> a_frag;

    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag_0; 
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag_1;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag_0;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag_1;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> c_frag_0;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> c_frag_1;

    wmma::fill_fragment(acc_frag_0, 0.0f);
    wmma::fill_fragment(acc_frag_1, 0.0f);

    // if (tileRow == 0 && tileCol == 0) {
    // * validate that the indices are inside the matrices
    if (tileRow < M && tileCol < N) {

        // * tile at the device level
        for (int sharedTile = 0; sharedTile < M; sharedTile += TILE_BLOCKS * WMMA_TILE) {

            loadSharedMemory(A, B, C, sharedA, sharedB, sharedC, N, M, tileRow, tileCol, sharedTile, TILE_BLOCKS);

            __syncthreads();

            // * tile at the warp level for performing matrix multiplciation A * B for each segment
            for (int wmmaTile = 0; wmmaTile < TILE_BLOCKS * WMMA_TILE; wmmaTile+=WMMA_TILE) {
                wmma::load_matrix_sync(a_frag, sharedA + wmmaTile, WMMA_TILE);

                wmma::load_matrix_sync(b_frag_0, sharedB, WMMA_TILE);
                wmma::load_matrix_sync(b_frag_1, sharedB, WMMA_TILE);

                updateBFrag(b_frag_0, b_frag_1);

                wmma::mma_sync(acc_frag_0, a_frag, b_frag_0, acc_frag_0);
                wmma::mma_sync(acc_frag_1, a_frag, b_frag_1, acc_frag_1);
            }
        }
        // * add the C segment
        wmma::load_matrix_sync(c_frag_0, sharedC, WMMA_TILE, wmma::mem_row_major);
        wmma::load_matrix_sync(c_frag_1, sharedC, WMMA_TILE, wmma::mem_row_major);

#pragma unroll
        for(int i=0; i < c_frag_0.num_elements; i++) {
            c_frag_0.x[i] = acc_frag_0.x[i] + c_frag_0.x[i];
        }
#pragma unroll
        for(int i=0; i < c_frag_1.num_elements; i++) {
            c_frag_1.x[i] = acc_frag_1.x[i] + c_frag_1.x[i];
        }

        __syncthreads();

        checkCFrag(c_frag_0, c_frag_1, fault);

        // * store the output segment from the fragment into the shared memory
        wmma::store_matrix_sync(sharedC, c_frag_0, WMMA_TILE, wmma::mem_row_major);

        storeGlobalMemory(D, sharedC, N, tileRow, tileCol);
    }
}

__global__ void matrixMulAddWMMA(half* A, half* B, float* C, float* D, int N, int M, int TILE_BLOCKS) {
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

            loadSharedMemory(A, B, C, sharedA, sharedB, sharedC, N, M, tileRow, tileCol, sharedTile, TILE_BLOCKS);

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

// Timing wrapper function
template <typename Func>
float measureKernelTime(Func kernel) {
    cudaEvent_t start, stop;
    float elapsedTime;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel (passed as a lambda)
    kernel();

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;  // Return time in milliseconds
}

void printfStatistics(int method, int N, float timeMs, int sharedMemory, float power, float powerBefore, int temperatureBefore, int temperatureAfter, int clockBefore, int clockAfter, nvmlPstates_t pStateAfter) {
    // * Calculate TFLOPs
    double flops = 2.0 * N * N * N + N * N;
    double elapsedTimeInSeconds = timeMs / 1000.0;
    double tflops = flops / (elapsedTimeInSeconds * 1e12);

    // * method, size, time, shared, flops, tflops
    // * method 0 - normal, 1 - detection, 2 - correction
    if (method == 0) printf("normal,%d,%2.4f,%d,%2.4f,%2.4f,%2.4f,%2.4f,%d,%d,%d,%d,%d\n", N, timeMs, sharedMemory, flops, tflops, powerBefore, power, temperatureBefore, temperatureAfter, clockBefore, clockAfter, (unsigned int)pStateAfter);
    if (method == 1) printf("detection,%d,%2.4f,%d,%2.4f,%2.4f,%2.4f,%2.4f,%d,%d,%d,%d,%d\n", N, timeMs, sharedMemory, flops, tflops, powerBefore, power, temperatureBefore, temperatureAfter, clockBefore, clockAfter, (unsigned int)pStateAfter);
    if (method == 2) printf("correction,%d,%2.4f,%d,%2.4f,%2.4f,%2.4f,%2.4f,%d,%d,%d,%d,%d\n", N, timeMs, sharedMemory, flops, tflops, powerBefore, power, temperatureBefore, temperatureAfter, clockBefore, clockAfter, (unsigned int)pStateAfter);
    
}

int main(int argc, char* argv[]) {
    // * Must be multiples of 16 for wmma code to work
    if (argv[1] == "") {
        printf("Matrix size must be initialized, and it must be multiples of 16");
        return 0;
    }

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    checkNvmlError(result, "Failed to initialize NVML");

    // Get the number of GPUs
    unsigned int deviceCount;
    result = nvmlDeviceGetCount(&deviceCount);
    checkNvmlError(result, "Failed to get device count");

    // Select the first GPU (you can modify this to target a different GPU)
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    checkNvmlError(result, "Failed to get handle for device");

    const int M = atoi(argv[1]);
    const int N = atoi(argv[1]);
    const int TILE_BLOCKS = atoi(argv[2]);

    size_t half_bytes = N * M * sizeof(half);
    size_t float_bytes = N * M * sizeof(float);

    int *fault;
    int *fault_device;

    int *tcu;
    int *tcu_device;

    fault = (int*)malloc(sizeof(int));
    fault[0] = 0;
    cudaMalloc((void**)&fault_device, sizeof(int));

    tcu = (int*)malloc(sizeof(int));
    tcu[0] = -1;
    cudaMalloc((void**)&tcu_device, sizeof(int));

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

    // Record power usage before kernel execution
    unsigned int powerBefore;
    result = nvmlDeviceGetPowerUsage(device, &powerBefore);
    checkNvmlError(result, "Failed to get power usage");

    // Get temperature before kernel execution
    unsigned int temperatureBefore;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperatureBefore);
    checkNvmlError(result, "Failed to get temperature");

    // Get GPU clock frequency before kernel execution
    unsigned int clockBefore;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clockBefore);
    checkNvmlError(result, "Failed to get GPU clock frequency");
    
    // Get P-state (performance state) before kernel execution
    nvmlPstates_t pStateBefore;
    result = nvmlDeviceGetPerformanceState(device, &pStateBefore);
    checkNvmlError(result, "Failed to get performance state");

    // * Measure kernel execution time using the wrapper
    float timeMs = measureKernelTime([&]() {
        matrixMulAddWMMADetection<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, d_D, N, M, fault_device, TILE_BLOCKS);
    });

    // Record power usage after kernel execution
    unsigned int powerAfter;
    result = nvmlDeviceGetPowerUsage(device, &powerAfter);
    checkNvmlError(result, "Failed to get power usage");

    // Get temperature after kernel execution
    unsigned int temperatureAfter;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperatureAfter);
    checkNvmlError(result, "Failed to get temperature");

    // Get GPU clock frequency after kernel execution
    unsigned int clockAfter;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clockAfter);
    checkNvmlError(result, "Failed to get GPU clock frequency");

    // Get P-state (performance state) after kernel execution
    nvmlPstates_t pStateAfter;
    result = nvmlDeviceGetPerformanceState(device, &pStateAfter);
    checkNvmlError(result, "Failed to get performance state");

    printfStatistics(1, N, timeMs, sharedMemSize, powerAfter / 1000.0, powerBefore / 1000.0, temperatureBefore, temperatureAfter, clockBefore, clockAfter, pStateAfter);

    // * Measure kernel execution time using the wrapper
    timeMs = measureKernelTime([&]() {
        matrixMulAddWMMACorrection<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, d_D, N, M, fault_device, tcu_device, TILE_BLOCKS);
    });

    result = nvmlDeviceGetPowerUsage(device, &powerAfter);
    checkNvmlError(result, "Failed to get power usage");

    // Get temperature after kernel execution
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperatureAfter);
    checkNvmlError(result, "Failed to get temperature");

    // Get GPU clock frequency after kernel execution
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clockAfter);
    checkNvmlError(result, "Failed to get GPU clock frequency");

    // Get P-state (performance state) after kernel execution
    result = nvmlDeviceGetPerformanceState(device, &pStateAfter);
    checkNvmlError(result, "Failed to get performance state");

    printfStatistics(2, N, timeMs, sharedMemSize, powerAfter / 1000.0, powerBefore / 1000.0, temperatureBefore, temperatureAfter, clockBefore, clockAfter, pStateAfter);

    // * Measure kernel execution time using the wrapper
    timeMs = measureKernelTime([&]() {
        matrixMulAddWMMA<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, d_D, N, M, TILE_BLOCKS);
    });

    result = nvmlDeviceGetPowerUsage(device, &powerAfter);
    checkNvmlError(result, "Failed to get power usage");

    // Get temperature after kernel execution
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperatureAfter);
    checkNvmlError(result, "Failed to get temperature");

    // Get GPU clock frequency after kernel execution
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clockAfter);
    checkNvmlError(result, "Failed to get GPU clock frequency");

    // Get P-state (performance state) after kernel execution
    result = nvmlDeviceGetPerformanceState(device, &pStateAfter);
    checkNvmlError(result, "Failed to get performance state");

    printfStatistics(0, N, timeMs, sharedMemSize, powerAfter / 1000.0, powerBefore / 1000.0, temperatureBefore, temperatureAfter, clockBefore, clockAfter, pStateAfter);

    cudaMemcpy(h_D, d_D, float_bytes, cudaMemcpyDeviceToHost);

    cudaMemcpy(fault, fault_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tcu, tcu_device, sizeof(int), cudaMemcpyDeviceToHost);

    if (fault[0] != 0) {
      printf("Fault detected at the TCU %d\n", tcu[0]);
    }

#ifdef DEBUG
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, N, M);

    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, N, M);

    std::cout << "Matrix C:" << std::endl;
    printMatrix(h_C, N, M);

    std::cout << "Matrix D (Result A*B+C):" << std::endl;
    printMatrix(h_D, N, M);
#endif

    // Shutdown NVML
    result = nvmlShutdown();
    checkNvmlError(result, "Failed to shutdown NVML");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(tcu_device);
    cudaFree(fault_device);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(tcu);
    free(fault);

    return 0;
}
