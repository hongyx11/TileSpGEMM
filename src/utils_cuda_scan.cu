#include "tilespgemm/common.h"
#include "tilespgemm/utils.hpp"
#include "tilespgemm/utils_cuda_scan.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define ITEM_PER_WARP 4
#define WARP_PER_BLOCK_SCAN 2


__global__
void init_sum_cuda_kernel(int *d_sum, int segnum)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_id == 0)
        d_sum[global_id] = 0;
    //__syncwarp();
    if (global_id != 0 && global_id < segnum)
        d_sum[global_id] = -1;
}

__global__
void exclusive_scan_cuda_kernel(int *d_key, int length, int *d_sum, int *d_id_extractor)
{
    //const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int s_key_block[WARP_PER_BLOCK_SCAN * WARP_SIZE * ITEM_PER_WARP];

    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    int *s_key = &s_key_block[local_warp_id * WARP_SIZE * ITEM_PER_WARP];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int segid = 0;
    if (!lane_id)
        segid = atomicAdd(d_id_extractor, 1);
    segid = __shfl_sync(0xffffffff, segid, 0);

    const int start = (segid * WARP_SIZE * ITEM_PER_WARP) > length ? length : (segid * WARP_SIZE * ITEM_PER_WARP);
    const int stop = ((segid + 1) * WARP_SIZE * ITEM_PER_WARP) > length ? length : ((segid + 1) * WARP_SIZE * ITEM_PER_WARP);

    if (start == stop)
        return;

    // load to smem
    for (int i = start + lane_id; i < stop; i += WARP_SIZE)
        s_key[i - start] = d_key[i];
//__syncwarp();
    // ex scan on smem
    int sum = exclusive_scan_warp_cuda(s_key, stop - start, lane_id);
//__syncwarp();
    // busy wait
    do {
        __threadfence_block();
    }
    while (d_sum[segid] == -1);

    // get incr
    int incr = d_sum[segid]; //segid ? d_sum[segid] : 0;
//__syncwarp();
    if (!lane_id)
        d_sum[segid+1] = incr + sum;
//__syncwarp();
    for (int i = start + lane_id; i < stop; i += WARP_SIZE)
        d_key[i] = s_key[i - start] + incr;
}

void exclusive_scan_device_cuda(      int   *d_key,
                                const int  length)
{
    // struct timeval tv;
/*
    printf("exclusive_scan_device_cuda, size = %i, start\n", length);
    cudaDeviceSynchronize();
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
*/
    //int *h_array1 = (int *)malloc(sizeof(int) * length);
    //cudaMemcpy(h_array1, d_key, length * sizeof(int),   cudaMemcpyDeviceToHost);
    //exclusive_scan<T>(h_array1, length);

    int *d_id_extractor;
    cudaMalloc((void **)&d_id_extractor, sizeof(int));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + sizeof(int);
    // (*index) += 1;

    cudaMemset(d_id_extractor, 0, sizeof(int));

    const int segnum = ceil((double)length / (double)(WARP_SIZE * ITEM_PER_WARP));
    int *d_sum;
    cudaMalloc((void **)&d_sum, sizeof(int) * (segnum+1));
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] + sizeof(int) * (segnum+1);
    // (*index) += 1;

//printf("segnum = %i\n", segnum);

    int num_threads = 64;
    int num_blocks = ceil ((double)(segnum+1) / (double)num_threads);
    init_sum_cuda_kernel<<<num_blocks, num_threads>>>(d_sum, segnum+1);

    num_threads = WARP_SIZE * WARP_PER_BLOCK_SCAN;
    num_blocks = ceil ((double)segnum / (double)(num_threads/WARP_SIZE));
    exclusive_scan_cuda_kernel<<<num_blocks, num_threads>>>(d_key, length, d_sum, d_id_extractor);

    cudaFree(d_id_extractor);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int);
    // (*index) += 1;

    cudaFree(d_sum);
    // gettimeofday(&tv, NULL );
    // time_node[*index] = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    // cuda_memory_use[*index] = cuda_memory_use[(*index) - 1] - sizeof(int) * (segnum+1);
    // (*index) += 1;

//    int *h_array2 = (int *)malloc(sizeof(int) * length);
//cudaMemcpy(h_array2, d_key, length * sizeof(int),   cudaMemcpyDeviceToHost);

//for (int i = 0; i < length; i++)
//    printf("[%4i] = %i, %i\n", i, h_array1[i], h_array2[i]);

/*
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("exclusive_scan_device_cuda, size = %i, RUNTIME = %4.2f ms\n", length, time);
*/
}

template<typename T>
void exclusive_scan_device_cuda_thrust(int *d_array, const int  length)
{
/*
    printf("exclusive_scan_device_cuda, size = %i, start\n", length);
    cudaDeviceSynchronize();
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
*/

//thrust::device_ptr<T> d_array_thrust(d_array);
//thrust::device_ptr<T> d_array_thrust = thrust::device_pointer_cast<T>(d_array);
//    thrust::exclusive_scan(thrust::device, d_array_thrust, d_array_thrust + length, d_array_thrust);
//    thrust::exclusive_scan(d_array, d_array + length, d_array);

//thrust::device_vector<T> d_array_thrust  (d_array,   d_array   + length);
//thrust::device_ptr<T> d_array_thrust = thrust::device_pointer_cast<T>(d_array);
//thrust::exclusive_scan(d_array_thrust.begin(), d_array_thrust.end(), d_array_thrust.begin());
thrust::exclusive_scan(thrust::device, d_array, d_array + length, d_array, 0); // in-place scan


//thrust::device_ptr<T> d_array_thrust(d_array);
//thrust::exclusive_scan(d_array_thrust, d_array_thrust + length, d_array_thrust);

//thrust::device_vector<T> d_input = d_array;
//thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());


//thrust::device_ptr<T> d_xxx = thrust::device_malloc<T>(length);
//thrust::device_vector<T> d_input = d_xxx;
//thrust::exclusive_scan(d_input.begin(), d_input.end(), d_input.begin());

//T *h_array = (T *)malloc(sizeof(T) * length);

// this part really works
//    thrust::device_ptr<int> d_array_thrust = thrust::device_pointer_cast(d_array);
//    thrust::exclusive_scan(d_array_thrust, d_array_thrust + length, d_array_thrust);


//    thrust::device_ptr<T> d_array_thrust = thrust::device_pointer_cast<T>(d_array);
//    thrust::exclusive_scan(d_array_thrust, d_array_thrust + length, d_array_thrust);

    //cudaDeviceSynchronize();
    //gettimeofday(&t2, NULL);
    //double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //printf("exclusive_scan_device_cuda, size = %i, RUNTIME = %4.2f ms\n", length, time);

/*
    T *h_array = (T *)malloc(sizeof(T) * length);
cudaMemcpy(h_array, d_array, length * sizeof(T),   cudaMemcpyDeviceToHost);
for (int i = 0; i < length; i++)
    printf("array[%i] = %i\n", i, h_array[i]);

exclusive_scan<T>(h_array, length);
cudaMemcpy(d_array, h_array, length * sizeof(T),   cudaMemcpyHostToDevice);
    free(h_array);
*/

/*
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("exclusive_scan_device_cuda, size = %i, RUNTIME = %4.2f ms\n", length, time);
*/
}



template void exclusive_scan_device_cuda_thrust<int>(int *d_array, const int  length);



