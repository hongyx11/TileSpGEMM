#ifndef _SCAN_CUDA_UTILS_
#define _SCAN_CUDA_UTILS_

#include "common.h"
#include "utils.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
// #include <thrust/device_malloc.h>
// #include <thrust/device_free.h>
// #include <thrust/device_vector.h>
// #include <thrust/execution_policy.h>
// #include <thrust/device_ptr.h>
// #include <thrust/scan.h>

#define ITEM_PER_WARP 4
#define WARP_PER_BLOCK_SCAN 2

// inclusive scan
__forceinline__ __device__
int scan_32_shfl(      int x,
                 const int lane_id)
{
    int y = __shfl_up_sync(0xffffffff, x, 1);
    x = lane_id >= 1 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 2);
    x = lane_id >= 2 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 4);
    x = lane_id >= 4 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 8);
    x = lane_id >= 8 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 16);
    x = lane_id >= 16 ? x + y : x;

    return x;
}

__forceinline__ __device__
int scan_16_shfl(      int x,
                 const int lane_id)
{
    int y = __shfl_up_sync(0xffffffff, x, 1);
    x = lane_id >= 1 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 2);
    x = lane_id >= 2 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 4);
    x = lane_id >= 4 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 8);
    x = lane_id >= 8 ? x + y : x;
    //y = __shfl_up_sync(0xffffffff, x, 16);
    //x = lane_id >= 16 ? x + y : x;

    return x;
}

template<typename iT>
__inline__ __device__
int exclusive_scan_warp_cuda(       iT  *key,
                              const  int  size,
                              const  int  lane_id)
{
    const int loop = ceil((float)size/(float)WARP_SIZE);
    int sum = 0;

    // all rounds except the last
    for (int li = 0; li < loop - 1; li++)
    {
        const int nid = li * WARP_SIZE + lane_id;
        const int lb = key[nid];
        const int lb_scan = scan_32_shfl(lb, lane_id); // this scan is inclusive
        key[nid] = lb_scan - lb + sum;
        sum += __shfl_sync(0xffffffff, lb_scan, WARP_SIZE-1); //__syncwarp();// sum of all values
    }

    // the last round
    const int len_processed = (loop - 1) * WARP_SIZE;
    const int len_last_round = size - len_processed;
    const int lb = lane_id < len_last_round ? key[len_processed + lane_id] : 0;
    const int lb_scan = scan_32_shfl(lb, lane_id); // this scan is inclusive
    if (lane_id < len_last_round)
        key[len_processed + lane_id] = lb_scan - lb + sum;
    sum += __shfl_sync(0xffffffff, lb_scan, WARP_SIZE-1); // sum of all values

    return sum;
}

template<typename iT>
__inline__ __device__
int exclusive_scan_block_cuda(       iT  *key,
                                     int *s_warpsync,
                              const  int  size,
                              const  int  warp_id,
                              const  int  warp_num,
                              const  int  lane_id)
{
    const int wnum = ceil((float)size / (float)WARP_SIZE);
    int lb, lb_scan;

    for (int wi = warp_id; wi < wnum; wi += warp_num)
    {
        const int pos = wi * WARP_SIZE + lane_id;
        lb = wi == wnum - 1 ? (pos < size ? key[pos] : 0) : key[pos];
        lb_scan = scan_32_shfl(lb, lane_id); // this scan is inclusive
        if (pos < size) key[pos] = lb_scan - lb;
        if (lane_id == WARP_SIZE-1) s_warpsync[wi] = lb_scan;
    }
    __syncthreads();
    //if (print_tag) printf("step1 key[%i] = %i\n", warp_id*WARP_SIZE+lane_id, key[warp_id*WARP_SIZE+lane_id]);
    //__syncthreads();

    if (!warp_id)
    {
        lb = lane_id < wnum ? s_warpsync[lane_id] : 0;
        lb_scan = scan_32_shfl(lb, lane_id); // this scan is inclusive
        if (lane_id < wnum) s_warpsync[lane_id] = lb_scan;
        //s_warpsync[lane_id] = lb_scan - lb;
    }
    __syncthreads();
    //if (print_tag && !warp_id) printf("before s_warpsync[%i] = %i\n", lane_id, s_warpsync[lane_id]);
    //__syncthreads();

    const int sum = s_warpsync[wnum-1];
    __syncthreads();

    if (!warp_id)
    {
        if (lane_id < wnum) s_warpsync[lane_id] = lb_scan - lb;
    }
    __syncthreads();
    //if (print_tag && !warp_id) printf("after s_warpsync[%i] = %i\n", lane_id, s_warpsync[lane_id]);
    //__syncthreads();

    for (int wi = warp_id; wi < wnum; wi += warp_num)
    {
        const int pos = wi * WARP_SIZE + lane_id;
        lb = wi == wnum - 1 ? (pos < size ? key[pos] : 0) : key[pos];
        if (pos < size) key[pos] = lb + s_warpsync[wi];
    }
    //if (print_tag) printf("step 2 key[%i] = %i\n", warp_id*WARP_SIZE+lane_id, key[warp_id*WARP_SIZE+lane_id]);
    //__syncthreads();

    return sum;
}

__global__
void init_sum_cuda_kernel(int *d_sum, int segnum);

__global__
void exclusive_scan_cuda_kernel(int *d_key, int length, int *d_sum, int *d_id_extractor);

void exclusive_scan_device_cuda(int *d_key, const int length);

template<typename T>
void exclusive_scan_device_cuda_thrust(int *d_array, const int length);

#endif




