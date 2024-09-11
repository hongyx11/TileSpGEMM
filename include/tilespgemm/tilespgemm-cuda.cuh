#ifndef TILESPGEMM_CUDA_CUH
#define TILESPGEMM_CUDA_CUH
#include "common.h"
#include "utils.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "tilespgemm/nsparse_asm.cuh"


__forceinline__ __device__ int sum_32_shfl(int sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

__forceinline__ __device__ int sum_16_shfl(int sum)
{
#pragma unroll
    for (int mask = 1; mask < HALFWARP_SIZE; mask <<= 1)
        sum += __shfl_xor_sync(-1, sum, mask);

    return sum;
}

__forceinline__ __device__ int binary_search_exact_kernel(const int *d_array, int l, int r, int key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        int elem = d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}

__forceinline__ __device__ int binary_search_exact_kernel_v2(const int *s_array, const int *d_array, int splitter,
                                                             int l, int r, int key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        int elem = m < splitter ? s_array[m] : d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}

__forceinline__ __device__ int binary_search_exact_uchar_kernel(const unsigned char *__restrict__ d_array, int l, int r, unsigned char key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        unsigned char elem = d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return -1;
}

__forceinline__ __device__ int binary_search_right_boundary_kernel(const int *__restrict__ d_row_pointer,
                                                                   const int key_input,
                                                                   const int size)
{
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = ld_gbl_int32(d_row_pointer + median);

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start - 1;
}

__device__ __forceinline__ int intersection_binarysearch_kernel(const int *d_arraya, int abase, int astop, int lena,
                                                                const int *d_arrayb, int bbase, int bstop, int lenb,
                                                                int *d_posa, int *d_posb, int lenpos, int *d_cnt,
                                                                int lane_id, int warpsize)
{
    if (lena == 0 || lenb == 0)
    {
    }
    else if (lena < lenb)
    {
        for (int i = lane_id; i < lena; i += warpsize)
        {
            int idxa = d_arraya[abase + i];
            int res = binary_search_exact_kernel(d_arrayb + bbase, 0, lenb - 1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = i;
                    d_posb[pos] = res;
                }
            }
        }
    }
    else
    {
        for (int i = lane_id; i < lenb; i += warpsize)
        {
            int idxb = d_arrayb[bbase + i];
            int res = binary_search_exact_kernel(d_arraya + abase, 0, lena - 1, idxb);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = res;
                    d_posb[pos] = i;
                }
            }
        }
    }

    return 0;
}

__device__ __forceinline__ int intersection_binarysearch_smem_kernel(const int *d_arraya, int abase, int astop, int lena,
                                                                     const int *d_arrayb, int bbase, int bstop, int lenb,
                                                                     int *s_intersection,
                                                                     int *d_posa, int *d_posb, int lenpos, int *d_cnt,
                                                                     int lane_id, int warpsize)
{
    if (lena == 0 || lenb == 0)
    {
    }
    else if (lena < lenb)
    {
        // optimize abase and lena, by search bstart and bstop in a
        const int bendidx = d_arrayb[bstop - 1];

        int use_smem = lenb <= SMEM_INTERSECTION_LEN && lena > SMEM_INTERSECTION_TH;
        if (use_smem)
        {
            for (int i = lane_id; i < lenb; i += warpsize)
                s_intersection[i] = d_arrayb[bbase + i];
        }

        for (int i = lane_id; i < lena; i += warpsize)
        {
            int idxa = d_arraya[abase + i];
            const int *searchspace = use_smem ? s_intersection : &d_arrayb[bbase];
            int res = binary_search_exact_kernel(searchspace, 0, lenb - 1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = i;
                    d_posb[pos] = res;
                }
            }
        }
    }
    else
    {
        // optimize abase and lena, by search bstart and bstop in a
        int use_smem = lena <= SMEM_INTERSECTION_LEN && lenb > SMEM_INTERSECTION_TH;
        if (use_smem)
        {
            for (int i = lane_id; i < lena; i += warpsize)
                s_intersection[i] = d_arraya[abase + i];
        }

        for (int i = lane_id; i < lenb; i += warpsize)
        {
            int idxb = d_arrayb[bbase + i];
            const int *searchspace = use_smem ? s_intersection : &d_arraya[abase];
            int res = binary_search_exact_kernel(searchspace, 0, lena - 1, idxb);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = res;
                    d_posb[pos] = i;
                }
            }
        }
    }
    return 0;
}


inline int binary_search_right_boundary_kernel_cpu(const int *d_row_pointer,
                                            const int key_input,
                                            const int size)
{
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = d_row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start - 1;
}



__global__ void tile_spgemm_step1_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA,
                                                  int *d_blkrowptrB, int *d_blkcolidxB, int blknB,
                                                  int *d_blkrowptrC);

__global__ void tile_spgemm_step1_numeric_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA,
                                                          int *d_blkrowptrB, int *d_blkcolidxB, int blknB,
                                                          int *d_blkrowptrC, int *d_blkrowidxC, int *d_blkcolidxC,
                                                          int *d_spec_intersection_cnt, int *d_spec_intersection_posa, int *d_spec_intersection_posb);


__global__ void tile_spgemm_step3_cuda_kernel_2level(const int *d_blkrowptrA,
                                                     const int *__restrict__ d_blkcolidxA,
                                                     const int *d_nnzb_A,
                                                     MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                     unsigned char *d_blkcsr_Col_A,
                                                     unsigned char *d_blkcsr_Ptr_A,
                                                     unsigned short *d_blkmaskA,
                                                     int blkmA, int blknA, int numblkA, int nnzA,
                                                     const int *__restrict__ d_blkcolptrB,
                                                     const int *__restrict__ d_blkrowidxB,
                                                     const int *__restrict__ d_nnzb_B,
                                                     const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                     const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                     const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                     const unsigned short *__restrict__ d_blkmaskB,
                                                     int blkmB, int blknB, int numblkB, int nnzB,
                                                     int *d_blkrowidxC,
                                                     int *d_blkcolidxC,
                                                     unsigned char *d_blkcsr_Ptr_C,
                                                     int *d_nnzb_C,
                                                     unsigned short *d_blkmaskC,
                                                     int *d_blksmem_tny_cnt,
                                                     int *d_blksmem_sml_cnt,
                                                     int *d_blksmem_lrg_cnt,
                                                     int *d_blksmem_dns_cnt,
                                                     int *d_blksmem_ful_cnt,
                                                     int *d_blkid_smem_tny,
                                                     int *d_blkid_smem_sml,
                                                     int *d_blkid_smem_lrg,
                                                     int *d_blkid_smem_dns,
                                                     int *d_blkid_smem_ful,
                                                     int numblkC);


__global__ void tile_spgemm_step3_cuda_kernel_2level_halfwarp(const int *d_blkrowptrA,
                                                              const int *__restrict__ d_blkcolidxA,
                                                              const int *d_nnzb_A,
                                                              MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                              unsigned char *d_blkcsr_Col_A,
                                                              unsigned char *d_blkcsr_Ptr_A,
                                                              unsigned short *d_blkmaskA,
                                                              int blkmA, int blknA, int numblkA, int nnzA,
                                                              const int *__restrict__ d_blkcolptrB,
                                                              const int *__restrict__ d_blkrowidxB,
                                                              const int *__restrict__ d_nnzb_B,
                                                              const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                              const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                              const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                              const unsigned short *__restrict__ d_blkmaskB,
                                                              int blkmB, int blknB, int numblkB, int nnzB,
                                                              unsigned int *d_blk_intersec_bitmask_A,
                                                              unsigned int *d_blk_intersec_bitmask_B,
                                                              int blk_intersec_bitmask_len,
                                                              int *d_blkrowidxC,
                                                              int *d_blkcolidxC,
                                                              unsigned char *d_blkcsr_Ptr_C,
                                                              int *d_nnzb_C,
                                                              unsigned short *d_blkmaskC,
                                                              int *d_blksmem_tny_cnt,
                                                              int *d_blksmem_sml_cnt,
                                                              int *d_blksmem_lrg_cnt,
                                                              int *d_blksmem_dns_cnt,
                                                              int *d_blksmem_ful_cnt,
                                                              int *d_blkid_smem_tny,
                                                              int *d_blkid_smem_sml,
                                                              int *d_blkid_smem_lrg,
                                                              int *d_blkid_smem_dns,
                                                              int *d_blkid_smem_ful,
                                                              int *d_spec_intersection_cnt,
                                                              int *d_spec_intersection_posa,
                                                              int *d_spec_intersection_posb,
                                                              int numblkC);


__global__ void tile_spgemm_step3_cuda_kernel_dns_thread(const int *d_blkrowptrA,
                                                         const int *__restrict__ d_blkcolidxA,
                                                         const int *__restrict__ d_nnzb_A,
                                                         MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                         unsigned char *__restrict__ d_blkcsr_Col_A,
                                                         unsigned char *d_blkcsr_Ptr_A,
                                                         unsigned short *d_blkmaskA,
                                                         int blkmA, int blknA, int numblkA, int nnzA,
                                                         const int *__restrict__ d_blkcolptrB,
                                                         const int *__restrict__ d_blkrowidxB,
                                                         const int *__restrict__ d_nnzb_B,
                                                         const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                         const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                         const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                         const unsigned short *__restrict__ d_blkmaskB,
                                                         int blkmB, int blknB, int numblkB, int nnzB,
                                                         unsigned int *d_blk_intersec_bitmask_A,
                                                         unsigned int *d_blk_intersec_bitmask_B,
                                                         int blk_intersec_bitmask_len,
                                                         int *d_blkrowidxC,
                                                         int *d_blkcolidxC,
                                                         unsigned char *d_blkcsr_Ptr_C,
                                                         int *d_nnzb_C,
                                                         unsigned short *d_blkmaskC,
                                                         int *d_blksmem_tny_cnt,
                                                         int *d_blksmem_sml_cnt,
                                                         int *d_blksmem_lrg_cnt,
                                                         int *d_blksmem_dns_cnt,
                                                         int *d_blksmem_ful_cnt,
                                                         int *d_blkid_smem_tny,
                                                         int *d_blkid_smem_sml,
                                                         int *d_blkid_smem_lrg,
                                                         int *d_blkid_smem_dns,
                                                         int *d_blkid_smem_ful,
                                                         int *d_spec_intersection_cnt,
                                                         int *d_spec_intersection_posa,
                                                         int *d_spec_intersection_posb,
                                                         int numblkC);

template <int SMEM_MATNNZ>
__global__ void tile_spgemm_step4_cuda_kernel_smem_v3(int *d_blkrowptrA,
                                                      const int *__restrict__ d_blkcolidxA,
                                                      int *d_nnzb_A,
                                                      MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                      unsigned char *d_blkcsr_Col_A,
                                                      unsigned char *d_blkcsr_Ptr_A,
                                                      int blkmA, int blknA, int numblkA, int nnzA,
                                                      const int *__restrict__ d_blkcolptrB,
                                                      const int *__restrict__ d_blkrowidxB,
                                                      const int *__restrict__ d_nnzb_B,
                                                      const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                      const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                      const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                      int blkmB, int blknB, int numblkB, int nnzB,
                                                      int *d_blkrowidxC,
                                                      int *d_blkcolidxC,
                                                      unsigned char *d_blkcsr_Ptr_C,
                                                      unsigned char *d_blkcsr_Col_C,
                                                      MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                      int *d_nnzb_C,
                                                      unsigned short *d_blkmaskC,
                                                      int numblkC,
                                                      int *d_blkid,
                                                      int *d_spec_intersection_cnt,
                                                      int *d_spec_intersection_posa,
                                                      int *d_spec_intersection_posb);



template <int SMEM_MATNNZ>
__global__ void tile_spgemm_step4_cuda_kernel_smem_v3_halfwarp(int *d_blkrowptrA,
                                                               const int *__restrict__ d_blkcolidxA,
                                                               int *d_nnzb_A,
                                                               MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                               unsigned char *d_blkcsr_Col_A,
                                                               unsigned char *d_blkcsr_Ptr_A,
                                                               int blkmA, int blknA, int numblkA, int nnzA,
                                                               const int *__restrict__ d_blkcolptrB,
                                                               const int *__restrict__ d_blkrowidxB,
                                                               const int *__restrict__ d_nnzb_B,
                                                               const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                               const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                               const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                               int blkmB, int blknB, int numblkB, int nnzB,
                                                               int *d_blkrowidxC,
                                                               int *d_blkcolidxC,
                                                               unsigned char *d_blkcsr_Ptr_C,
                                                               unsigned char *d_blkcsr_Col_C,
                                                               MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                               int *d_nnzb_C,
                                                               unsigned short *d_blkmaskC,
                                                               int numblkC,
                                                               int *d_blkid,
                                                               int *d_spec_intersection_cnt,
                                                               int *d_spec_intersection_posa,
                                                               int *d_spec_intersection_posb);

__global__ void tile_spgemm_step4_cuda_kernel_dns_noatomic_halfwarp(int *d_blkrowptrA,
                                                                    const int *__restrict__ d_blkcolidxA,
                                                                    int *d_nnzb_A,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                                    unsigned char *d_blkcsr_Col_A,
                                                                    unsigned char *d_blkcsr_Ptr_A,
                                                                    int blkmA, int blknA, int numblkA, int nnzA,
                                                                    const int *__restrict__ d_blkcolptrB,
                                                                    const int *__restrict__ d_blkrowidxB,
                                                                    const int *__restrict__ d_nnzb_B,
                                                                    const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                                    const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                                    const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                                    int blkmB, int blknB, int numblkB, int nnzB,
                                                                    int *d_blkrowidxC,
                                                                    int *d_blkcolidxC,
                                                                    unsigned char *d_blkcsr_Ptr_C,
                                                                    unsigned char *d_blkcsr_Col_C,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                                    int *d_nnzb_C,
                                                                    unsigned short *d_blkmaskC,
                                                                    int numblkC,
                                                                    int *d_blkid,
                                                                    int *d_spec_intersection_cnt,
                                                                    int *d_spec_intersection_posa,
                                                                    int *d_spec_intersection_posb);


__global__ void tile_spgemm_step4_cuda_kernel_ful_noatomic_halfwarp(int *d_blkrowptrA,
                                                                    const int *__restrict__ d_blkcolidxA,
                                                                    int *d_nnzb_A,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                                    unsigned char *d_blkcsr_Col_A,
                                                                    unsigned char *d_blkcsr_Ptr_A,
                                                                    int blkmA, int blknA, int numblkA, int nnzA,
                                                                    const int *__restrict__ d_blkcolptrB,
                                                                    const int *__restrict__ d_blkrowidxB,
                                                                    const int *__restrict__ d_nnzb_B,
                                                                    const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                                    const unsigned char *__restrict__ d_blkcsr_Col_B,
                                                                    const unsigned char *__restrict__ d_blkcsr_Ptr_B,
                                                                    int blkmB, int blknB, int numblkB, int nnzB,
                                                                    int *d_blkrowidxC,
                                                                    int *d_blkcolidxC,
                                                                    unsigned char *d_blkcsr_Ptr_C,
                                                                    unsigned char *d_blkcsr_Col_C,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                                    int *d_nnzb_C,
                                                                    unsigned short *d_blkmaskC,
                                                                    int numblkC,
                                                                    int *d_blkid,
                                                                    int *d_spec_intersection_cnt,
                                                                    int *d_spec_intersection_posa,
                                                                    int *d_spec_intersection_posb);


void tilespgemm(SMatrix *matrixA,
                SMatrix *matrixB,
                SMatrix *matrixC,
                unsigned int *blk_intersec_bitmask_A,
                unsigned int *blk_intersec_bitmask_B,
                int blk_intersec_bitmask_len,
                double densityA,
                double densityB,
                unsigned long long int nnzCub,
                unsigned long long int *nnzC_computed,
                double *compression_rate,
                double *time_tile,
                double *gflops_tile,
                char *filename,
                double *time_step1, double *time_step2, double *time_step3, double *time_malloc);

#endif // TILESPGEMM_CUDA_CUH

