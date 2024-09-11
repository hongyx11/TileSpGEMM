#ifndef _SPGEMM_CUDA_NSPARSE_KERNEL_
#define _SPGEMM_CUDA_NSPARSE_KERNEL_


#include "common.h"
#include <cuda.h>
#include "nsparse_asm.cuh"
#include "utils_cuda_scan.cuh"

typedef struct {
    cudaStream_t *stream;
    int *bin_size;
    int *bin_offset;
    int *d_bin_size;
    int *d_bin_offset;
    int *d_row_nz;
    int *d_row_perm;
    int max_intprod;
    int max_nz;
    int *d_max;
} sfBIN;

void init_bin(sfBIN *bin, int M);

void release_bin(sfBIN bin);

__global__ void set_intprod_num(int *d_arpt, int *d_acol,
                                const int* __restrict__ d_brpt,
                                int *d_row_intprod, int *d_max_intprod,
                                int M);

__global__ void set_bin(int *d_row_nz, int *d_bin_size, int *d_max,
                        int M, int min, int mmin);

__global__ void init_row_perm(int *d_permutation, int M);

__global__ void set_row_perm(int *d_bin_size, int *d_bin_offset,
                             int *d_max_row_nz, int *d_row_perm,
                             int M, int min, int mmin);

void set_max_bin(int *d_arpt, int *d_acol, int *d_brpt, sfBIN *bin, int M);

void set_min_bin(sfBIN *bin, int M);

__global__ void init_value(real *d_val, int nz);

__global__ void init_check(int *d_check, int nz);

__global__ void set_row_nz_bin_pwarp(const int *d_arpt, const int *d_acol,
                                     const int* __restrict__ d_brpt,
                                     const int* __restrict__ d_bcol,
                                     const int *d_row_perm,
                                     int *d_row_nz,
                                     int bin_offset, int M);

template <int SH_ROW>
__global__ void set_row_nz_bin_each(const int *d_arpt, const int *d_acol,
                                    const int* __restrict__ d_brpt,
                                    const int* __restrict__ d_bcol,
                                    const int *d_row_perm,
                                    int *d_row_nz, int bin_offset, int M);

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb(const int *d_arpt, const int *d_acol,
                                       const int* __restrict__ d_brpt,
                                       const int* __restrict__ d_bcol,
                                       int *d_row_perm, int *d_row_nz,
                                       int bin_offset, int M);

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb_large(const int *d_arpt, const int *d_acol,
                                             const int* __restrict__ d_brpt,
                                             const int* __restrict__ d_bcol,
                                             int *d_row_perm, int *d_row_nz,
                                             int *d_fail_count, int *d_fail_perm,
                                             int bin_offset, int M);

__global__ void set_row_nz_bin_each_gl(const int *d_arpt, const int *d_acol,
                                       const int* __restrict__ d_brpt,
                                       const int* __restrict__ d_bcol,
                                       const int *d_row_perm,
                                       int *d_row_nz, int *d_check,
                                       int max_row_nz, int bin_offset, int M);

void set_row_nnz(int *d_arpt, int *d_acol,
                 int *d_brpt, int *d_bcol,
                 int *d_crpt,
                 sfBIN *bin,
                 int M, int *nnz);

__global__ void calculate_value_col_bin_pwarp(const int *d_arpt,
                                              const int *d_acol,
                                              const real *d_aval,
                                              const int* __restrict__ d_brpt,
                                              const int* __restrict__ d_bcol,
                                              const real* __restrict__ d_bval,
                                              int *d_crpt,
                                              int *d_crow, 
                                              int *d_ccol,
                                              real *d_cval,
                                              const int *d_row_perm,
                                              int *d_nz,
                                              int bin_offset,
                                              int bin_size);

template <int SH_ROW>
__global__ void calculate_value_col_bin_each_tb(const int *d_arpt,
                                                const int *d_acol,
                                                const real *d_aval,
                                                const int* __restrict__ d_brpt,
                                                const int* __restrict__ d_bcol,
                                                const real* __restrict__ d_bval,
                                                int *d_crpt,
                                                int *d_crow, 
                                                int *d_ccol,
                                                real *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int bin_offset,
                                                int bin_size);

__global__ void calculate_value_col_bin_each_gl(const int *d_arpt,
                                                const int *d_acol,
                                                const real *d_aval,
                                                const int* __restrict__ d_brpt,
                                                const int* __restrict__ d_bcol,
                                                const real* __restrict__ d_bval,
                                                int *d_crpt,
                                                int *d_crow, 
                                                int *d_ccol,
                                                real *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int *d_check,
                                                real *d_value,
                                                int max_row_nz,
                                                int bin_offset,
                                                int M);

void calculate_value_col_bin(int *d_arpt, int *d_acol, real *d_aval,
                             int *d_brpt, int *d_bcol, real *d_bval,
                             int *d_crpt, int *d_crow, int *d_ccol, real *d_cval,
                             sfBIN *bin,
                             int M, int N);

#endif




