#ifndef _UTILS_
#define _UTILS_

#include "common.h"

void binary_search_right_boundary_item_kernel(const MAT_PTR_TYPE *row_pointer, 
                                              const MAT_PTR_TYPE key_input, 
                                              const int size, 
                                              int *colpos, 
                                              MAT_PTR_TYPE *nnzpos);

void exclusive_scan(MAT_PTR_TYPE *input, int length);

void exclusive_scan_char(unsigned char *input, int length);

void swap_key(int *a, int *b);

void swap_val(MAT_VAL_TYPE *a, MAT_VAL_TYPE *b);

int partition_key_val_pair(int *key, MAT_VAL_TYPE *val, int length, int pivot_index);

void quick_sort_key_val_pair(int *key, MAT_VAL_TYPE *val, int length);

int partition_key(int *key, int length, int pivot_index);

void quick_sort_key(int *key, int length);

void matrix_transposition(const int m,
                          const int n,
                          const MAT_PTR_TYPE nnz,
                          const MAT_PTR_TYPE *csrRowPtr,
                          const int *csrColIdx,
                          const MAT_VAL_TYPE *csrVal,
                          int *cscRowIdx,
                          MAT_PTR_TYPE *cscColPtr,
                          MAT_VAL_TYPE *cscVal);

#endif



