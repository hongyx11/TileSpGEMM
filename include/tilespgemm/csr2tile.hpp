#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "common.h"
#include "utils.hpp"

// Function declarations
void step1_kernel(SMatrix *matrix);
void step2_kernel(SMatrix *matrix, unsigned char *tile_csr_ptr);
void step3_kernel(SMatrix *matrix, int nnz_max, int tilecnt_max);
void csr2tile_row_major(SMatrix *matrix);
void csr2tile_col_major(SMatrix *matrix);
void matrix_destroy(SMatrix *matrix);

#endif // MATRIX_OPS_H




