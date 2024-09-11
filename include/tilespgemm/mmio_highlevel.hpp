#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

#include "tilespgemm/common.h"

// Function declarations
int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, char *filename);
int mmio_data(int *csrRowPtr, int *csrColIdx, MAT_VAL_TYPE *csrVal, char *filename);
int mmio_allinone(int *m, int *n, MAT_PTR_TYPE *nnz, int *isSymmetric, 
                  MAT_PTR_TYPE **csrRowPtr, int **csrColIdx, MAT_VAL_TYPE **csrVal, 
                  char *filename);

#endif


