#ifndef _SPGEMM_CPU_H_
#define _SPGEMM_CPU_H_

#include "common.h"

void step1(int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA,
           int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB,
           int *blkrowptrC, int *numblkC);

void step2(int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA,
           int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB,
           int *blkrowptrC, int *blkcolidxC);

void step3(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA, int blknA, int *nnzb_A, int mA,
           MAT_VAL_TYPE *blkcsr_Val_A, unsigned char *blkcsr_Col_A, unsigned char *blkcsr_Ptr_A,
           int *d_blkcolptrB, int *d_blkrowidxB, int blkmB, int blknB, int *nnzb_B, int nB,
           MAT_VAL_TYPE *blkcsr_Val_B, unsigned char *blkcsr_Col_B, unsigned char *blkcsr_Ptr_B,
           int *d_blkrowptrC, int *d_blkcolidxC, int *nnzb_C, int *nnzC);

void step4(int *blkrowptrA, int *blkcolidxA, int blkmA, int blknA, int *nnzb_A, int mA,
           MAT_VAL_TYPE *blkcsr_Val_A, unsigned char *blkcsr_Col_A, unsigned char *blkcsr_Ptr_A,
           int *blkcolptrB, int *blkrowidxB, int blkmB, int blknB, int *nnzb_B,
           MAT_VAL_TYPE *blkcsr_Val_B, unsigned char *blkcsr_Col_B, unsigned char *blkcsr_Ptr_B,
           int *blkrowptrC, int *blkcolidxC, int *nnzb_C,
           MAT_VAL_TYPE *blkcsr_Val_C, unsigned char *blkcsr_Col_C, unsigned char *blkcsr_Ptr_C);

void spgemm_cpu(SMatrix *A, SMatrix *B, SMatrix *C);

#endif



