#ifndef _SPGEMM_PARALLELREF_NEW_
#define _SPGEMM_PARALLELREF_NEW_

#include <stdbool.h>
#include "common.h"
#include "utils.hpp"

// Declaration of the spgemm_spa function
void spgemm_spa(        const int           *d_csrRowPtrA,
                        const int           *d_csrColIdxA,
                        const MAT_VAL_TYPE    *d_csrValA,
                        const int            mA,
                        const int            nA,
                        const int            nnzA,
                        const int           *d_csrRowPtrB,
                        const int           *d_csrColIdxB,
                        const MAT_VAL_TYPE    *d_csrValB,
                        const int            mB,
                        const int            nB,
                        const int            nnzB,
                            int           *d_csrRowPtrC,
                            int           *d_csrColIdxC,
                           MAT_VAL_TYPE    *d_csrValC,
                        const int            mC,
                        const int            nC,
                            int           *nnzC,
                        const int           get_nnzC_only);

#endif
