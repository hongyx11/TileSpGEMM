# TileSpGEMM

 

**TileSpGEMM** is an open source code of paper:

Yuyao Niu, Zhengyang Lu, Haonan Ji, Shuhui Song, Zhou Jin, and Weifeng Liu. 2022. TileSpGEMM: A Tiled Algorithm for Parallel Sparse General Matrix-Matrix Multiplication on GPUs. In 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Program- ming (PPoPP ’22), 17 pages. https://doi.org/10.1145/3503221.3508431

-------------------


## Introduction

General sparse matrix-matrix multiplication(SpGEMM) executes AB=C, where A, B and C are all sparse matrices. TileSpGEMM sparsifies the tiled method in dense general matrix-matrix multiplication (GEMM) and saves each non-empty tile in a sparse form. By this way, the three performance issues of load imbalance, allocating proper size for intermediate products and designing a sparse accumulator can be resolved. Several optimization techniques, such as binary search for set intersection, bit mask operations for symbolic SpGEMM, and an adaptive method for selecting sparse or dense accumulator in on-chip memory, are also developed to improve efficiency. TileSpGEMM provides a version of CUDA on a high parallelism currently. 


<!-- ## Structure
```
beidoublas/README     instructions on installation
beidoublas/src        C source code, to be compiled into libbeidoublas.so
beidoublas/test       testing code
beidoublas/Makefile   top-level Makefile that does installation and testing
``` -->

## Installation

<!-- To use this code, you need to modify the Makefile with correct g++ installation path and use make for automatic installation. -->
To better reproduce experiment results, we suggest an NVIDIA GPU with compute capability 8.6.
TileSpGEMM evaluation requires the CUDA GPU driver, nvcc CUDA compiler, and the cuSPARSE library, all of them are included with the CUDA Toolkit. The artifacts have been tested on Ubuntu 18.04/20.04, and are expected to run correctly under other Linux distributions.

## Execution of TileSpGEMM
Our test programs currently support input files encoded using the matrix market format. All matrix market datasets used in this evaluation are publicly available from the SuiteSparse Matrix Collection. 

1. Set CUDA path in the Makefile

2. The command 'make' generates an executable file 'test' for double precision.
> **make**

3. Run SpGEMM code on matrix data with auto-tuning in double precision. The GPU compilation takes an optional d=<gpu-device, e.g., 0> parameter that specifies the GPU device to run if multiple GPU devices are available at the same time, and another optional aat=<transpose, e.g., 0> parameter that means computing C = A^2 (-aat 0) or C = AA^T (-aat 1)). 
> **$ ./test -d 0 -aat 0 <path/to/dataset/mtx>**

## Output Information

Lines 1-2 outputs the input matrix's information including the path of matrix file, The number of rows, columns and nonzeros.

Line 3 prints the file loading time (in seconds).

Line 4 prints the size of tile used in our TileSpGEMM algorithm.

Line 5 prints the number of floating point operations during the multiplication.

Line 6 prints the runtime of transforming the input matrix from the CSR format to our tiled data structure (in millisec- onds) (Figure 12 in our paper).

Line 7 prints TileSpGEMM data structure's space consump- tion (in million bytes) (Figure 11 in our paper).

Lines 8-14 print execution time (in milliseconds) of the three algorithm steps and all memory allocation on CPU and GPU (Figure 10 in our paper).

Line 15 prints the number of tiles of the resulting matrix C. Line 16 prints the number of nonzeros of the resulting matrix C.

Line 17 prints TileSpGEMM runtime (in milliseconds) and performance (in GFlOPs) (Figures 6 and 7 in our paper).

Line 18 prints the checking result after comparing our output with the one generated by cuSPARSE.

## Release version
Jan 3,2022 Version Alpha

 




