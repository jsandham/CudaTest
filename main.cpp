#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparseSpMV
#include <stdio.h>         // printf
#include <stdlib.h>        // EXIT_FAILURE
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

//int main()
//{
//    cusparseHandle_t     handle = 0;
//    cusparseSpMatDescr_t matA;
//    cusparseDnMatDescr_t matB, matC;
//
//    return 0;
//}


template<typename J, typename T>
void printMatrix(std::vector<T> matrix, J m, J n)
{
    std::cout << "matrix" << std::endl;
    for(J i = 0; i < m; i++){
	for(J j = 0; j < n; j++){
	    std::cout << matrix[j*m + i] << " ";
	}
	std::cout << "" << std::endl;
    }
}

template<typename T>
void printVector(std::vector<T> vec)
{
    for(size_t i = 0; i < vec.size(); i++){
        std::cout << vec[i] << " ";
    }
    std::cout << "" << std::endl;
}

template<typename J, typename T>
void createDenseMatrix(std::vector<T>& A, J m, J n)
{
    A.resize(m * n);

    srand(0);
    for(J i = 0; i < m; i++){
        for(J j = 0; j < n; j++){
            if(rand() % 100 > 20){
                A[j*m + i] = rand() % 10 - 5;
            }
            else{
                A[j*m + i] = 0.0f;
            }
        }
    }

    if(m <= 32 && n <= 32){
        printMatrix(A, m, n);
    }
}

template<typename I, typename J, typename T>
void createCSRMatrix(std::vector<I>& rowPtr, std::vector<J>& cols, std::vector<T>& vals, J m, J n)
{
    srand(123456);

    std::vector<float> A(m * n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(rand() % 100 < 30){
                A[j * m + i] = rand() % 10;// - 5;
            }
            else{
                A[j * m + i] = 0.0f;
            }
        }
    }

    if(m <= 32 && n <= 32){
        printMatrix(A, m, n);
    }

    rowPtr.resize(m + 1, 0);

    for(int i = 0; i < m; i++){
        int row_nnz = 0;
        for(int j = 0; j < n; j++){
            if(A[j * m + i] != 0){
                cols.push_back(j);
                vals.push_back(A[j * m + i]);
                row_nnz++;
            }
        }

        rowPtr[i + 1] = rowPtr[i] + row_nnz;
    }

    std::cout << "CSR row ptr" << std::endl;
    for(int i = 0; i < rowPtr.size(); i++)
    {
        std::cout << rowPtr[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "CSR col ind" << std::endl;
    for(int i = 0; i < cols.size(); i++)
    {
        std::cout << cols[i] << " ";
    }
    std::cout << "" << std::endl;

    std::cout << "CSR val" << std::endl;
    for(int i = 0; i < vals.size(); i++)
    {
        std::cout << vals[i] << " ";
    }
    std::cout << "" << std::endl;
}

int main()
{
    int32_t m = 5;
    int32_t n = 3;
    int32_t k = 2;
    cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transB = CUSPARSE_OPERATION_TRANSPOSE;//rocsparse_operation_transpose;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseSpMMAlg_t alg = CUSPARSE_SPMM_CSR_ALG1;

    cusparseIndexType_t itype = CUSPARSE_INDEX_64I;
    cusparseIndexType_t jtype = CUSPARSE_INDEX_32I;
    cudaDataType_t ttype = CUDA_R_32F;

    // J ldb
    //     = (trans_B == rocsparse_operation_none) ? (trans_A == rocsparse_operation_none ? K : M) : N;
    // J ldc = (trans_A == rocsparse_operation_none) ? M : K;

    // J ncol_B = (trans_B == rocsparse_operation_none ? N : K);

    cusparseHandle_t handle;
    cusparseCreate(&handle);


    float halpha = 1.0f;
    float hbeta = 0.0f;

    // Host arrays
    std::vector<int64_t> hcsr_row_ptr;
    std::vector<int32_t> hcsr_col_ind;
    std::vector<float> hcsr_val;

    std::vector<float> hB;
    std::vector<float> hC;

    createCSRMatrix<int64_t, int32_t, float>(hcsr_row_ptr, hcsr_col_ind, hcsr_val, m, k);
    createDenseMatrix<int32_t, float>(hB, n, k);
    //createDenseMatrix<int32_t, float>(hB, k, n);
    createDenseMatrix<int32_t, float>(hC, m, n);

    int64_t nnzA = hcsr_val.size();
    int64_t nnzB = hB.size();
    int64_t nnzC = hC.size();

    // Device arrays
    int64_t* dcsr_row_ptr;
    int32_t* dcsr_col_ind;
    float* dcsr_val;
    float* dB;
    float* dC;
    CHECK_CUDA(cudaMalloc((void**)&dcsr_row_ptr, (m + 1) * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc((void**)&dcsr_col_ind, nnzA * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc((void**)&dcsr_val, nnzA * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dB, nnzC * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dC, nnzC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), (m + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dcsr_col_ind, hcsr_col_ind.data(), nnzA * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dcsr_val, hcsr_val.data(), nnzA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), nnzB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), nnzC * sizeof(float), cudaMemcpyHostToDevice));

    // Create descriptors
    cusparseSpMatDescr_t A;
    cusparseDnMatDescr_t B;
    cusparseDnMatDescr_t C;
    CHECK_CUSPARSE(cusparseCreateCsr(&A,
                                m,
                                k,
                                nnzA,
                                dcsr_row_ptr,
                                dcsr_col_ind,
                                dcsr_val,
                                itype,
                                jtype,
                                base,
                                ttype));
    CHECK_CUSPARSE(cusparseCreateDnMat(&B, n, k, n, dB, ttype, order));
    //cusparseCreateDnMat(&B, k, n, k, dB, ttype, order);
    CHECK_CUSPARSE(cusparseCreateDnMat(&C, m, n, m, dC, ttype, order));

    // Query SpMM buffer
    size_t buffer_size;
    cusparseStatus_t status = cusparseSpMM_bufferSize(
        handle, transA, transB, &halpha, A, B, &hbeta, C, ttype, alg, &buffer_size);

    std::cout << "buffer size: " << buffer_size << " status: " << status << std::endl;

    // Allocate buffer
    void* dbuffer;
    CHECK_CUDA(cudaMalloc((void**)&dbuffer, buffer_size));

    // Pointer mode host
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    status = cusparseSpMM(handle,
                    transA,
                    transB,
                    &halpha,
                    A,
                    B,
                    &hbeta,
                    C,
                    ttype,
                    alg,
                    dbuffer);

    std::cout << "status: " << status << std::endl;

    std::vector<float> hC_GPU(nnzC);
    cudaMemcpy(hC_GPU.data(), dC, nnzC * sizeof(float), cudaMemcpyDeviceToHost);


    printMatrix(hC_GPU, m, n);

    return 0;
}

