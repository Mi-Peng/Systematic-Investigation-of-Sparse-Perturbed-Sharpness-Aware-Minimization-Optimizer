#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cstdio>             // printf

#include <iostream>
#include <time.h>
#include <vector>
#include <cstdlib>            // std::rand


/*
    Basic Check Definition: Check CUDA & Check cusparseLt
*/
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

typedef struct{
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    
    size_t workspace_size;
    size_t compressed_size;
    float* dA_compressed;
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
} spmmDescriptor;


std::vector<spmmDescriptor> spmmDescriptorVector;

/*
    Init the number of DescriptorVectors
*/
void initSpmmDescriptorVectorNumber(int num){
    spmmDescriptorVector.resize(num);
}


/*
    Check the Support of cusparseLt;
*/
constexpr int EXIT_UNSUPPORTED = 2;
int checkCusparseLtSupport(){
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    else{
        return EXIT_SUCCESS;
    }
}


/*
    Initial the Variables of cusparseLt;
*/
int initSpmmDescriptor(
    int index, int num_batches,
    int num_A_rows,
    int num_A_cols,
    int lda,
    int num_B_rows,
    int num_B_cols,
    int ldb,
    int num_C_rows,
    int num_C_cols,
    int ldc
){
    auto order = CUSPARSE_ORDER_ROW;
    auto opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // auto opA   = CUSPARSE_OPERATION_TRANSPOSE;
    auto opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // auto type          = CUDA_R_16F;             // CUDA_R_16F / CUDA_R_32F 
    auto type  = CUDA_R_32F;
    unsigned alignment = 16;
    // unsigned alignment = 32;
    // auto compute_type  = CUSPARSE_COMPUTE_16F; // CUSPARSE_COMPUTE_16F / CUSPARSE_COMPUTE_TF32
    auto compute_type  = CUSPARSE_COMPUTE_TF32;

    int64_t batch_strideA = 0;
    int64_t batch_strideB = num_B_rows * num_B_cols;
    int64_t batch_strideC = num_C_rows * num_C_cols;
    CHECK_CUSPARSE( cusparseLtInit(&spmmDescriptorVector[index].handle) )
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &spmmDescriptorVector[index].handle, 
                                            &spmmDescriptorVector[index].matA, 
                                            num_A_rows, num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // ---------------------------------------------------
    // SET NUM BATCHES 
    CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matA,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matB,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    CHECK_CUSPARSE( cusparseLtMatDescSetAttribute(&spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matC,
                                            CUSPARSELT_MAT_NUM_BATCHES,
                                            &num_batches, sizeof(num_batches)) )
    // // -----------------------------------------------------
    // SET BATCH STRIDE (if batch_strideA = 0, the matrix multiplication performs a broadcast of the matrix A)
    CHECK_CUSPARSE(  cusparseLtMatDescSetAttribute(&spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matA,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideA,
                                                sizeof(batch_strideA)) )
    CHECK_CUSPARSE(  cusparseLtMatDescSetAttribute(&spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matB,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideB,
                                                sizeof(batch_strideB)) )
    CHECK_CUSPARSE(  cusparseLtMatDescSetAttribute(&spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matC,
                                                CUSPARSELT_MAT_BATCH_STRIDE,
                                                &batch_strideC,
                                                sizeof(batch_strideC)) )
    //--------------------------------------------------------------------------
    // MATMUL DESCRIPTOR INITIALIZATION
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matmul, opA, opB,
                                            &spmmDescriptorVector[index].matA, 
                                            &spmmDescriptorVector[index].matB, 
                                            &spmmDescriptorVector[index].matC, 
                                            &spmmDescriptorVector[index].matC,
                                            compute_type) )
    //--------------------------------------------------------------------------
    // SET BIAS POINTER(NEED TO CHECK)
    // CHECK_CUSPARSE( cusparseLtMatmulDescSetAttribute(&spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].matmul,
    //                                             CUSPARSELT_MATMUL_BIAS_POINTER,
    //                                             &dBias, sizeof(dBias)))


    //--------------------------------------------------------------------------
    // Algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].alg_sel, &spmmDescriptorVector[index].matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].plan, &spmmDescriptorVector[index].matmul, &spmmDescriptorVector[index].alg_sel,
                                             spmmDescriptorVector[index].workspace_size) )                                        
    
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &spmmDescriptorVector[index].handle, &spmmDescriptorVector[index].alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    return 0;
}

/*
    Prune the Matrix;
*/
int cusparseLtPrune(
    int index,
    float* dA,
    float* dA_prunned
){
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&spmmDescriptorVector[index].handle, 
                                         &spmmDescriptorVector[index].matmul, 
                                         dA, dA_prunned,
                                         CUSPARSELT_PRUNE_SPMMA_STRIP,   // See more on `https://docs.nvidia.com/cuda/cusparselt/types.html#cusparseltprunealg-t`
                                         spmmDescriptorVector[index].stream) )
    return 0;
}

/*
    Check the Matrix Prunned Right
*/
int checkSpmmMatrixPrunnedRight(
    int index,
    float* dA_prunned
){
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&spmmDescriptorVector[index].handle, 
                                              &spmmDescriptorVector[index].matmul, 
                                              dA_prunned, d_valid, spmmDescriptorVector[index].stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, spmmDescriptorVector[index].stream) )
    CHECK_CUDA( cudaStreamSynchronize(spmmDescriptorVector[index].stream) )
    if (is_valid != 0) {
        std::printf("[Warning] The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will Not provide correct results\n");
        return EXIT_FAILURE;
    }
    return 0;                                              

}

/*
    Compress the Prunned Right Matrix
*/
int compressPrunnedMatrix(
    int index,
    float* dA_prunned
){
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&spmmDescriptorVector[index].handle, 
                                                  &spmmDescriptorVector[index].plan,
                                                  &spmmDescriptorVector[index].compressed_size) )

    CHECK_CUDA( cudaMalloc((void**) &spmmDescriptorVector[index].dA_compressed, spmmDescriptorVector[index].compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&spmmDescriptorVector[index].handle, 
                                            &spmmDescriptorVector[index].plan, 
                                            dA_prunned,
                                            spmmDescriptorVector[index].dA_compressed, 
                                            spmmDescriptorVector[index].stream) )
    return 0;
}



int initDeviceWorkspace(int index){
    if (spmmDescriptorVector[index].d_workspace == nullptr){
        CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(
                                                 &spmmDescriptorVector[index].handle, 
                                                 &spmmDescriptorVector[index].plan,
                                                 &spmmDescriptorVector[index].workspace_size))                                                                         
        CHECK_CUDA( cudaMalloc((void**)&spmmDescriptorVector[index].d_workspace, spmmDescriptorVector[index].workspace_size) )
    }
    return 0;
}

/*
    Search the best kernel for the Sparse Matrix Multiplication
*/
int searchSpmmKernel(
    int index,
    float* dB, 
    float* dC
){  
    float alpha = 1.0f;
    float beta  = 0.0f;
    auto* dD = dC;
    CHECK_CUSPARSE( cusparseLtMatmulSearch(&spmmDescriptorVector[index].handle, 
                                           &spmmDescriptorVector[index].plan, 
                                           &alpha,
                                           spmmDescriptorVector[index].dA_compressed, 
                                           dB, 
                                           &beta,
                                           dC, dD, 
                                           spmmDescriptorVector[index].d_workspace,
                                           spmmDescriptorVector[index].streams, 
                                           spmmDescriptorVector[index].num_streams) )
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &spmmDescriptorVector[index].handle, 
                                           &spmmDescriptorVector[index].alg_sel,
                                           CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &alg_id, sizeof(alg_id)) )
    int32_t splitK, splitKBuffers;
    cusparseLtSplitKMode_t splitKMode;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &spmmDescriptorVector[index].handle, 
                                           &spmmDescriptorVector[index].alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K,
                                           &splitK, sizeof(splitK)) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &spmmDescriptorVector[index].handle, 
                                           &spmmDescriptorVector[index].alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K_MODE,
                                           &splitKMode, sizeof(splitKMode)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &spmmDescriptorVector[index].handle,
                                           &spmmDescriptorVector[index].alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
                                           &splitKBuffers,
                                           sizeof(splitKBuffers)) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(
                                            &spmmDescriptorVector[index].handle, 
                                            &spmmDescriptorVector[index].plan, 
                                            &spmmDescriptorVector[index].matmul, 
                                            &spmmDescriptorVector[index].alg_sel,
                                            spmmDescriptorVector[index].workspace_size) )
    initDeviceWorkspace(index);
    return 0;
}



/*
    Perform the Sparse Matrix Multiplication
*/
int spmmPerform(
    int index, 
    float* dB, 
    float* dC
){
    /*
        Aactual Spmm    
    */
    float alpha = 1.0f;
    float beta  = 0.0f;
    auto* dD = dC;
    CHECK_CUSPARSE( cusparseLtMatmul(&spmmDescriptorVector[index].handle, 
                                     &spmmDescriptorVector[index].plan, 
                                     &alpha, 
                                     spmmDescriptorVector[index].dA_compressed, 
                                     dB,
                                     &beta, dC, dD, 
                                     spmmDescriptorVector[index].d_workspace,
                                     spmmDescriptorVector[index].streams,
                                     spmmDescriptorVector[index].num_streams) )
}
    

int spmm_forward_implementation(
    float* dA,
    float* dB,
    float* dC,
    int m,
    int n,
    int k
)
{   
    // int chekc_pass = checkcusparseLtSupport();
    /*
        Perform the matrix multiplication: dA(m, n)*dB(n, k) => dC(m, k)
    */
    float* dA_compressed;
    float alpha = 1.0f;
    float beta  = 0.0f;
    auto* dD = dC;

    // Set the attribute of matrix
    auto     num_A_rows     = m;
    auto     num_A_cols     = n;
    auto     num_B_rows     = n;
    auto     num_B_cols     = k;
    auto     num_C_rows     = m;
    auto     num_C_cols     = k;

    unsigned alignment      = 32;
    auto     lda            = num_A_cols;
    auto     ldb            = num_B_cols;
    auto     ldc            = num_C_cols;


    // Set the basic setting of spmm;
    auto     order = CUSPARSE_ORDER_ROW;
    auto     opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto     opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto     type  = CUDA_R_32F;   // CUDA_R_16F;
    auto     compute_type = CUSPARSE_COMPUTE_TF32; // CUSPARSE_COMPUTE_16F
    

    // Device Memory Management
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )


    // Define the Variable of spmm;
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;

    // Init the Variable of spmm;
    CHECK_CUSPARSE( cusparseLtInit(&handle))
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )                                    

    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size;                                        
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )
    // Prune the A matrix and check the correctness
    /*
        Prun the A matrix (Some code) 
    */
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }

    // Compress the A matrix
    size_t compressed_size;
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
                                            dA_compressed, stream) )

    // Search the best kernel for spmm algorithm
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;

    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                        dA_compressed, dB, &beta,
                                        dC, dD, d_workspace,
                                        streams, num_streams) )
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                        &handle, &alg_sel,
                                        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                        &alg_id, sizeof(alg_id)) )

    int32_t splitK, splitKBuffers;
    cusparseLtSplitKMode_t splitKMode;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                        &handle, &alg_sel,
                                        CUSPARSELT_MATMUL_SPLIT_K,
                                        &splitK, sizeof(splitK)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                        &handle, &alg_sel,
                                        CUSPARSELT_MATMUL_SPLIT_K_MODE,
                                        &splitKMode, sizeof(splitKMode)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                        &handle, &alg_sel,
                                        CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
                                        &splitKBuffers,
                                        sizeof(splitKBuffers)) )
    // Init the Matmul Plan & WorkSpace
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                            workspace_size) )

    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                &workspace_size))
    CHECK_CUDA( cudaMalloc((void**)&d_workspace, workspace_size) )

    // Perform the matrix multiplication
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                    &beta, dC, dD, d_workspace, streams,
                                    num_streams) )

    // Check the result
    float hA[m * n];
    float hB[n * k];
    float hC[m * k];
    int A_size = m * n *sizeof(float);
    int B_size = n * k *sizeof(float);
    int C_size = m * k *sizeof(float);
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hB, dB, B_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

    bool is_rowmajor = true;
    bool A_std_layout = true;
    bool B_std_layout = true;
    // host computation
    float hC_result[m * k];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum  = 0.0f;
            for (int k1 = 0; k1 < n; k1++) {
                auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                sum      += static_cast<float>(hA[posA]) *  // [i][k]
                            static_cast<float>(hB[posB]);   // [k][j]
            }
            auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            hC_result[posC] = sum;  // [i][j]
        }
    }
     // host-device comparison
    int correct = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            auto device_value = static_cast<float>(hC[pos]);
            auto host_value   = hC_result[pos];
            if (device_value != host_value) {
                // direct floating point comparison is not reliable
                std::printf("(%d, %d):\t%f vs. %f\n",
                            i, j, host_value, device_value);
                correct = 0;
                break;
            }
        }
    }
    if (correct)
        std::printf("spmma_example test PASSED\n");
    else
        std::printf("spmma_example test FAILED: wrong result\n");



    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )

    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(d_valid) ) 
    CHECK_CUDA( cudaFree(d_workspace) )
    return EXIT_SUCCESS;
}

// int main()
// {   
//     __half hA[32 * 32];
//     __half hB[32 * 32];
//     __half hC[32 * 32] = {};
//     auto A_size = 32 * 32 * sizeof(__half);
//     auto B_size = 32 * 32 * sizeof(__half);
//     auto C_size = 32 * 32 * sizeof(__half);
//     for (int i = 0; i < 32 * 32; i++)
//         hA[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));
//     for (int i = 0; i < 32 * 32; i++)
//         hB[i] = static_cast<__half>(static_cast<float>(std::rand() % 10));

//     __half *dA, *dB, *dC;
//     CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
//     CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
//     CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
//     std::printf("Malloc Cuda\n");
//     CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )


//     int check = checkcusparseLtSupport();
//     std::printf("checkcusparseLtSupport: %d\n", check);
//     int init_return = cusparseLtDescriptorInit(
//         32, 32, 32,
//         32, 32, 32,
//         32, 32, 32
//     );
//     int prune_return = cusparseLtPrune(dA);
//     std::printf("cusparseLtPrune: %d\n", prune_return);

//     int compress_return = cusparseLtCompress(dA);
//     std::printf("cusparseLtCompress: %d\n", compress_return);

//     int serach_return = cusparseLtSearchKernel(dB, dC);
//     std::printf("cusparseLtSearchKernel: %d\n", serach_return);

//     int before_return = cusparseLtBeforeSpmm();
//     std::printf("cusparseLtBeforeSpmm: %d\n", before_return);

//     int spmm_return = cusparseLtSpmm(dB, dC);
//     std::printf("cusparseLtSpmm: %d\n", spmm_return);

//     // matrix A has been pruned
//     CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
//     CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

//     bool is_rowmajor = 1;
//     int lda = 32;
//     int ldb = 32;
//     int ldc = 32;
//     bool A_std_layout = 1;
//     bool B_std_layout = 1;
//     // host computation
//     float hC_result[32 * 32];
//     for (int i = 0; i < 32; i++) {
//         for (int j = 0; j < 32; j++) {
//             float sum  = 0.0f;
//             for (int k1 = 0; k1 < 32; k1++) {
//                 auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
//                 auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
//                 sum      += static_cast<float>(hA[posA]) *  // [i][k]
//                             static_cast<float>(hB[posB]);   // [k][j]
//             }
//             auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
//             hC_result[posC] = sum;  // [i][j]
//         }
//     }
//     // host-device comparison
//     int correct = 1;
//     for (int i = 0; i < 32; i++) {
//         for (int j = 0; j < 32; j++) {
//             auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
//             auto device_value = static_cast<float>(hC[pos]);
//             auto host_value   = hC_result[pos];
//             if (device_value != host_value) {
//                 // direct floating point comparison is not reliable
//                 std::printf("(%d, %d):\t%f vs. %f\n",
//                             i, j, host_value, device_value);
//                 correct = 0;
//                 break;
//             }
//         }
//     }
//     if (correct)
//         std::printf("spmma_example test PASSED\n");
//     else
//         std::printf("spmma_example test FAILED: wrong result\n");

//     // float* dx;
//     // float* dweight;
//     // float* dout;
//     // int x_size = 256 * 256;
//     // int weight_size = 256 * 256;
//     // int out_size = 256 * 256;
//     // CHECK_CUDA( cudaMalloc((void**) &dx, x_size))
//     // CHECK_CUDA( cudaMalloc((void**) &dweight, weight_size))
//     // CHECK_CUDA( cudaMalloc((void**) &dout, out_size))

//     // float hx[x_size];
//     // float hweight[weight_size];
//     // float hout[out_size];
//     // for(int i = 0; i < x_size; i++){
//     //     hx[i] = 1.2;
//     // }
//     // for(int i = 0; i < x_size; i++){
//     //     hweight[i] = 2.1;
//     // }
//     // for(int i = 0; i < x_size; i++){
//     //     hout[i] = 0.0;
//     // }
//     // CHECK_CUDA( cudaMemcpy(dx, hx, x_size, cudaMemcpyHostToDevice))
//     // CHECK_CUDA( cudaMemcpy(dweight, hweight, weight_size, cudaMemcpyHostToDevice))
//     // CHECK_CUDA( cudaMemcpy(dout, hout, out_size, cudaMemcpyHostToDevice))

//     // spmm_forward_implementation(
//     //     dx,
//     //     dweight,
//     //     dout,
//     //     256, 
//     //     256,
//     //     256
//     // );
//     // return 0;
// }
