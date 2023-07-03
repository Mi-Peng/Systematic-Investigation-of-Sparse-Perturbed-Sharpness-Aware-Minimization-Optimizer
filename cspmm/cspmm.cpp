#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <torch/extension.h>

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK(x.type().is_cuda(), x, " must be a CUDAtensor ")
#define CHECK_TENSOR_CONTIGUOUS(x) \
   TORCH_CHECK(x.is_contiguous(), x, " must be contiguous ")
#define CHECK_INPUT(x) \
   CHECK_CUDA_TENSOR(x);       \
   CHECK_TENSOR_CONTIGUOUS(x)

// --------------------------------------------------------------- // 
// Function Declaration Starting:
int initSpmmDescriptorVectorNumber(int num);
int checkCusparseLtSupport();
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
);

int cusparseLtPrune(int index, float* dA, float* dA_prunned);
int checkSpmmMatrixPrunnedRight(int index, float* dA_prunned);
int compressPrunnedMatrix(int index, float* dA_prunned);
int searchSpmmKernel(int index, float*dB, float* dC);
int spmmPerform(int index, float*dB, float* dC);



int spmm_forward_implementation(
    float* dA,
    float* dB,
    float* dC,
    int mt,
    int nt,
    int kt
);
// Function Declaration Done
// --------------------------------------------------------------- //

int spmm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out
){
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(out);
    
    if ((x.scalar_type() == torch::kFloat32) &&
        (weight.scalar_type() == torch::kFloat32) &&
        (out.scalar_type() == torch::kFloat32))
        {
            return spmm_forward_implementation(
                x.data_ptr<float>(),
                weight.data_ptr<float>(),
                out.data_ptr<float>(),
                x.size(0),
                x.size(1),
                weight.size(1)
            );
        }
    else {
        std::cout << "type of input is not 32-b float" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}


int pruneMatrix(
    int index,
    torch::Tensor dA,
    torch::Tensor dA_prunned
){
    CHECK_INPUT(dA);
    CHECK_INPUT(dA_prunned);
    if (dA.scalar_type() == torch::kFloat32 &&
        dA_prunned.scalar_type() == torch::kFloat32){
        return cusparseLtPrune(index, dA.data_ptr<float>(), dA_prunned.data_ptr<float>());
    }
    else {
        std::cout << "type of input is not 32-b float" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}

int checkPrunned(
    int index,
    torch::Tensor dA_prunned
){
    CHECK_INPUT(dA_prunned);
    if (dA_prunned.scalar_type() == torch::kFloat32){
        return checkSpmmMatrixPrunnedRight(index, dA_prunned.data_ptr<float>());
    }
    else {
        std::cout << "type of input is not 32-b float" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}

int compressMatrix(
    int index,
    torch::Tensor dA_prunned
){
    CHECK_INPUT(dA_prunned);
    if (dA_prunned.scalar_type() == torch::kFloat32){
        return compressPrunnedMatrix(index, dA_prunned.data_ptr<float>());
    }
    else {
        std::cout << "type of input is not 32-b float" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}


int searchKernel(
    int index,
    torch::Tensor dB,
    torch::Tensor dC
){
    CHECK_INPUT(dB);
    CHECK_INPUT(dC);
    if (dB.scalar_type() == torch::kFloat32 &&
        dC.scalar_type() == torch::kFloat32){
            return searchSpmmKernel(index, dB.data_ptr<float>(), dC.data_ptr<float>());
        }
    else {
        std::cout << "type of input is not 32-b float" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}

int spmm(
    int index,
    torch::Tensor dB,
    torch::Tensor dC
){
    CHECK_INPUT(dB);
    CHECK_INPUT(dC);
    if (dB.scalar_type() == torch::kFloat32 &&
        dC.scalar_type() == torch::kFloat32){
            return spmmPerform(index, dB.data_ptr<float>(), dC.data_ptr<float>());
        }
    else {
        std::cout << "type of input is not 32-b float" << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("initSpmmNum", &initSpmmDescriptorVectorNumber, "Init the number of vector.");
    m.def("checkCusparseLt", &checkCusparseLtSupport, "Check the hardware for perparation");
    m.def("initSpmmDescriptor", &initSpmmDescriptor, "Init the handle, and so on.");
    m.def("pruneMatrix", &pruneMatrix, "Prune the matrix.");
    m.def("checkPrunned", &checkPrunned, "Check whether the prunned matrix is prunned in a right way or not");
    m.def("compressMatrix", &compressMatrix, "Compress the prunned matrix.");
    m.def("searchKernel", &searchKernel, "Search the best kernel");
    m.def("spmm", &spmm, "Sparse Matrix Multiplication");


    m.def("forward", &spmm_forward, "Sparse Matrix Multiplication Forward Porcess.");
    
}