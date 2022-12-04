#include <thundersvm/kernel/kernelmatrix_kernel.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<chrono>
#include<omp.h>
#include <CL/sycl.hpp>
#include "oneapi/mkl/blas.hpp" 
#include<mkl.h>
#include<thread>
#include"tranpose_sse.h"
using namespace std::chrono;
namespace mkl = oneapi::mkl;  //# shorten mkl namespace
#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))
namespace svm_kernel {
    //load balancing
    const long long GPUMAXM=12884901888;//10512000000
    long long GPUM;
    long long CPUM;
    unsigned long gpu_m;
    unsigned long cpu_m;
    const float ComputingPowerRatio=5;
    //gpu内划分
    long devide = 4;
    long m_remainder;
    long quotient;
    bool isInited=false;
    //sycl
    mkl::transpose transA = mkl::transpose::nontrans;
    mkl::transpose transB = mkl::transpose::nontrans;
    auto async_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
        }
    };
    sycl::queue queue;
    sycl::queue queue2;
    sycl::event gemm_done;
    std::vector<sycl::event> gemm_dependencies;
    kernel_type **A_usm;
    kernel_type *B_usm;
    kernel_type *B_usm2;
    kernel_type **C_usm;
    kernel_type *A_remainder_USM;
    kernel_type *C_remainder_USM;
    kernel_type *tem_result;
    kernel_type *tem_result2;

    std::thread gpu_thread;
    std::thread cpu_thread;

    void get_working_set_ins(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                            const SyncArray<int> &data_row_idx, SyncArray<kernel_type> &data_rows, int m, int n) {
        const int *data_row_idx_data = data_row_idx.host_data();
        kernel_type *data_rows_data = data_rows.host_data();
        const int *row_ptr_data = row_ptr.host_data();
        const int *col_ind_data = col_ind.host_data();
        const kernel_type *val_data = val.host_data();
#pragma omp parallel for schedule(guided)
        for (unsigned long i = 0; i < m; i++) {
            unsigned long row = data_row_idx_data[i];
            for (unsigned long j = row_ptr_data[row]; j < row_ptr_data[row + 1]; ++j) {
                unsigned long col = col_ind_data[j];
                data_rows_data[i * n + col] = val_data[j]; //row major
            }
        }
    }

    void get_working_set_ins_col(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                            const SyncArray<int> &data_row_idx, SyncArray<kernel_type> &data_rows, int m, int n) {
        const int *data_row_idx_data = data_row_idx.host_data();
        kernel_type *data_rows_data = data_rows.host_data();
        const int *row_ptr_data = row_ptr.host_data();
        const int *col_ind_data = col_ind.host_data();
        const kernel_type *val_data = val.host_data();
#pragma omp parallel for schedule(guided)
        for (unsigned long i = 0; i < m; i++) {
            unsigned long row = data_row_idx_data[i];
            for (unsigned long j = row_ptr_data[row]; j < row_ptr_data[row + 1]; ++j) {
                unsigned long col = col_ind_data[j];
                data_rows_data[col*m+i] = val_data[j]; //col major
            }
        }
    }

    void get_working_set_ins(const SyncArray<kernel_type> &val, const SyncArray<int> &col_ind, const SyncArray<int> &row_ptr,
                        const SyncArray<int> &data_row_idx, SyncArray <kernel_type>& ws_val,
                        SyncArray<int> &ws_col_ind, SyncArray<int> &ws_row_ptr, int m){
        const int *data_row_idx_data = data_row_idx.host_data();
        const int *row_ptr_data = row_ptr.host_data();
        const int *col_ind_data = col_ind.host_data();
        const kernel_type *val_data = val.host_data();

        //three arrays for csr representation
        vector<kernel_type> csr_val;
        vector<int> csr_col_ind;//index of each value of all the instances
        vector<int> csr_row_ptr(1, 0);//the start positions of the instances
        //ws_row_ptr_data[0] = 0;
        for(int i = 0; i < m; i++){
            int row = data_row_idx_data[i];
            for(int j = row_ptr_data[row]; j < row_ptr_data[row+1]; ++j){
                csr_col_ind.push_back(col_ind_data[j]);
                csr_val.push_back(val_data[j]);
            }
            csr_row_ptr.push_back(csr_row_ptr.back() + row_ptr_data[row+1] - row_ptr_data[row]);
        }
        //three arrays (on GPU/CPU) for csr representation
        ws_val.resize(csr_val.size());
        ws_col_ind.resize(csr_col_ind.size());
        ws_row_ptr.resize(csr_row_ptr.size());
        //copy data to the three arrays
        ws_val.copy_from(csr_val.data(), ws_val.size());
        ws_col_ind.copy_from(csr_col_ind.data(), ws_col_ind.size());
        ws_row_ptr.copy_from(csr_row_ptr.data(), ws_row_ptr.size());
    }

    void RBF_kernel(const SyncArray<kernel_type> &self_dot0, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma) {
        kernel_type *dot_product_data = dot_product.host_data();
        const kernel_type *self_dot0_data = self_dot0.host_data();
        const kernel_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                dot_product_data[i * n + j] = expf(
                        -(self_dot0_data[i] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) * gamma);
            }
        }
    }

    void RBF_kernel(const SyncArray<int> &self_dot0_idx, const SyncArray<kernel_type> &self_dot1,
               SyncArray<kernel_type> &dot_product, int m,
               int n, kernel_type gamma) {
        kernel_type *dot_product_data = dot_product.host_data();
        const int *self_dot0_idx_data = self_dot0_idx.host_data();
        const kernel_type *self_dot1_data = self_dot1.host_data();
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                dot_product_data[i * n + j] = expf(
                        -(self_dot1_data[self_dot0_idx_data[i]] + self_dot1_data[j] - dot_product_data[i * n + j] * 2) *
                        gamma);
            }
        }
    }

    void poly_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int degree, int mn) {
        kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < mn; idx++) {
            dot_product_data[idx] = powf(gamma * dot_product_data[idx] + coef0, degree);
        }
    }

    void sigmoid_kernel(SyncArray<kernel_type> &dot_product, kernel_type gamma, kernel_type coef0, int mn) {
        kernel_type *dot_product_data = dot_product.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < mn; idx++) {
            dot_product_data[idx] = tanhf(gamma * dot_product_data[idx] + coef0);
        }
    }

    void sum_kernel_values(const SyncArray<float_type> &coef, int total_sv, const SyncArray<int> &sv_start,
                           const SyncArray<int> &sv_count, const SyncArray<float_type> &rho,
                           const SyncArray<kernel_type> &k_mat,
                           SyncArray<float_type> &dec_values, int n_classes, int n_instances) {
        const int *sv_start_data = sv_start.host_data();
        const int *sv_count_data = sv_count.host_data();
        const float_type *coef_data = coef.host_data();
        const kernel_type *k_mat_data = k_mat.host_data();
        float_type *dec_values_data = dec_values.host_data();
        const float_type* rho_data = rho.host_data();
#pragma omp parallel for schedule(guided)
        for (int idx = 0; idx < n_instances; idx++) {
            int k = 0;
            int n_binary_models = n_classes * (n_classes - 1) / 2;
            for (int i = 0; i < n_classes; ++i) {
                for (int j = i + 1; j < n_classes; ++j) {
                    int si = sv_start_data[i];
                    int sj = sv_start_data[j];
                    int ci = sv_count_data[i];
                    int cj = sv_count_data[j];
                    const float_type *coef1 = &coef_data[(j - 1) * total_sv];
                    const float_type *coef2 = &coef_data[i * total_sv];
                    const kernel_type *k_values = &k_mat_data[idx * total_sv];
                    double sum = 0;
#pragma omp parallel for reduction(+:sum)
                    for (int l = 0; l < ci; ++l) {
                        sum += coef1[si + l] * k_values[si + l];
                    }
#pragma omp parallel for reduction(+:sum)
                    for (int l = 0; l < cj; ++l) {
                        sum += coef2[sj + l] * k_values[sj + l];
                    }
                    dec_values_data[idx * n_binary_models + k] = sum - rho_data[k];
                    k++;
                }
            }
        }
    }
//注释了eigen的，使用dpcpp
    void dns_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz,
                     SyncArray<kernel_type> &result) {
        Eigen::Map<const Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> denseMat(dense_mat.host_data(), k, n);
        Eigen::Map<const Eigen::SparseMatrix<kernel_type, Eigen::RowMajor>> sparseMat(m, k, nnz, csr_row_ptr.host_data(),
                                                                                csr_col_ind.host_data(),
                                                                                csr_val.host_data());
        Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = sparseMat * denseMat;
        Eigen::Map<Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
                                                                                           retMat.rows(),
                                                                                           retMat.cols()) = retMat;

    }

    void csr_csr_mul(int m, int n, int k, const SyncArray<kernel_type> &ws_val, const SyncArray<int> &ws_col_ind,
                     const SyncArray<int> &ws_row_ptr, const SyncArray<kernel_type> &csr_val,
                     const SyncArray<int> &csr_row_ptr, const SyncArray<int> &csr_col_ind, int nnz, int nnz2,
                     SyncArray<kernel_type> &result) {
        Eigen::Map<const Eigen::SparseMatrix<kernel_type, Eigen::RowMajor>> sparseMat1(m, k, nnz, csr_row_ptr.host_data(),
                                                                                      csr_col_ind.host_data(),
                                                                                      csr_val.host_data());
        Eigen::Map<const Eigen::SparseMatrix<kernel_type>> sparseMat2(k, n, nnz2, ws_row_ptr.host_data(),
                                                                      ws_col_ind.host_data(),
                                                                      ws_val.host_data());
        Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> retMat = sparseMat1 * sparseMat2;
        Eigen::Map<Eigen::Matrix<kernel_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> >(result.host_data(),
                                                                                                 retMat.rows(),
                                                                                                 retMat.cols()) = retMat;
    }

    void gemm_on_gpu(long m,long n,long k){
        // std::cout<<"thread start\n";
        for (size_t i = devide/2; i < devide; i++)
        {
            mkl::blas::row_major::gemm(queue2, mkl::transpose::nontrans,mkl::transpose::nontrans, quotient,(long) n,(long) k, 1.0, A_usm[i],k, B_usm2,n, 0.0, C_usm[i],n);
        }
        queue2.wait();
        // std::cout<<"thread compute over\n";
        for (size_t i = devide/2; i < devide; i++){
             queue2.memcpy(&tem_result[i*quotient*n],C_usm[i],(long)sizeof(kernel_type)*quotient*n);
        }
        queue2.wait();
    }

    void gemm_on_cpu(long m, long n, long k,const kernel_type *A, const kernel_type *B,kernel_type *C){
        cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, m, n, k, 1.0,A, k, B, n, 0.0, C, n);
    }


    void dns_dns_mul(int m, int n, int k, const SyncArray<kernel_type> &dense_mat,
                     const SyncArray<kernel_type> &origin_dense, SyncArray<kernel_type> &result){                                
        // printf("m:%d  n:%d  k:%d\n",m,n,k);                        
        unsigned long ldA = k, ldB = n, ldC = n;
        if(!isInited){
            gpu_m=(GPUMAXM-(long)k*n*sizeof(kernel_type))/(((long)k+n)*sizeof(kernel_type));
            if(gpu_m>m){
                gpu_m=m;
                cpu_m=0;
            }else{
                cpu_m=m-gpu_m;
            }
            //remainder init
            m_remainder = gpu_m % devide;
            quotient = gpu_m / devide;
            //queue init
            sycl::device de[7];
            int dp=0;
            auto platforms = sycl::platform::get_platforms();
            for(auto &platform : platforms) {
                auto devices = platform.get_devices();
                for (auto &device : devices) de[dp++]=device;
            }        
            queue=sycl::queue(de[2], async_handler);
            queue2=sycl::queue(de[3], async_handler);
            //USM init
            A_usm=new kernel_type*[devide];
            C_usm=new kernel_type*[devide];
            B_usm =  sycl::aligned_alloc_device<kernel_type>(sizeof(kernel_type), long(n)*k, queue);
            B_usm2 =  sycl::aligned_alloc_device<kernel_type>(sizeof(kernel_type), long(n)*k, queue2);
            for (size_t i = 0; i < devide/2; i++){
                A_usm[i] = sycl::aligned_alloc_device<kernel_type>(4, long(quotient)*k, queue);
                C_usm[i] = sycl::aligned_alloc_device<kernel_type>(4, long(quotient)*n, queue);
            }
             for (size_t i = devide/2; i < devide; i++){
                A_usm[i] = sycl::aligned_alloc_device<kernel_type>(4, long(quotient)*k, queue2);
                C_usm[i] = sycl::aligned_alloc_device<kernel_type>(4, long(quotient)*n, queue2);
            }
            A_remainder_USM=sycl::malloc_device<kernel_type>(m_remainder*k, queue);
            C_remainder_USM=sycl::malloc_device<kernel_type>(m_remainder*k, queue);
            for (size_t i = 0; i < devide/2; i++){
               queue.memcpy(A_usm[i], &origin_dense.host_data()[i*quotient*k], (long)sizeof(kernel_type)*quotient*k);
            }
            for (size_t i = devide/2; i < devide; i++){
               queue2.memcpy(A_usm[i], &origin_dense.host_data()[i*quotient*k], (long)sizeof(kernel_type)*quotient*k);
            }
            queue.memcpy(A_remainder_USM, &origin_dense.host_data()[devide*quotient*k], sizeof(kernel_type)*m_remainder*k);
            queue.wait();
            //transelate init
            tem_result=new kernel_type[(long)m*n];
            tem_result2=new kernel_type[(long)m*n];
            isInited=true;
        }

        queue.memcpy(B_usm,dense_mat.host_data(),(long)sizeof(kernel_type)*n*k);
        queue2.memcpy(B_usm2,dense_mat.host_data(),(long)sizeof(kernel_type)*n*k);
        queue.wait();

        gpu_thread = std::thread(gemm_on_gpu,m,n,k);
        if(cpu_m!=0){
            cpu_thread = std::thread(gemm_on_cpu,cpu_m,n,k,&origin_dense.host_data()[gpu_m*k],dense_mat.host_data(),&tem_result[gpu_m*n]);
        }

        float alpha=1.0,beta=0.0; 
        for (size_t i = 0; i < devide/2; i++){
            gemm_done = mkl::blas::row_major::gemm(queue, mkl::transpose::nontrans,mkl::transpose::nontrans, quotient,(long) n,(long) k, alpha, A_usm[i],ldA, B_usm,ldB, beta, C_usm[i],ldC,gemm_dependencies);
        }
        //这里计算余数部分
        mkl::blas::row_major::gemm(queue, mkl::transpose::nontrans,mkl::transpose::nontrans, m_remainder,(long) n,(long) k, alpha, A_remainder_USM,ldA, B_usm,ldB, beta, C_remainder_USM,ldC,gemm_dependencies);
        //CPU计算
      
        queue.wait();

        for (size_t i = 0; i < devide/2; i++){
             queue.memcpy(&tem_result[i*quotient*n],C_usm[i],(long)sizeof(kernel_type)*quotient*n);
        }
        queue.memcpy(&tem_result[devide*quotient*n],C_remainder_USM,(long)sizeof(kernel_type)*m_remainder*n);
        queue.wait();
        gpu_thread.join();
        if (cpu_m!=0){
            // cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, cpu_m, n, k, 1.0,&origin_dense.host_data()[gpu_m*k], k, dense_mat.host_data(), n, 0.0, &tem_result[gpu_m*n], n);
            cpu_thread.join();
        }  
        
        //transpose
        // #pragma omp parallel for schedule(guided)   
        // for (long i = 0; i <m; i++) //m
        //     for (long j = 0; j < n; j++)    //k
        //         result.host_data()[j*m+i]=tem_result[i*n+j];
        transpose_block_SSE4x4(tem_result,result.host_data(),m,n,ROUND_UP(n,16),ROUND_UP(m,16),32);
        

    }
};

