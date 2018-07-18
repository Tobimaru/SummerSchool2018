#include <iostream>

#include <cuda.h>

#include "util.hpp"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

// TODO implement dot product kernel
template <int THREADS>
__global__
void dot_gpu_kernel(const double *x, const double *y, double *result, const int n) {

    __shared__ double sums[THREADS];

    const auto i = threadIdx.x;
    
    sums[i] = x[i] * y[i];

    __syncthreads();

    int half_n = n/2;
    while (half_n > 0){

        if (i < half_n){
            sums[i] += sums[half_n + i];
        }
        
        __syncthreads();
      
       half_n /= 2;
    }  

    if (i == 0){
        *result = sums[0];
    }

   /*if (i == 0){
        double sum = 0;
        
        for (auto j = 0; j < n; ++j){
            sum += sums[j];
        }
        
        *result = sum; 
    }*/
}

double dot_gpu(const double *x, const double* y, int n) {
    static double* result = malloc_managed<double>(1);
  
    const int maxThreads = 1024;

    for (int i = 0; i < n; i += maxThreads)
    {
        double sum = 0.0;
        dot_gpu_kernel<maxThreads><<<1, maxThreads, maxThreads>>>
        (x+i, y+i, &sum, maxThreads);
        *result += sum;
    }

    cudaDeviceSynchronize();
    return *result;
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product CUDA of length n = " << n
              << " : " << size_in_bytes*1e-9 << "MB\n";

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto x_d = malloc_device<double>(n);
    auto y_d = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_h, x_d, n);
    copy_to_device<double>(y_h, y_d, n);

    auto result   = dot_gpu(x_d, y_d, n);
    auto expected = dot_host(x_h, y_h, n);
    printf("expected %f got %f\n", (float)expected, (float)result);

    return 0;
}

