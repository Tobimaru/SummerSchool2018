#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "util.hpp"

__global__
void reverseString(char* string, const int n)
{
    extern __shared__ char buffer[];
    
    const auto i = threadIdx.x;
 
    buffer[i] = string[i];

    __syncthreads();

    string[i] = buffer[n - i - 1]; 
}

int main(int argc, char** argv) {
    // check that the user has passed a string to reverse
    if(argc<2) {
        std::cout << "useage : ./string_reverse \"string to reverse\"\n" << std::endl;
        exit(0);
    }

    // determine the length of the string, and copy in to buffer
    auto n = strlen(argv[1]);
    auto string = malloc_managed<char>(n+1);
    std::copy(argv[1], argv[1]+n, string);
    string[n] = 0; // add null terminator

    std::cout << "string to reverse:\n" << string << "\n";

    reverseString<<<1,n,n>>>(string, n);

    // print reversed string
    cudaDeviceSynchronize();
    std::cout << "reversed string:\n" << string << "\n";

    // free memory
    cudaFree(string);

    return 0;
}

