#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cufft.h"
#include <cmath>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/complex.h>
#include <chrono>

using namespace std;
using namespace chrono;

#define PI 3.1415926535897932384626433832

typedef std::chrono::high_resolution_clock Clock;


//struct cuComplex
//{
//    double real, imag;
//    __device__ cuComplex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}
//    __device__ cuComplex operator+(const cuComplex& a)
//    {
//        return cuComplex(real + a.real, imag + a.imag);
//    }
//    __device__ cuComplex operator-(const cuComplex& a)
//    {
//        return cuComplex(real - a.real, imag - a.imag);
//    }
//    __device__ cuComplex operator*(const cuComplex& a)
//    {
//        return cuComplex(real * a.real - imag * a.imag, real * a.imag + imag * a.real);
//    }
//};
//
//__device__ void cuComplexSwap(cuComplex& a, cuComplex& b)
//{
//    cuComplex temp = a;
//    a = b;
//    b = temp;
//}
//
//__device__ cuComplex cuComplexExp(double theta)
//{
//    return cuComplex(cos(theta), sin(theta));
//}
//
//__global__ void fft2_kernel(cuComplex* input, int n)
//{
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//    // 数据重排
//    for (int i = 1, j = 0; i < n; i++)
//    {
//        int bit = n >> 1;
//        for (; j >= bit; bit >>= 1) j -= bit;
//        j += bit;
//        if (i < j)
//            cuComplexSwap(input[i], input[j]);
//    }
//
//    // 蝴蝶运算
//    for (int k = 2; k <= n; k <<= 1)
//    {
//        int m = k >> 1;
//        double theta = - PI / m;
//        cuComplex w_m = cuComplexExp(theta);
//
//        for (int i = tid; i < n; i += blockDim.x * gridDim.x)
//        {
//            cuComplex w = cuComplex(1.0, 0.0);
//            for (int j = 0; j < m; j++)
//            {
//                cuComplex t = w * input[i + j + m];
//                input[i + j + m] = input[i + j] - t;
//                input[i + j] = input[i + j] + t;
//                w = w * w_m;
//            }
//        }
//    }
//}
//
//void fft(std::vector<std::complex<double>>& input)
//{
//    int n = input.size();   // 
//    std::vector<cuComplex> input_gpu(n);
//    for (int i = 0; i < n; i++)
//    {
//        input_gpu[i].real = input[i].real();
//        input_gpu[i].imag = input[i].imag();
//    }
//
//    cuComplex* input_device;
//    cudaMalloc((void**)&input_device, sizeof(cuComplex) * n);
//    cudaMemcpy(input_device, input_gpu.data(), sizeof(cuComplex) * n, cudaMemcpyHostToDevice);
//
//    int threadsPerBlock = 128;
//    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
//
//    fft2_kernel <<<blocksPerGrid, threadsPerBlock>>> (input_device, n);
//
//    cudaMemcpy(input_gpu.data(), input_device, sizeof(cuComplex) * n, cudaMemcpyDeviceToHost);
//    cudaFree(input_device);
//
//    input.resize(n);
//    for (int i = 0; i < n; i++)
//    {
//        input[i] = std::complex<double>(input_gpu[i].real / n, input_gpu[i].imag /n );
//    }
//}

//void fft2(vector<complex<double>>& input)
//{
//    int n = input.size();
//
//    // 数据重排
//    for (int i = 1, j = 0; i < n; i++)
//    {
//        int bit = n >> 1;
//        for (; j >= bit; bit >>= 1) j -= bit;
//        j += bit;
//        if (i < j) swap(input[i], input[j]);
//    }
//    // 蝴蝶运算
//    for (int k = 2; k <= n; k <<= 1)
//    {
//        int m = k >> 1;
//        complex<double> w_m(cos(PI / m), -sin(PI / m));
//
//        for (int i = 0; i < n; i += k)
//        {
//            complex<double> w(1);
//            for (int j = 0; j < m; j++)
//            {
//                complex<double> t = w * input[i + j + m];
//                input[i + j + m] = input[i + j] - t;
//                input[i + j] += t;
//                w *= w_m;
//            }
//        }
//    }
//}

//int main()
//{
//    ifstream fi("fft_1024.txt");
//    vector<double> data;
//    string read_temp;
//    while (fi.good())
//    {
//        getline(fi, read_temp);
//        data.push_back(stod(read_temp));
//    }
//    fi.close();
//    vector<complex<double> > input(data.size());
//    for (int i = 0; i < data.size(); i++)
//    {
//        input[i] = complex<double>(data[i], 0);
//    }
//
//    // 执行FFT
//    fft(input);
//
//    ofstream fo;
//    fo.open("fft_cuda_result.txt", ios::out);
//    for (int i = 0; i < data.size(); i++)
//    {
//        fo << '(' << input[i].real() << ',' << input[i].imag() << ')' << endl;
//    }
//    fo.close();
//
//    return 0;
//}

const int N = 8388608;

void cufft(cufftComplex* input, cufftComplex *data)
{
    cufftComplex* output;
    cufftHandle plan;
    // 分配GPU内存
    cudaMalloc((void**)&input, sizeof(cufftComplex) * N);
    cudaMalloc((void**)&output, sizeof(cufftComplex) * N);
    // 创建FFT计划
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    // 将输入数据从主机内存复制到GPU内存
    cudaMemcpy(input, data, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    // 执行FFT计算
    cufftExecC2C(plan, input, output, CUFFT_FORWARD);

    // 将结果数据从GPU内存复制回主机内存
    cudaMemcpy(data, output, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);


    // 释放内存和计划
    cudaFree(input);
    cudaFree(output);
    cufftDestroy(plan);
}

cufftComplex *input;
vector<double> v;
cufftComplex *data1;

int main()
{
    ifstream fi("fft_8388608.txt");
    input = new cufftComplex[N];
    data1 = new cufftComplex[N];
    string read_temp;
    while (fi.good())
    {
        getline(fi, read_temp);
        v.push_back(stod(read_temp));
    }
    fi.close();

    for (int i = 0; i < N; ++i) {
        data1[i].x = v[i];
        data1[i].y = 0;
    }

    auto t1 = Clock::now();
    int count = 1000;
    for (int i = 0; i < count; i++)
    {
        cufft(input, data1);
    }
    auto t2 = Clock::now();// 计时结束
    cout << "cufft cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 / count << " ms.\n";
    // 打印结果
    //ofstream fo;
    //fo.open("fft_cuda_result.txt", ios::out);
    //for (int i = 0; i < N; i++)
    //{
    //    fo << '(' << data[i].x << ',' << data[i].y << ')' << endl;
    //}
    //fo.close();
    return 0;
}

