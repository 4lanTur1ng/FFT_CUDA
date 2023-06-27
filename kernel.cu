#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>
#include <fstream>
#include <chrono>
#include <vector>

using namespace std;
using namespace chrono;
typedef std::chrono::high_resolution_clock Clock;

#define PI 3.1415926535897932384626433832

class Complex {
public:
    double real;
    double imag;

    Complex() {

    }

    // Wn 获取n次单位复根中的主单位根
    __device__ static Complex W(int n) {
        Complex res = Complex(cos(2.0 * PI / n), sin(2.0 * PI / n));
        return res;
    }

    // Wn^k 获取n次单位复根中的第k个
    __device__ static Complex W(int n, int k) {
        Complex res = Complex(cos(2.0 * PI * k / n), sin(2.0 * PI * k / n));
        return res;
    }

    // 实例化并返回一个复数（只能在Host调用）
    static Complex GetComplex(double real, double imag) {
        Complex r;
        r.real = real;
        r.imag = imag;
        return r;
    }

    // 随机返回一个复数
    static Complex GetRandomComplex() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = (double)rand() / rand();
        return r;
    }

    // 随即返回一个实数
    static Complex GetRandomReal() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = 0;
        return r;
    }

    // 随即返回一个纯虚数
    static Complex GetRandomPureImag() {
        Complex r;
        r.real = 0;
        r.imag = (double)rand() / rand();
        return r;
    }

    // 构造函数（只能在Device上调用）
    __device__ Complex(double real, double imag) {
        this->real = real;
        this->imag = imag;
    }

    // 运算符重载
    __device__ Complex operator+(const Complex& other) {
        Complex res(this->real + other.real, this->imag + other.imag);
        return res;
    }

    __device__ Complex operator-(const Complex& other) {
        Complex res(this->real - other.real, this->imag - other.imag);
        return res;
    }

    __device__ Complex operator*(const Complex& other) {
        Complex res(this->real * other.real - this->imag * other.imag, this->imag * other.real + this->real * other.imag);
        return res;
    }
};

// 根据数列长度n获取二进制位数
int GetBits(int n) {
    int bits = 0;
    while (n >>= 1) {
        bits++;
    }
    return bits;
}
// 在二进制位数为bits的前提下求数值i的二进制逆转
__device__ int BinaryReverse(int i, int bits) {
    int r = 0;
    do {
        r += i % 2 << --bits;
    } while (i /= 2);
    return r;
}
// 蝴蝶操作, 输出结果直接覆盖原存储单元的数据, factor是旋转因子
__device__ void Bufferfly(Complex* a, Complex* b, Complex factor) {
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *a = a1;
    *b = b1;
}
__global__ void FFT(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            int k = i;
            if (n - tid < k) k = n - tid;
            for (int j = 0; j < k / 2; ++j) {
                Bufferfly(&nums[BinaryReverse(tid + j, bits)], &nums[BinaryReverse(tid + j + k / 2, bits)], Complex::W(k, -j));
            }
        }
        __syncthreads();
    }
    result[tid] = nums[BinaryReverse(tid, bits)];
}

int main() {
    const int TPB = 2048; // 每个Block的线程数，即blockDim.x
    int count = 100;
    ifstream fi("fft_1024.txt");
    vector<double> data;
    string read_temp;
    while (fi.good()) {
        getline(fi, read_temp);
        data.push_back(stod(read_temp));
    }
    fi.close();
    const int N = data.size(); // 数列大小
    const int bits = GetBits(N);
    // 生成实数数列
    Complex* nums = (Complex*)malloc(sizeof(Complex) * N), * dNums, * dResult;
    for (int i = 0; i < N; ++i) {
        nums[i].real = data[i];
        nums[i].imag = 0.0;
    }
    printf("Length of Sequence: %d\n", N);
    // 保存开始时间

    auto t1 = Clock::now();
    // 分配device内存，拷贝数据到device
    auto ans = 0.0;
    for (int i = 0; i < count; i++) {
        auto t3 = Clock::now();
        cudaMalloc((void**)&dNums, sizeof(Complex) * N);
        cudaMalloc((void**)&dResult, sizeof(Complex) * N);
        cudaMemcpy(dNums, nums, sizeof(Complex) * N, cudaMemcpyHostToDevice);

        // 调用kernel
        dim3 threadPerBlock = dim3(TPB);
        dim3 blockNum = dim3((N + threadPerBlock.x - 1) / threadPerBlock.x);
        FFT << <blockNum, threadPerBlock >> > (dNums, dResult, N, bits);
        //// 拷贝回结果
        //cudaMemcpyAsync(nums, dResult, sizeof(Complex) * N, cudaMemcpyDeviceToHost);
        //// 同步等待数据传输完成
        //cudaDeviceSynchronize();
        
        // 拷贝回结果
        cudaMemcpy(nums, dResult, sizeof(Complex) * N, cudaMemcpyDeviceToHost);
        // 释放内存
        cudaFree(dNums);
        cudaFree(dResult);
        auto t4 = Clock::now(); // 计时结束
        cout << "fft_cuda cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e+6 << " ms.\n";
    }

    auto t2 = Clock::now(); // 计时结束
    cout << "fft_cuda cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";

    //ofstream fo;
    //fo.open("fft_cuda_result.txt", ios::out);
    //for (int i = 0; i < data.size(); i++)
    //{
    //    fo << '(' << nums[i].real << ',' << nums[i].imag << ')' << endl;
    //}
    //fo.close();

    // 释放内存
    free(nums);
}
