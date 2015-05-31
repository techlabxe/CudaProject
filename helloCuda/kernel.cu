#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

using namespace std;

__global__
void addKernel( int* a, int* b, int* c )
{
    int idx = threadIdx.x;
    c[idx] = a[idx] + b[idx];
}
void testCudaAdd() {
    const int COUNT = 10;
    int* a = new int[COUNT];
    int* b = new int[COUNT];
    int* c = new int[COUNT];

    for( int i=0;i<COUNT;++i) {
        a[i] = i;
        b[i] = i*100;
    }

    int *devArrayA = 0, *devArrayB = 0, *devArrayC = 0;
    cudaMalloc( &devArrayA, sizeof(int) * COUNT );
    cudaMalloc( &devArrayB, sizeof(int) * COUNT );
    cudaMalloc( &devArrayC, sizeof(int) * COUNT );

    // 入力データの転送.
    cudaMemcpy( devArrayA, a, sizeof(int)*COUNT, cudaMemcpyHostToDevice );
    cudaMemcpy( devArrayB, b, sizeof(int)*COUNT, cudaMemcpyHostToDevice );

    // 実行.
    addKernel<<<1, COUNT>>>(devArrayA, devArrayB, devArrayC );
    cudaDeviceSynchronize();

    // 結果の読み戻し.
    cudaMemcpy( c, devArrayC, sizeof(int)*COUNT, cudaMemcpyDeviceToHost );
    for(int i=0;i<COUNT;++i) {
        printf( "%d ", c[i] );
    }
    printf( "\n" );

    cudaFree( devArrayC );
    cudaFree( devArrayB );
    cudaFree( devArrayA );

    delete[] a;
    delete[] b;
    delete[] c;
}

__global__
void multMatrix( float* a, float* b, float* c, int COUNT )
{
    int idx = blockDim.x * threadIdx.y + threadIdx.x;
    int idxCol = blockDim.x * blockIdx.x + threadIdx.x;
    int idxRow = blockDim.y * blockIdx.y + threadIdx.y;
    float scanSum = 0;
    for( int i=0;i<COUNT;++i) {
        if( idxCol >= COUNT || idxRow >= COUNT ) {
            continue;
        }
#if 01
        scanSum += a[ idxRow*COUNT + i ] * b[ idxCol + i*COUNT ]; 
#else
        // 制度の問題が出たら.
        scanSum = __fadd_rn( scanSum, __fmul_rn( a[idxRow*COUNT+i], b[idxCol+i*COUNT] ) );
#endif
    }
    if( idxCol < COUNT && idxRow < COUNT ) {
        c[idxCol+idxRow*COUNT] = scanSum;
    }
}
void testCudaMult() {
    const int COUNT = 1024;
    const int SIZE = COUNT*COUNT; // 行列サイズ
    float* a = new float[SIZE];
    float* b = new float[SIZE];
    float* c = new float[SIZE];
    for( int i=0;i<SIZE;++i) {
        a[i] = float(0.001f * i );
        b[i] = float(0.005f * i );
    }

    chrono::high_resolution_clock::time_point start, stop;
    start = chrono::high_resolution_clock::now();
    for( int i=0;i<COUNT;++i ) {
        for( int j=0;j<COUNT;++j ) {
            float tmp = float(0);
            for(int t=0;t<COUNT;++t) {
                tmp += a[t+i*COUNT] * b[j+COUNT*t];
            }
            c[i*COUNT+j] = tmp;
        }
    }
    stop = chrono::high_resolution_clock::now();

#if 0
    for(int i=0;i<COUNT;++i) {
        for(int j=0;j<COUNT;++j) {
            printf( "%f ", c[i*COUNT+j] );
        }
        printf( "\n" );
    }
#endif
    chrono::microseconds cpuTime = chrono::duration_cast<chrono::microseconds>(stop-start);
    printf( "CPU: %d (us)\n", cpuTime.count() );
    printf( "\n" );

    float* gpuC = new float[SIZE];
    for( int i=0;i<SIZE;++i) {
        a[i] = float(0.001f * i );
        b[i] = float(0.005f * i );
        gpuC[i] = 0.0f;
    }

    float *devA, *devB, *devC;
    cudaMalloc( &devA, sizeof(float) * SIZE );
    cudaMalloc( &devB, sizeof(float) * SIZE );
    cudaMalloc( &devC, sizeof(float) * SIZE );
    cudaMemcpy( devA, a, sizeof(float)*SIZE, cudaMemcpyHostToDevice );
    cudaMemcpy( devB, b, sizeof(float)*SIZE, cudaMemcpyHostToDevice );
    
    start = chrono::high_resolution_clock::now();

    const int thrCount=32;
    int blockXY = ( COUNT + (thrCount-1) ) / thrCount;
    dim3 blk( blockXY,blockXY);
    dim3 thr( thrCount,thrCount);
    multMatrix<<<blk,thr>>>( devA, devB, devC, COUNT );
    cudaDeviceSynchronize();
    stop = chrono::high_resolution_clock::now();
    cudaMemcpy( gpuC, devC, sizeof(float)*SIZE, cudaMemcpyDeviceToHost );
#if 0
    for(int i=0;i<COUNT;++i) {
        for(int j=0;j<COUNT;++j) {
            printf( "%.2f(%.2f) ", gpuC[i*COUNT+j], c[i*COUNT+j] );
        }
        printf( "\n" );
    }
#endif
    int mismatchCount = 0;
    for(int i=0;i<SIZE;++i) {
        if( c[i] != gpuC[i] ) {
            mismatchCount++;
        }
    }
    if( mismatchCount > 0 ) {
        printf( "mismatchCount= %d (%d)\n", mismatchCount, SIZE );
    }

    chrono::microseconds gpuTime = chrono::duration_cast<chrono::microseconds>(stop-start);
    printf( "GPU: %d (us)\n", gpuTime.count() );
    printf( "\n" );
    printf( "rate = %.4f (%dx%d matrix)\n", cpuTime.count() / (double)gpuTime.count(), COUNT, COUNT );
}

int main() {
    printf( "Hello,CUDA\n" );
    testCudaAdd();
    testCudaMult();
    cudaDeviceReset();
    return 0;
}
