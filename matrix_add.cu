#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.h"

void sumMatrix2D_CPU(float * MatA,float * MatB,float * MatC,int nx,int ny){
    for(int j=0;j<ny;j++){
        for(int i=0;i<nx;i++){
            int ind = j*nx + i;
            MatC[ind] = MatA[ind] + MatB[ind];
        }
    }
}

__global__ void sumMatrix(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    int ix = blockDim.x*blockIdx.x + threadIdx.x;
    int iy = blockDim.y*blockIdx.y + threadIdx.y;
    int idx = ix + iy*ny;
    if (ix<nx && iy<ny){
      MatC[idx]=MatA[idx]+MatB[idx];
    }
}

int main(int argc,char** argv){
    // init devices
    initDevice(0);
    int nx=1<<12;
    int ny=1<<12;
    int nxy=nx*ny;
    int nBytes=nxy*sizeof(float);

    //Malloc
    float* A_host=(float*)malloc(sizeof(float) * nBytes);
    float* B_host=(float*)malloc(sizeof(float) * nBytes);
    float* C_host=(float*)malloc(sizeof(float) * nBytes);

    initialData(A_host,nxy);
    initialData(B_host,nxy);

    //cudaMalloc
    float* C_from_gpu=(float*)malloc(sizeof(float) * nBytes);
    float *A_dev, *B_dev, *C_dev;
    cudaMalloc((void**)&A_dev,sizeof(float) * nBytes);
    cudaMalloc((void**)&B_dev,sizeof(float) * nBytes);
    cudaMalloc((void**)&C_dev,sizeof(float) * nBytes);

    cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev,B_host,nBytes,cudaMemcpyHostToDevice);

    int dimx=32, dimy=32;

    // cpu compute
    double iStart=cpuSecond();

    sumMatrix2D_CPU(A_host,B_host,C_host,nx,ny);

    double iElaps=cpuSecond()-iStart;
    printf("CPU Execution Time elapsed %f sec\n",iElaps);

    // 2d block and 2d grid
    dim3 block_0(dimx,dimy);
    dim3 grid_0((nx-1)/block_0.x+1,(ny-1)/block_0.y+1);
    printf("grid : (%d, %d), block : (%d, %d)\n", grid_0.x, grid_0.y, block_0.x, block_0.y);

    iStart=cpuSecond();

    sumMatrix<<<grid_0,block_0>>>(A_dev,B_dev,C_dev,nx,ny);
    cudaDeviceSynchronize();

    iElaps=cpuSecond()-iStart;
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
        grid_0.x,grid_0.y,block_0.x,block_0.y,iElaps);
    cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost);

    // checkresult
    checkResult(C_host,C_from_gpu,nxy);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    cudaDeviceReset();

    free(C_from_gpu);
    free(A_host);
    free(B_host);
    free(C_host);

    return 0;
}
