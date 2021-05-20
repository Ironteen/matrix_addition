#### 每天一个CUDA小技巧 ： 矩阵加法的加速



> 题目：两个4096 x 4096大小的矩阵进行相加

##### CPU实现

- 思路：For 循环，每个点进行相加

- 步骤：

  - 分配内存，初始化两个矩阵的值

    ```
    int nx=1<<12;
    int ny=1<<12;
    int nxy=nx*ny;
    int nBytes=nxy*sizeof(float);
    
    //Malloc
    float* A_host=(float*)malloc(sizeof(float)*nBytes);
    float* B_host=(float*)malloc(sizeof(float)*nBytes);
    float* C_host=(float*)malloc(sizeof(float)*nBytes);
    
    initialData(A_host,nxy);
    initialData(B_host,nxy);
    ```

    其中初始化函数如下：

    ```
    void initialData(float* ip,int size){
      time_t t;
      srand((unsigned )time(&t)); 
      for(int i=0;i<size;i++){
        ip[i]=(float)(rand()&0xffff)/1000.0f;
      }
    }
    ```

  - 逐点相加，以指针的形式传递数据

    ```
    void sumMatrix2D_CPU(float * MatA,float * MatB,float * MatC,int nx,int ny){
        float * a=MatA;
        float * b=MatB;
        float * c=MatC;
        for(int j=0;j<ny;j++){
        	for(int i=0;i<nx;i++){
          		c[i]=a[i]+b[i];
        	}
            c+=nx;
            b+=nx;
            a+=nx;
        }
    }
    
    sumMatrix2D_CPU(A_host,B_host,C_host,nx,ny);
    ```

    更简洁的实现方法

    ```
    void sumMatrix2D_CPU(float * MatA,float * MatB,float * MatC,int nx,int ny){
        for(int j=0;j<ny;j++){
            for(int i=0;i<nx;i++){
                int ind = j*nx + i;
                MatC[ind]=MatA[ind]+MatB[ind];
            }
        }
    }
    
    sumMatrix2D_CPU(A_host,B_host,C_host,nx,ny);
    ```

  - 释放内存

    ```
    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);
    ```

##### CUDA实现

- 思路：矩阵各个点的相加互相不干扰，可以多个线程并行计算

- 步骤：

  - 分配内存，从主机(host) 向 显卡 (device)搬运原始数据

    ```
    float* C_from_gpu = (float*)malloc(nBytes);
    
    // cudaMalloc
    float *A_dev, *B_dev, *C_dev;
    
    cudaMalloc((void**)&A_dev,sizeof(float)*nBytes);
    cudaMalloc((void**)&B_dev,sizeof(float)*nBytes);
    cudaMalloc((void**)&C_dev,sizeof(float)*nBytes);
    
    cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev,B_host,nBytes,cudaMemcpyHostToDevice);
    ```

  - 设置并行数，分配线程

    ```
    int dimx=32, dimy=32;
    
    // 2d block and 2d grid
    dim3 block_0(dimx,dimy);
    dim3 grid_0((nx-1)/block_0.x+1, (ny-1)/block_0.y+1);
    ```

    这里相当于将原始的4096 x 4096的矩阵划分成128 x 128个小块，每个小块都是32 x 32的小矩阵，然后每个点都会单独分配一个线程去进行加法计算

  - 启动核函数，进行计算，并进行同步

    ```
    __global__ void sumMatrix(float * MatA,float * MatB,float * MatC,int nx,int ny)
    {
        int ix = blockDim.x*blockIdx.x + threadIdx.x;
        int iy = blockDim.y*blockIdx.y + threadIdx.y;
        int idx = iy*ny + ix;
        if (ix<nx && iy<ny){
          MatC[idx] = MatA[idx]+MatB[idx];
        }
    }
    
    sumMatrix<<<grid_0,block_0>>>(A_dev,B_dev,C_dev,nx,ny);
    cudaDeviceSynchronize();
    ```

  - 将计算结果从显卡 (device)搬运回主机(host)

    ```
    cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost)
    ```

  - 释放内存，重置 CUDA

    ```
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    cudaDeviceReset();
    ```

#### 检查结果并时间对比

```
Using device 0: GeForce RTX 2080 Ti
CPU Execution Time elapsed 0.069137 sec
grid : (128, 128), block : (32, 32)
GPU Execution configuration<<<(128,128),(32,32)>>> Time elapsed 0.000416 sec
Check result success!
```

**加速比** ： 0.069137 / 0.000416 = 166.19
