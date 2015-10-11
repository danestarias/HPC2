#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 512
#define blockSize 1024

#define TILE_WIDTH 32


__global__ void vecAdd(int *d_M, int *d_N, int *d_P, int Width){

	__shared__ float partialSum[TILE_WIDTH];

	unsigned int t = threadIdx.x;
	//printf("entre ");
	for (unsigned int stride = blockDim.x; stride > 1; stride /= 2)
	{
		__syncthreads();
    if(t< stride){
			partialSum[t] += partialSum[t+stride];
    }else{
    	d_P[t]=partialSum[t];
    }
    
  }

}



void vectorAdd(int *A, int *B, int *CM, int n){
	int size= n*sizeof(int);
	int *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A,size);															//reserva memoria en el device
	cudaMalloc((void **)&d_B,size);
	cudaMalloc((void **)&d_C,size);
  
  	clock_t t2;
  	t2 = clock();

	cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice);										//se copian al device
	cudaMemcpy( d_B, B, size, cudaMemcpyHostToDevice);

	//int dimGrid= ceil((float)n/(float)blockSize);
  
  
  int numBlockX=32;
  
	while(n % numBlockX!=0){
		numBlockX = numBlockX-1;
	}
	printf("numero de bloques en X -> %d \n",numBlockX);
  
  dim3 dimBlock(numBlockX,1,1); 		
  dim3 dimGrid(ceil(n/dimBlock.x),1,1);
  
  

	vecAdd<<<  dimGrid, dimBlock >>>(d_A, d_B, d_C, n);												//ejecuta el kernel ,,n-> numero de hilos por block, max 1024
	cudaMemcpy( CM,d_C, size, cudaMemcpyDeviceToHost);
  

  
  	t2 = clock() - t2;
  	printf ("\nTiempo desde la GPU: (%f seconds).\n",((float)t2)/CLOCKS_PER_SEC);

	cudaFree(d_A);																			//libera memoria del dispositivo
	cudaFree(d_B);
	cudaFree(d_C);

}


void sumar(int *A, int *B, int *C, int n){
 	clock_t t;
   	t = clock();
   
   	for(int i=0;i<(n/2);i++){
     	C[i]= A[i]+B[i];
      C[(n/2)+i]= A[(n/2)+i]+B[(n/2)+i];
     //printf("%d",C[i]);
      //printf("%d",C[(n/2)+i]);
      //printf("\n");
   	}
   
   	t = clock() - t;
  	printf ("Tiempo desde la CPU: (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);
   
}
    

int main(){
	int n; //longitud del vector
	int * A;
	int * B;
	int * C;
  int * CM;
  	n=TAM;

	A = (int*)malloc( n*sizeof(int) );
	B = (int*)malloc( n*sizeof(int) );
	C = (int*)malloc( n*sizeof(int) );
  CM = (int*)malloc( n*sizeof(int) );

	for(int i=0;i<n;i++){
		A[i]=rand() % 3;
    	//printf("%d",A[i]);
		B[i]=rand() % 3;
    	//printf("%d\n",B[i]);
	}

	//vecAddGPU(A,B,C);
  	sumar(A,B,C,n);
  	vectorAdd(A,B,CM,n);
  
  int entre=0;
  for(int i=0;i<n;i++){
				//--printf("%d ",CK[fil*columnasB+col]);
    		printf("%d ",C[i]);
    		printf("%d \n",CM[i]);
				if(C[i]==CM[i]){
					entre++;
				}
			}
  
  printf("coincidencias --> %d ",entre);
	return 0;
}