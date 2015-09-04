#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 3000
#define blockSize 1024


__global__ void vecAdd(float *A, float *B, float *C, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x; // a*b+c -- a->id del bloque utilizado
	if(i<n){
		C[i]=A[i]+B[i];
	}
}
 

void vectorAdd(int *A, int *B, int *C, int n){
	int size= n*sizeof(int);
	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A,size);							//reserva memoria en el device
	cudaMalloc((void **)&d_B,size);
	cudaMalloc((void **)&d_C,size);
  
  	clock_t t2;
  	t2 = clock();

	cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice);		//se copian al device
	cudaMemcpy( d_B, B, size, cudaMemcpyHostToDevice);

	float dimGrid= ceil(n/blockSize);

	vecAdd<<< dimGrid, n >>>(d_A, d_B, d_C, n);					//ejecuta el kernel ,,n-> numero de hilos por block, max 1024
	cudaMemcpy( C,d_C, size, cudaMemcpyDeviceToHost);
  
  	t2 = clock() - t2;
  	printf ("\nTiempo desde la GPU: (%f seconds).\n",((float)t2)/CLOCKS_PER_SEC);

	cudaFree(d_A);											//libera memoria del dispositivo
	cudaFree(d_B);
	cudaFree(d_C);

}


int * sumar(int *A, int *B, int *C, int n){
 	clock_t t;
   	t = clock();
   
   	for(int i=0;i<n;i++){
     	C[i]= A[i]+B[i];
     //printf("%d",C[i]);
   	}
   
   	t = clock() - t;
  	printf ("Tiempo desde la CPU: (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);
   
   	return C;
}
    

int main(){
	int n; //longitud del vector
	int * A;
	int * B;
	int * C;
  	n=TAM;

	A = (int*)malloc( n*sizeof(int) );
	B = (int*)malloc( n*sizeof(int) );
	C = (int*)malloc( n*sizeof(int) );

	for(int i=0;i<n;i++){
		A[i]=rand() % 10;
    	//printf("%d",A[i]);
		B[i]=rand() % 10;
    	//printf("%d\n",B[i]);
	}

	//vecAddGPU(A,B,C);
  	sumar(A,B,C,n);
  	vectorAdd(A,B,C,n);

	return 0;
}