#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define blockSize 1024

#define TAM 8*2*2*2*2*2*2*2




__global__ void vecRed(int *d_M, int *d_P, int Width){

	__shared__ float partialSum[TAM];

	unsigned int t = threadIdx.x;
  	unsigned int i = blockIdx.x*blockDim.x+ threadIdx.x;
  	partialSum[t] = d_M[i];
  	__syncthreads();
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
	{	
	    if(t%(2*stride)==0){
				partialSum[t] += partialSum[t+stride];
	    }
    	__syncthreads();
    
	}
	if(t==0){
	   d_P[blockIdx.x]=partialSum[0];
	}

}



void reducirp(int *A, int *Rp, int n){
	int size= n*sizeof(int);
	int *d_A;
	int *d_C;
  
	cudaMalloc((void **)&d_A,size);
	cudaMalloc((void **)&d_C,size);	
  
  	clock_t t2;
  	t2 = clock();

	cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice);

  	float dimGrid= ceil((float)n/(float)blockSize);
  
		vecRed<<<  dimGrid, n >>>(d_A, d_C, n);												//ejecuta el kernel ,,n-> numero de hilos por block, max 1024
		cudaMemcpy( Rp,d_C, size, cudaMemcpyDeviceToHost);
  	
  	t2 = clock() - t2;
  	printf ("\nTiempo desde la GPU: (%f seconds).\n",((float)t2)/CLOCKS_PER_SEC);

	cudaFree(d_A);																			//libera memoria del dispositivo
	cudaFree(d_C);
  
}



void reducir(int *A, int *R, int n){
 	clock_t t;
   	t = clock();
   	for(int i=0;i<TAM;i++){
     	*R += A[i];
   	}
   	
   	t = clock() - t;
  	printf ("Tiempo desde la CPU: (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);
   
}
    

int main(){
	int n; //longitud del vector
	int * A;
	int *R;
	int *Rp;

  	n=TAM;

	A = 	(int*)malloc( n*sizeof(int) );
  	R = 	(int*)malloc( sizeof(int) );
	Rp = 	(int*)malloc( n*sizeof(int) );

	for(int i=0;i<n;i++){
		A[i]=i;
	}

	//vecAddGPU(A,B,C);
  	reducir(A,R, n);
  	printf("Resultado CPU -> %d\n",R[0]);
  
  	reducirp(A,Rp,n);
  	printf("Resultado GPU -> %d",Rp[0]);
  
	return 0;
}