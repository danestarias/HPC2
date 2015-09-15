#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 3
#define blockSize 1024


__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width){
	
	int Row = blockIdx.y*blockDim.y+threadIdx.y;

	// Calculate the coumn index of d_Pelement and d_M
	int Col = blockIdx.x*blockDim.x+threadIdx.x;

	if((Row < Width) && (Col < Width)){
		float Pvalue = 0;
		for (int k=0;k < Width;k++){
			Pvalue += d_M[Row*Width+k]*d_N[k*Width+Col];
		}
		d_P[Row*Width+Col] = Pvalue;

	}

}



void vectorAdd(int *A, int *B, int *C, int n){
	int size= n*sizeof(int);
	float *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A,size);															//reserva memoria en el device
	cudaMalloc((void **)&d_B,size);
	cudaMalloc((void **)&d_C,size);
  
  	clock_t t2;
  	t2 = clock();

	cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice);										//se copian al device
	cudaMemcpy( d_B, B, size, cudaMemcpyHostToDevice);

	float dimGrid= ceil((float)n/(float)blockSize);

	MatrixMulKernel<<< dimGrid, n >>>(d_A, d_B, d_C, n);												//ejecuta el kernel ,,n-> numero de hilos por block, max 1024
	cudaMemcpy( C,d_C, size, cudaMemcpyDeviceToHost);
  
  	t2 = clock() - t2;

  	printf("\nResultados de la gpu\n");
	for(int fil=0;fil<n;fil++){
		for(int col=0;col<n;col++){
	    	printf("%d",C[fil*n+col]);
	    }
	    printf("\n");
	}
  	printf ("\nTiempo desde la GPU: (%f seconds).\n",((float)t2)/CLOCKS_PER_SEC);

	cudaFree(d_A);																			//libera memoria del dispositivo
	cudaFree(d_B);
	cudaFree(d_C);

}

void multiplicar(int *A, int *B, int *C, int filas, int columnas){
 	clock_t t;
   	t = clock();
    int acomulado;
    for(int it=0;it<filas*columnas;it++){
	   	for(int fil=0;fil<filas;fil++){
	   		acomulado=0;
			for(int col=0;col<columnas;col++){
				acomulado += A[fil*columnas+col]*B[col*columnas+fil];
	     	}
	     	C[it]= acomulado;
	   	}	
	    /*printf("%d",C[it]);
	    if(((it+1)%filas)==0){
	     	printf("\n");
	    }*/
	   	
   	}
   	t = clock() - t;
  	printf ("Tiempo desde la CPU: (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);
}
    

int main(){
	int n; //longitud del vector
	int * A;
	int * B;
	int * C;
  	n=TAM;
  	int filas=n;
  	int columnas=n;

	A = (int*)malloc( filas*columnas*sizeof(int) );
	B = (int*)malloc( filas*columnas*sizeof(int) );
	C = (int*)malloc( filas*columnas*sizeof(int) );

	for(int fil=0;fil<filas;fil++){
		for(int col=0;col<columnas;col++){
			A[fil*columnas+col]=1;
			B[fil*columnas+col]=1;
		}
	}

	//vecAddGPU(A,B,C);
  	multiplicar(A,B,C,filas,columnas);
  	vectorAdd(A,B,C,n);

	return 0;
}