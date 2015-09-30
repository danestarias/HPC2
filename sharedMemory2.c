#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>

#define TAM 200
#define blockSize 1024
#define TILE_WIDTH 25

__global__ void MatrixMulKernelShared(int *d_M, int *d_N, int *d_P, int Width){

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;	int ty = threadIdx.y;

	// Identifico la fila y la columna de el elemento d_P a trabajar
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	int Pvalue = 0;
	// Loop over the d_M and d_N tiles required to coompute d_p element
	for (int m=0; m< Width/TILE_WIDTH; ++m){

		// Colaborative loading of d_M and d_N tiles into shared memory
		Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
		Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();

		for (int k=0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	d_P[Row*Width + Col] = Pvalue;
}


__global__ void MatrixMulKernel(int *d_M, int *d_N, int *d_P, int Width){
	
	int Row = blockIdx.y*blockDim.y+threadIdx.y;

	// Calculate the coumn index of d_Pelement and d_M
	int Col = blockIdx.x*blockDim.x+threadIdx.x;

	if((Row < Width) && (Col < Width)){
		int Pvalue = 0;
		for (int k=0;k < Width;k++){
			Pvalue += d_M[Row*Width+k]*d_N[k*Width+Col];
		}
		d_P[Row*Width+Col] = Pvalue;
	}

}


void vectorAdd(int *A, int *B, int *CK, int *CKS, int n){
	int size= n*n*sizeof(int);
	int *d_A, *d_B, *d_C, *d_CKS;
	cudaMalloc((void **)&d_A,size);															//reserva memoria en el device
	cudaMalloc((void **)&d_B,size);
	cudaMalloc((void **)&d_C,size);
	cudaMalloc((void **)&d_CKS,size);


	int numBlock=32;
	while(n%numBlock!=0){
		numBlock=numBlock-1;
	}
	printf("numero de bloques -> %d \n",numBlock);
  
  	clock_t t2;
  	t2 = clock();	//tiempo de asignacion de memoria
	cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice);										//se copian al device
	cudaMemcpy( d_B, B, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(numBlock,numBlock,1); 		//->30 hx 30 hy ->900 hilos ->100 bloques (bloques en X y Y?)
  	dim3 dimGrid(ceil(n/dimBlock.x),ceil(n/dimBlock.y),1);	//100 bloques en X y 100 bloques en Y
  	t2 = clock() - t2;

  	clock_t t3;
  	t3 = clock();
	MatrixMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, n);												//ejecuta el kernel ,,n-> numero de hilos por block, max 1024
	cudaMemcpy( CK,d_C, size, cudaMemcpyDeviceToHost);
	t3 = clock() - t3;

  	clock_t t4;
  	t4 = clock();
  	MatrixMulKernelShared<<< dimGrid, dimBlock >>>(d_A, d_B, d_CKS, n);												//ejecuta el kernel ,,n-> numero de hilos por block, max 1024
	cudaMemcpy( CKS,d_CKS, size, cudaMemcpyDeviceToHost);
  	t4 = clock() - t4;

  	printf ("\nTiempo desde la GPU: (%f seconds).\n",((float)(t2+t3))/CLOCKS_PER_SEC);
  	printf ("\nTiempo desde la GPU con memoria compartida: (%f seconds).\n",((float)(t2+t4))/CLOCKS_PER_SEC);

	cudaFree(d_A);																			//libera memoria del dispositivo
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_CKS);

}

void multiplicar(int *A, int *B, int *C, int filas, int columnas){
 	clock_t t;
   	t = clock();

    for(int it=0;it<filas;it++){

	   	for(int fil=0;fil<filas;fil++){
			for(int col=0;col<columnas;col++){
	        	C[it*columnas+fil] += A[it*columnas+col]*B[col*columnas+fil];
	          	//printf("%d ",A[fil*columnas+col]);
		     }
        	//printf("%d ",C[it*columnas+fil]);
	   	}	
      	//printf("\n");
   	}
   	t = clock() - t;
  	printf ("Tiempo desde la CPU: (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);
}
    

int main(){
	int * A;
	int * B;
	int * C;
  	int * CK;  	
  	int * CKS;

  	int	n=TAM;
  	int filas=n;
  	int columnas=n;

	A = (int*)malloc( filas*columnas*sizeof(int) );
	B = (int*)malloc( filas*columnas*sizeof(int) );
	C = (int*)malloc( filas*columnas*sizeof(int) );		//para resultado secuencial
	CK = (int*)malloc( filas*columnas*sizeof(int) );	//para resultado paralelo
	CKS = (int*)malloc( filas*columnas*sizeof(int) );	//para resultado paralelo con memoria compartida

	for(int fil=0;fil<filas;fil++){
		for(int col=0;col<columnas;col++){
			A[fil*columnas+col]=rand() % 3;
      		//printf("%d",A[fil*columnas+col]);
			B[fil*columnas+col]=rand() % 3;
      		//printf("%d ",B[fil*columnas+col]);
		}
    	//printf("\n");
	}
  	//printf("\n");printf("\n");printf("\n");
  
  	multiplicar(A,B,C,filas,columnas);
  	vectorAdd(A,B,CK,CKS,n);

  	int entre=0;

  	for(int fil=0;fil<filas;fil++){
		for(int col=0;col<columnas;col++){
			//printf("%d ",CK[fil*columnas+col]);
			if(C[fil*columnas+col]==CK[fil*columnas+col] && C[fil*columnas+col]==CKS[fil*columnas+col]){
				entre++;
			}
		}
		//printf("\n");
	}
	printf("Hay %d coincidencias.",entre);

  	free(A);
  	free(B);
  	free(C);
  	free(CK);
  	free(CKS);

	return 0;
}