#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>

#define TAM 200
#define blockSize 1024
#define TILE_WIDTH 32

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


__global__ void MatrixMulKernel(int *d_M, int *d_N, int *d_P, int filasA, int columnasA, int columnasB){
	
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// Calculate the coumn index of d_Pelement and d_M
	int Col = blockIdx.x*blockDim.x+threadIdx.x;

	if((Row < filasA) && (Col < columnasB)){
		int Pvalue = 0;
		for (int k=0;k < columnasA;k++){
			Pvalue += d_M[Row*columnasA+k]*d_N[k*columnasB+Col];
		}
		d_P[Row*columnasB+Col] = Pvalue;
	}

}


void vectorAdd(int *A, int *B, int *CK, int *CKS, int filasA, int columnasA,int columnasB){
  int sizeA= filasA*columnasA*sizeof(int);
  int sizeB= columnasA*columnasB*sizeof(int);
	int sizeC= filasA*columnasB*sizeof(int);
  
	int *d_A, *d_B, *d_C, *d_CKS;
	cudaMalloc((void **)&d_A,sizeA);															//reserva memoria en el device
	cudaMalloc((void **)&d_B,sizeB);
	cudaMalloc((void **)&d_C,sizeC);
	cudaMalloc((void **)&d_CKS,sizeC);


	int numBlockX=32;
	while(columnasB%numBlockX!=0){
		numBlockX=numBlockX-1;
	}
	printf("numero de bloques en X -> %d \n",numBlockX);
  
  int numBlockY=32;
	while(filasA%numBlockY!=0){
		numBlockY=numBlockY-1;
	}
	printf("numero de bloques en Y-> %d \n",numBlockY);
  
  
  	clock_t t2;
  	t2 = clock();	//tiempo de asignacion de memoria
	cudaMemcpy( d_A, A, sizeA, cudaMemcpyHostToDevice);										//se copian al device
	cudaMemcpy( d_B, B, sizeB, cudaMemcpyHostToDevice);

	dim3 dimBlock(numBlockX,numBlockY,1); 		//->30 hx 30 hy ->900 hilos ->100 bloques (bloques en X y Y?)
  	dim3 dimGrid(ceil(columnasB/dimBlock.x),ceil(filasA/dimBlock.y),1);	//100 bloques en X y 100 bloques en Y
  	t2 = clock() - t2;

  	clock_t t3;
  	t3 = clock();
	MatrixMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, filasA, columnasA, columnasB);												//ejecuta el kernel ,,n-> numero de hilos por block, max 1024
	cudaMemcpy( CK,d_C, sizeC, cudaMemcpyDeviceToHost);
	t3 = clock() - t3;

  	clock_t t4;
  	t4 = clock();
  	MatrixMulKernelShared<<< dimGrid, dimBlock >>>(d_A, d_B, d_CKS, columnasB);												//ejecuta el kernel ,,n-> numero de hilos por block, max 1024
	cudaMemcpy( CKS,d_CKS, sizeC, cudaMemcpyDeviceToHost);
  	t4 = clock() - t4;

  	printf ("\nTiempo desde la GPU: (%f seconds).\n",((float)(t2+t3))/CLOCKS_PER_SEC);
  	printf ("\nTiempo desde la GPU con memoria compartida: (%f seconds).\n",((float)(t2+t4))/CLOCKS_PER_SEC);

	cudaFree(d_A);																			//libera memoria del dispositivo
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_CKS);

}

void multiplicar(int *A, int *B, int *C, int filasA, int columnasA,int columnasB){
 	clock_t t;
   	t = clock();

    for(int it=0;it<filasA;it++){

	   	for(int fil=0;fil<columnasB;fil++){
			for(int col=0;col<columnasA;col++){
	        	C[it*columnasB+fil] += A[it*columnasA+col]*B[col*columnasB+fil];
		     }
        	//--printf("%d ",C[it*columnasB+fil]);
	   	}	
      	//--printf("\n");
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

  	int n=1024;
  	int filasA=n;
  	int columnasA=n;
  
  	int filasB=n;
  	int columnasB=n;
  
  	if(columnasA==filasB){

		A = (int*)malloc( filasA*columnasA*sizeof(int) );
		B = (int*)malloc( filasB*columnasB*sizeof(int) );
		C = (int*)malloc( filasA*columnasB*sizeof(int) );		//para resultado secuencial
		CK = (int*)malloc( filasA*columnasB*sizeof(int) );	//para resultado paralelo
		CKS = (int*)malloc( filasA*columnasB*sizeof(int) );	//para resultado paralelo con memoria compartida

		for(int fil=0;fil<filasA;fil++){
			for(int col=0;col<columnasA;col++){
				A[fil*columnasA+col]=rand() % 3;
	      		//--printf("%d",A[fil*columnasA+col]);
			}
	    	//--printf("\n");
		}
		//--printf("\n");

	    for(int fil=0;fil<filasB;fil++){
			for(int col=0;col<columnasB;col++){
				B[fil*columnasB+col]=rand() % 3;
	      		//--printf("%d ",B[fil*columnasB+col]);
			}
	    	//--printf("\n");
		}
      //--printf("\n");
	  
	  	multiplicar(A,B,C,filasA,columnasA,columnasB);
	  	vectorAdd(A,B,CK,CKS,filasA,columnasA,columnasB);

      //verifico resultado
	  	int entre=0;
	  	for(int fil=0;fil<filasA;fil++){
			for(int col=0;col<columnasB;col++){
				//--printf("%d ",CK[fil*columnasB+col]);
				if(C[fil*columnasB+col]==CK[fil*columnasB+col] && C[fil*columnasB+col]==CKS[fil*columnasB+col]){
					entre++;
				}
			}
			//--printf("\n");
		}
		printf("Hay %d coincidencias.",entre);

	  	free(A);
	  	free(B);
	  	free(C);
	  	free(CK);
	  	free(CKS);
	}else{
		printf("Dimensiones incorrectas");
	}
	return 0;
}