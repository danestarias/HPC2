#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>

#define TAM 200
#define TILE_SIZE 4

#define MAX_MASK_WIDTH 5
__constant__ int M[MAX_MASK_WIDTH];


__global__ void convolution_1D_basic_kernel_tile(int *N, int *P, int Mask_Width, int Width){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH -1];

	int n = Mask_Width/2;

	int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
	if (threadIdx.x >= blockDim.x - n) {
		N_ds[threadIdx.x - (blockDim.x -n)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
	}

	N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];

	int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
	if (threadIdx.x < n) {
		N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : N[halo_index_right];
	}

	__syncthreads();

	float Pvalue = 0;
	for (int j=0; j < Mask_Width;j++){
		Pvalue += N_ds[threadIdx.x +j]*M[j];
	}
	P[i] = Pvalue;

}

__global__ void convolution_1D_basic_kernel_caching(int *N, int *P, int Mask_Width, int Width){
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	float Pvalue = 0;
	int N_start_point = i- (Mask_Width/2);
	for (int j = 0;j < Mask_Width;j++){
		if(N_start_point + j >= 0 && N_start_point + j < Width){
			Pvalue += N[N_start_point +j]*M[j];
		}
	}
	P[i] = Pvalue;
}


__global__ void convolution_1D_basic_kernel(int *N, int *M, int *P, int Mask_Width, int Width){
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	float Pvalue = 0;
	int N_start_point = i- (Mask_Width/2);
	for (int j = 0;j < Mask_Width;j++){
		if(N_start_point + j >= 0 && N_start_point + j < Width){
			Pvalue += N[N_start_point +j]*M[j];
		}
	}
	P[i] = Pvalue;
}


void callCuda(int *W, int *B, int *Rp, int *Rpc, int *Rpct, int backTam, int winTam){
  	int sizeW= winTam*sizeof(int);
  	int sizeB= backTam*sizeof(int);
	int sizeR= (winTam+backTam)*sizeof(int);
  
	int *d_W, *d_WC, *d_B, *d_R, *d_RC, *d_RCT;
	cudaMalloc((void **)&d_W,sizeW);	//reserva memoria en el device
	cudaMalloc((void **)&d_WC,sizeW);	//reserva memoria en el device

	cudaMalloc((void **)&d_B,sizeB);

	cudaMalloc((void **)&d_R,sizeR);
	cudaMalloc((void **)&d_RC,sizeR);
	cudaMalloc((void **)&d_RCT,sizeR);

	int numBlockX=32;
	while(backTam%numBlockX!=0){
		numBlockX=numBlockX-1;
	}
	printf("numero de bloques en X -> %d \n",numBlockX);
  
  	clock_t t2;
  	t2 = clock();	//tiempo de asignacion de memoria
	cudaMemcpy( d_W, W, sizeW, cudaMemcpyHostToDevice);	//se copian al device

	cudaMemcpy( d_B, B, sizeB, cudaMemcpyHostToDevice);

	dim3 dimBlock(numBlockX,1,1); 		//->3es (bloques en X y Y?)
  	dim3 dimGrid(ceil(backTam/dimBlock.x),1,1);	//100 bloques en X y 100 bloques en Y
  	t2 = clock() - t2;

  	clock_t t3;
  	t3 = clock();
	convolution_1D_basic_kernel<<< dimGrid, dimBlock >>>(d_B, d_W, d_R, winTam, backTam);
	cudaMemcpy( Rp,d_R, sizeR, cudaMemcpyDeviceToHost);
	t3 = clock() - t3;
	
	printf ("\nTiempo desde la GPU : (%f seconds).\n",((float)(t2+t3))/CLOCKS_PER_SEC);

	clock_t t4;
  	t4 = clock();
	cudaMemcpy( d_WC, W, sizeW, cudaMemcpyHostToDevice);	//se copian al device
  	cudaMemcpyToSymbol(M,d_WC,winTam*sizeof(int));
	convolution_1D_basic_kernel_caching<<< dimGrid, dimBlock >>>(d_B, d_RC, winTam, backTam);
	cudaMemcpy( Rpc,d_RC, sizeR, cudaMemcpyDeviceToHost);
	t4 = clock() - t4;

  	printf ("\nTiempo desde la GPU con CACHING: (%f seconds).\n",((float)(t2+t4))/CLOCKS_PER_SEC);

  	clock_t t5;
  	t5 = clock();
	cudaMemcpy( d_WC, W, sizeW, cudaMemcpyHostToDevice);	//se copian al device
  	cudaMemcpyToSymbol(M,d_WC,winTam*sizeof(int));
	convolution_1D_basic_kernel_tile<<< dimGrid, dimBlock >>>(d_B, d_RCT, winTam, backTam);
	cudaMemcpy( Rpct,d_RCT, sizeR, cudaMemcpyDeviceToHost);
	t5 = clock() - t5;

  	printf ("\nTiempo desde la GPU con CACHING: (%f seconds).\n",((float)(t2+t5))/CLOCKS_PER_SEC);

	cudaFree(d_W);			//libera memoria del dispositivo
	cudaFree(d_B);
	cudaFree(d_R);
	cudaFree(d_RC);

}

 
int main(){
	int * W;	//winddow
	int * B;

	int * Rp;
	int * Rpc;
	int * Rpct;

  	int winTam=MAX_MASK_WIDTH;
  	int backTam=TAM;
  
  	if(backTam>winTam){

		W = (int*)malloc( winTam*sizeof(int) );
		B = (int*)malloc( backTam*sizeof(int) );

	    Rp = (int*)malloc( (backTam+winTam)*sizeof(int) );
	    Rpc = (int*)malloc( (backTam+winTam)*sizeof(int) );
	    Rpct = (int*)malloc( (backTam+winTam)*sizeof(int) );

		//printf("W ->");
		for(int i=0;i<winTam;i++){
			W[i]=rand() % 3;
	      	//printf("%d",W[i]);
		}
	    //printf("\nB ->");
		for(int i=0;i<backTam;i++){
			B[i]=rand() % 3;
	      	//printf("%d",B[i]);
		}
	      
	  	callCuda(W, B, Rp, Rpc, Rpct, backTam, winTam);

	  	/*printf("\nAlgoritmo con kernel BASICO = ");
		for(int i=0;i<backTam;i++){
	  		printf(" %d ",Rp[i]);
		}*/

		/*printf("\nAlgoritmo con CACHING = ");
		for(int i=0;i<backTam;i++){
	  		printf(" %d ",Rpc[i]);
		}*/

		/*printf("\nalgoritmo con TILES = ");
		for(int i=0;i<backTam;i++){
	  		printf(" %d ",Rpct[i]);
		}*/

	     //verifico resultado
	  	bool entre=true;
	  	for(int i=0;i<backTam;i++){
	  		if(Rp[i]!=Rpc[i] || Rpc[i]!=Rpct[i]){
	  			entre=false;
	  			break;
	  		}
		}

		if(entre){
			printf("Correcto!");
		}else{
			printf("NONONONO!");
		}

	  	free(W);
	  	free(B);
	  	free(Rp);
	  	free(Rpc);
	  	free(Rpct);
	}else{
		printf("Dimensiones incorrectas");
	}
	return 0;
}