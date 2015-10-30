#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

#define TILE_SIZE 32
#define MASK_WIDTH 3

__constant__ int M[MASK_WIDTH*MASK_WIDTH];

using namespace cv;



__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

__global__ void Union(unsigned char *Sobel_X, unsigned char *Sobel_Y, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if((row < height) && (col < width)){
      imageOutput[row*width+col]= clamp(sqrtf( Sobel_X[row*width+col]*Sobel_X[row*width+col]+Sobel_Y[row*width+col]*Sobel_Y[row*width+col]) );
    }
}

__global__ void sobelFilterShareMemTest(unsigned char *imageInput, int width, int height,unsigned int maskWidth,unsigned char *imageOutput){
    __shared__ float N_ds[TILE_SIZE + MASK_WIDTH - 1][TILE_SIZE+ MASK_WIDTH - 1];

    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+MASK_WIDTH-1), destX = dest % (TILE_SIZE+MASK_WIDTH-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + MASK_WIDTH - 1), destX = dest % (TILE_SIZE + MASK_WIDTH - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + MASK_WIDTH - 1) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int accum = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = clamp(accum);
    __syncthreads();
}

__global__ void Sobel_kernel_caching(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    int aux=row*width+col;
    int sum=0;
    if((row < height) && (col < width)){
        
        if(( aux-width-2) > 0 ){
            sum += M[0]*imageInput[aux-width-2];
        }
        if((aux-1) > 0){
            sum += M[1]*imageInput[aux-width-1];
        }
        if(aux-width > 0){
            sum += M[2]*imageInput[aux-width];
        }
        //------------------------------------
        if(aux-1 > 0){
            sum += M[3]*imageInput[aux-1];
        }

        sum += M[4]*imageInput[aux];

        if(aux+1 < width*height){
            sum += M[5]*imageInput[aux+1];
        }
        //---------------------------------
        if(aux+width < width*height){
            sum += M[6]*imageInput[aux+width];
        }
        if(aux+width+1 < width*height){
            sum += M[7]*imageInput[aux+width+1];
        }
        if(aux+width+2 < width*height){
            sum += M[8]*imageInput[aux+width+2];
        }
        imageOutput[row*width+col]= clamp(sum);
    }
}

__global__ void Sobel(unsigned char *imageInput,int *mask, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    int aux=row*width+col;
    int sum=0;
    if((row < height) && (col < width)){
        if(( aux-width-2) > 0 ){
            sum += mask[0]*imageInput[aux-width-2];
        }
        if((aux-1) > 0){
            sum += mask[1]*imageInput[aux-width-1];
        }
        if(aux-width > 0){
            sum += mask[2]*imageInput[aux-width];
        }
        //------------------------------------
        if(aux-1 > 0){
            sum += mask[3]*imageInput[aux-1];
        }

        sum += mask[4]*imageInput[aux];

        if(aux+1 < width*height){
            sum += mask[5]*imageInput[aux+1];
        }
        //---------------------------------
        if(aux+width < width*height){
            sum += mask[6]*imageInput[aux+width];
        }
        if(aux+width+1 < width*height){
            sum += mask[7]*imageInput[aux+width+1];
        }
        if(aux+width+2 < width*height){
            sum += mask[8]*imageInput[aux+width+2];
        }
   
        imageOutput[row*width+col]= clamp(sum);
    }
}

__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if((row < height) && (col < width)){
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}

int main(int argc, char **argv){
    //INICIALIZO VARIABLES
    cudaError_t error = cudaSuccess;
    clock_t startCPU, endCPU, startGPUGRAY, endGPUGRAY, startGPU, endGPU, startGPUC, endGPUC, startGPUCS, endGPUCS;
    unsigned char *dataRawImage;                    //aqui se guardara la imagen original
    unsigned char *d_dataRawImage;                  //imagen normal en device
    unsigned char *d_imageOutput;                   //imagen grises en device
    unsigned char *d_SobelOutput_X, *d_SobelOutput_Y, *d_SobelOutput, *h_SobelOutput;   //usados para guardar sobel, en device y host
    unsigned char *d_SobelOutput_XC, *d_SobelOutput_YC, *d_SobelOutputC, *h_SobelOutputC;
    unsigned char *d_SobelOutput_XCS, *d_SobelOutput_YCS, *d_SobelOutputCS, *h_SobelOutputCS;
    Mat image;
    //CARGO IMAGEN
    image = imread("./inputs/img1.jpg", 1);
    if(!image.data){
        printf("No image Data \n");
        return -1;
    }
    //INICIALIZO ATRIBUTOS DE LA IMAGEN
    Size s = image.size();
    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;
    printf("Dimensiones de la imagen: %d x %d \n", width,height);
    dataRawImage = (unsigned char*)malloc(size);        //SEPARO MEMORIA PARA IMAGEN NORMAL EN CPU
    error = cudaMalloc((void**)&d_dataRawImage,size);   //SEPARO MEMORIA PARA IMAGEN NORMAL EN GPU
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_dataRawImage\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_imageOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageOutput\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput_X,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_X\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput_Y,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_Y\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput_XC,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_Y\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput_YC,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_Y\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutputC,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_Y\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput_XCS,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_Y\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput_YCS,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_Y\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutputCS,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_Y\n");
        exit(-1);
    }

    dataRawImage = image.data;      //cargo imagen en memoria host

    //-----------------------CONVERSION A GRISES GPU--------------------------------
    startGPUGRAY = clock();

    error = cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);   //copio imagen a memoria cuda
    if(error != cudaSuccess){
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    }
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);

    img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    cudaDeviceSynchronize();

    endGPUGRAY = clock();

    //INICIALIZO MASCARA
    int * Mask_x = (int*)malloc( 3*3*sizeof(int) ); //separo memoria en host para la mascara en X
    Mask_x[0]=-1;Mask_x[1]=0;Mask_x[2]=1;
    Mask_x[3]=-2;Mask_x[4]=0;Mask_x[5]=2;
    Mask_x[6]=-1;Mask_x[7]=0;Mask_x[8]=1;

    int * Mask_y = (int*)malloc( 3*3*sizeof(int) ); //separo memoria en host para la mascara en X
    Mask_y[0]=-1;Mask_y[1]=-2;Mask_y[2]=-1;
    Mask_y[3]=0;Mask_y[4]=0;Mask_y[5]=0;
    Mask_y[6]=1;Mask_y[7]=2;Mask_y[8]=1;

    int sizeM= 3*3*sizeof(int);
    int *d_M;

    //--------------------------------------SOBEL GPU-------------------------------------------

    startGPU = clock();

    cudaMalloc((void **)&d_M,sizeM);                 //separo memoria para la mascara en GPU
    cudaMemcpy( d_M, Mask_x, sizeM, cudaMemcpyHostToDevice);

    //aplico filtro en x
    Sobel<<<dimGrid,dimBlock>>>(d_imageOutput,d_M,width,height,d_SobelOutput_X);
    //cudaDeviceSynchronize();
    cudaMemcpy( d_M, Mask_y, sizeM, cudaMemcpyHostToDevice);//cambio mascara
    //aplico filtro en y
    Sobel<<<dimGrid,dimBlock>>>(d_imageOutput,d_M,width,height,d_SobelOutput_Y);
    cudaDeviceSynchronize();

    // sobel_X   U   sobel_Y
    Union<<<dimGrid,dimBlock>>>(d_SobelOutput_X,d_SobelOutput_Y,width,height,d_SobelOutput);
    cudaDeviceSynchronize();
    h_SobelOutput = (unsigned char *)malloc(sizeGray);

    cudaMemcpy(h_SobelOutput,d_SobelOutput,sizeGray,cudaMemcpyDeviceToHost);

    endGPU = clock();

    Mat sobel_image;
    sobel_image.create(height,width,CV_8UC1);
    sobel_image.data = h_SobelOutput;   

    //--------------------------------SOBEL CONSTANT MEMORY------------------------------------
    startGPUC = clock();

    cudaMemcpy( d_M, Mask_x, sizeM, cudaMemcpyHostToDevice);//cambio mascara a x
    cudaMemcpyToSymbol(M,d_M,sizeM);                        //copio a memoria constante
    //aplico filtro en x
    Sobel_kernel_caching<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,d_SobelOutput_XC);

    cudaMemcpy( d_M, Mask_y, sizeM, cudaMemcpyHostToDevice);//cambio mascara
    cudaMemcpyToSymbol(M,d_M,sizeM); 
    //aplico filtro en y
    Sobel_kernel_caching<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,d_SobelOutput_YC);
    cudaDeviceSynchronize();

    // sobel_X   U   sobel_Y
    Union<<<dimGrid,dimBlock>>>(d_SobelOutput_XC,d_SobelOutput_YC,width,height,d_SobelOutputC);
    cudaDeviceSynchronize();
    h_SobelOutputC = (unsigned char *)malloc(sizeGray);
    cudaMemcpy(h_SobelOutputC,d_SobelOutputC,sizeGray,cudaMemcpyDeviceToHost);

    endGPUC = clock();

    Mat sobel_imageC;
    sobel_imageC.create(height,width,CV_8UC1);
    sobel_imageC.data = h_SobelOutputC; 

    //--------------------------------SOBEL SHARED MEMORY------------------------------------
    startGPUCS = clock();

    cudaMemcpy( d_M, Mask_x, sizeM, cudaMemcpyHostToDevice);//cambio mascara a x
    cudaMemcpyToSymbol(M,d_M,sizeM);                        //copio a memoria constante
    //aplico filtro en x
    sobelFilterShareMemTest<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,MASK_WIDTH,d_SobelOutput_XCS);

    cudaMemcpy( d_M, Mask_y, sizeM, cudaMemcpyHostToDevice);//cambio mascara
    cudaMemcpyToSymbol(M,d_M,sizeM); 
    //aplico filtro en y
    sobelFilterShareMemTest<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,MASK_WIDTH,d_SobelOutput_YCS);
    
    cudaDeviceSynchronize();
    // sobel_X   U   sobel_Y
    Union<<<dimGrid,dimBlock>>>(d_SobelOutput_XCS,d_SobelOutput_YCS,width,height,d_SobelOutputCS);
    cudaDeviceSynchronize();
    h_SobelOutputCS = (unsigned char *)malloc(sizeGray);
    cudaMemcpy(h_SobelOutputCS,d_SobelOutputCS,sizeGray,cudaMemcpyDeviceToHost);

    endGPUCS = clock();

    Mat sobel_imageCS;
    sobel_imageCS.create(height,width,CV_8UC1);
    sobel_imageCS.data = h_SobelOutputCS; 

    //------------------------------ALGORITMO DE SOBEL EN CPU----------------------------------
    startCPU = clock();
    Mat grad;
    Mat gray_image_opencv;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);    //convierto imagen en escala de grises
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    /// Gradient X
    Sobel( gray_image_opencv, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    /// Gradient Y
    Sobel( gray_image_opencv, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    endCPU = clock();

    //GUARDO IMAGEN
    imwrite("./outputs/1088279598.png",grad);
    //imwrite("./outputs/1088279598.png",sobel_image);//grad -- sobel_image
    //imwrite("./outputs/1088279598.png",sobel_imageC);
    //imwrite("./outputs/1088279598.png",sobel_imageCS);

    //CALCULO E IMPRIMO RESULTADOS
    double timeGray= ((double) (endGPUGRAY-startGPUGRAY)) / CLOCKS_PER_SEC;

    double timeGPU = ( ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC ) + timeGray;
    double timeGPUC =( ((double) (endGPUC - startGPUC)) / CLOCKS_PER_SEC)+ timeGray;
    double timeGPUCS =( ((double) (endGPUCS - startGPUCS)) / CLOCKS_PER_SEC)+ timeGray;
    double timeCPU = ((double) (endCPU- startCPU)) / CLOCKS_PER_SEC;

    printf("Tiempo Algoritmo OpenCV: %.10f\n",timeCPU);
    printf("Tiempo del algoritmo en paralelo: %.10f\n",timeGPU);
    printf("Tiempo del algoritmo en paralelo memoria con constante : %.10f\n",timeGPUC);
    printf("Tiempo del algoritmo en paralelo memoria con compartida: %.10f\n \n",timeGPUCS);

    printf("Aceleración del algoritmo en paralelo: %.10fX\n",timeCPU/timeGPU);
    printf("Aceleración del algoritmo en paralelo con memoria constante : %.10fX\n",timeCPU/timeGPUC);
    printf("Aceleración del algoritmo en paralelo con memoria compartida: %.10fX\n",timeCPU/timeGPUCS);

    //printf("ESCALA GRISES La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);

    //LIBERO MEMORIA EN GPU
    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
    cudaFree(d_SobelOutput_X);
    cudaFree(d_SobelOutput_Y);
    cudaFree(d_SobelOutput_X);
    cudaFree(d_SobelOutput_XC);
    cudaFree(d_SobelOutput_YC);
    cudaFree(d_SobelOutput_XCS);
    cudaFree(d_SobelOutput_YCS);


    return 0;

}