#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

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
    //int aux;
    if((row < height) && (col < width)){
      //aux= sqrtf( Sobel_X[row*width+col]*Sobel_X[row*width+col]+Sobel_Y[row*width+col]*Sobel_Y[row*width+col]);
      //imageOutput[row*width+col]= clamp(Sobel_X[row*width+col]+Sobel_Y[row*width+col]);
      imageOutput[row*width+col]= clamp(sqrtf( Sobel_X[row*width+col]*Sobel_X[row*width+col]+Sobel_Y[row*width+col]*Sobel_Y[row*width+col]) );
      /*if(aux >= 0 || aux <= 255){
        imageOutput[row*width+col]=aux;
      }else{
        if(aux>255){
          imageOutput[row*width+col]=255;
        }else{
          imageOutput[row*width+col]=0;
        }
      }*/
      //imageOutput[row*width+col]= ceil(sqrtf( Sobel_X[row*width+col]*Sobel_X[row*width+col]+Sobel_Y[row*width+col]*Sobel_Y[row*width+col]));
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
    clock_t start, end, startGPU, endGPU;
    double cpu_time_used, gpu_time_used;
    unsigned char *dataRawImage;                    //aqui se guardara la imagen original
    unsigned char *d_dataRawImage;                  //imagen normal en device
    unsigned char *d_imageOutput;                   //imagen grises en device
    unsigned char *d_dataRawImageGray;              //imagen grises en device para sobel
    unsigned char *h_imageOutput;                   //imagen grises en host
    unsigned char *d_SobelOutput_X, *d_SobelOutput_Y, *d_SobelOutput, *h_SobelOutput_X, *h_SobelOutput_Y, *h_SobelOutput;   //usados para guardar sobel, en device y host
    
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
        printf("\nsizeGray-> %d",width);
    
    dataRawImage = (unsigned char*)malloc(size);        //SEPARO MEMORIA PARA IMAGEN NORMAL EN CPU
    error = cudaMalloc((void**)&d_dataRawImage,size);   //SEPARO MEMORIA PARA IMAGEN NORMAL EN GPU
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_dataRawImage\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_dataRawImageGray,size);//NO FALTA,,SEPARO MEMORIA CUDA PARA SUBIR IMA GRIS A KERNEL
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_dataRawImageGray\n");
        exit(-1);
    }


    h_imageOutput = (unsigned char *)malloc(sizeGray);
    error = cudaMalloc((void**)&d_imageOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageOutput\n");
        exit(-1);
    }

    h_SobelOutput_X = (unsigned char *)malloc(sizeGray);
    error = cudaMalloc((void**)&d_SobelOutput_X,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_X\n");
        exit(-1);
    }

    h_SobelOutput_Y = (unsigned char *)malloc(sizeGray);
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


    dataRawImage = image.data;      //cargo imagen en memoria host



    //-----------------------CONVERSION A GRISES GPU--------------------------------
    startGPU = clock();

    error = cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);//paso imagen a memoria cuda
    if(error != cudaSuccess){
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    }
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);

    img2gray<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    cudaMemcpy(h_imageOutput,d_imageOutput,sizeGray,cudaMemcpyDeviceToHost);
    endGPU = clock();

    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;        //Guardo imgen en grises hecha por la GPU


    //--------------------------------------SOBEL GPU-------------------------------------------
    Size sG = gray_image.size();                    //tomo imagen en escala de grises

    int widthG = sG.width;
    int heightG = sG.height;
    //int sizeS = sizeof(unsigned char)*widthG*heightG*gray_image.channels();
    int sizeSobel = sizeof(unsigned char)*widthG*heightG;
  
    printf("\nsizeSobel-> %d",widthG);

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
    cudaMalloc((void **)&d_M,sizeM);                 //separo memoria para la mascara en GPU
    cudaMemcpy( d_M, Mask_x, sizeM, cudaMemcpyHostToDevice);


    /*error = cudaMemcpy(d_dataRawImageGray,h_imageOutput,sizeS, cudaMemcpyHostToDevice); //genfun
    if(error != cudaSuccess){
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    }*/
  
    int blockSize2 = 32;
    dim3 dimBlock2(blockSize2,blockSize2,1);
    dim3 dimGrid2(ceil(widthG/float(blockSize2)),ceil(heightG/float(blockSize2)),1);

    //aplico filtro en x
    Sobel<<<dimGrid2,dimBlock2>>>(d_imageOutput,d_M,widthG,heightG,d_SobelOutput_X);//d_dataRawImageGray
    cudaDeviceSynchronize();
    cudaMemcpy(h_SobelOutput_X,d_SobelOutput_X,sizeSobel,cudaMemcpyDeviceToHost);
    //endGPU = clock();

    //aplico filtro en y
    cudaMemcpy( d_M, Mask_y, sizeM, cudaMemcpyHostToDevice);//cambio mascara

    Sobel<<<dimGrid2,dimBlock2>>>(d_imageOutput,d_M,widthG,heightG,d_SobelOutput_Y);//d_dataRawImageGray
    cudaDeviceSynchronize();
    cudaMemcpy(h_SobelOutput_Y,d_SobelOutput_Y,sizeSobel,cudaMemcpyDeviceToHost);


    //union sobel_X y sobel_Y
    Union<<<dimGrid2,dimBlock2>>>(d_SobelOutput_X,d_SobelOutput_Y,width,height,d_SobelOutput);//d_dataRawImageGray
    cudaDeviceSynchronize();
    h_SobelOutput = (unsigned char *)malloc(sizeSobel);

    cudaMemcpy(h_SobelOutput,d_SobelOutput,sizeSobel,cudaMemcpyDeviceToHost);

    endGPU = clock();

    Mat sobel_image;
    sobel_image.create(heightG,widthG,CV_8UC1);
    sobel_image.data = h_SobelOutput;   




    //------------------------------ALGORITMO DE SOBEL EN CPU----------------------------------
    start = clock();
    Mat grad;
    Mat gray_image_opencv;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);    //convierto imagen en escala de grises
      /// Generate grad_x and grad_y
      Mat grad_x, grad_y;
      Mat abs_grad_x, abs_grad_y;

      /// Gradient X
      //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
      Sobel( gray_image_opencv, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
      convertScaleAbs( grad_x, abs_grad_x );
            /// Gradient Y
      //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
      Sobel( gray_image_opencv, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
      convertScaleAbs( grad_y, abs_grad_y );
      /// Total Gradient (approximate)
      addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    end = clock();

    //GUARDO IMAGEN
    imwrite("./outputs/1088279598.png",sobel_image);//grad -- sobel_image

    //CALCULO E IMPRIMO RESULTADOS
    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("ESCALA GRISES Tiempo Algoritmo Paralelo: %.10f\n",gpu_time_used);
    cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf("ESCALA GRISES Tiempo Algoritmo OpenCV: %.10f\n",cpu_time_used);
    printf("ESCALA GRISES La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);

    //LIBERO MEMORIA EN GPU
    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
    return 0;
}