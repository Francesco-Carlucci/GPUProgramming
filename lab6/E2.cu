//
// Created by Francesco on 22/11/2022.
//
#include <iostream>
#include "img.h"

__global__ void gaussBlur3x3(unsigned char *input,unsigned char *output,int w, int h){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int rBlur=0,gBlur=0,bBlur=0,i,j,x,y;
    int kernel[3][3]={{1,2,1},{2,4,2},{1,2,1}};

    i=tid/w;
    j=tid%w;

    if(i<h&&j<w) {
        //printf("%d i:%d j:%d h:%d w:%d \n",tid,i,j,h,w);
        for (x = -1; x < 2; x++) {
            if (i + x >= 0 && i + x < h) {
                for (y = -1; y < 2; y++) {
                    if (j + y >= 0 && j + y < w) {
                        //printf("zippu %d %d %d %d %d\n",tid,i,j,x,y);
                        rBlur += input[3 * (tid + w * x + y)] * kernel[x+1][y+1];
                        //printf("idx:%d %d %d",3 * (tid + w * x + y),rBlur,kernel[x+1][y+1]);

                        gBlur += input[3 * (tid + w * x + y) + 1] * kernel[x+1][y+1];
                        //printf("idx:%d %d",3 * (tid + w * x + y)+1,gBlur);

                        bBlur += input[3 * (tid + w * x + y) + 2] * kernel[x+1][y+1];
                        //printf("idx:%d %d\n",3 * (tid + w * x + y)+2,bBlur);
                    }
                }
            }
        }

    //printf("%d %d %d\n",rBlur,gBlur,bBlur);

        output[3 * tid] = rBlur/16;
        output[3 * tid + 1] = gBlur/16;
        output[3 * tid + 2] = bBlur/16;
    }
}

void gaussBlurCPU(unsigned char *input,unsigned char *output,int w, int h, int blockNumber, int threadNumber){
    //unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int rBlur=0,gBlur=0,bBlur=0,i,j,x,y;
    int kernel[3][3]={{1,2,1},{2,4,2},{1,2,1}};

    /*
    for(i=0;i<3;i++){
        for(j=0;j<3;j++){
            printf("%d",kernel[i][j]);
        }
        printf("\n");
    }
     */

    for(int tid=0;tid<blockNumber*threadNumber;tid++) {
        i = tid / w;
        j = tid % w;
        rBlur=0;
        gBlur=0;
        bBlur=0;

        for (x = -1; x < 2; x++) {
            if (i + x >= 0 && i + x < h) {
                for (y = -1; y < 2; y++) {
                    if (j + y >= 0 && j + y < w) {
                        //printf("zippu %d %d %d %d %d\n",tid,i,j,x,y);
                        rBlur += input[3 * (tid + w * x + y)] * kernel[x+1][y+1];
                        //printf("idx:%d %d %d",3 * (tid + w * x + y),rBlur,kernel[x+1][y+1]);

                        gBlur += input[3 * (tid + w * x + y) + 1] * kernel[x+1][y+1];
                        //printf("idx:%d %d",3 * (tid + w * x + y)+1,gBlur);

                        bBlur += input[3 * (tid + w * x + y) + 2] * kernel[x+1][y+1];
                        //printf("idx:%d %d\n",3 * (tid + w * x + y)+2,bBlur);
                    }
                }
            }
        }
        /*
        for(int i=0;i<9;i++){
            index=3 * (tid + i - 4);
               if(index>0) { // &&index+2<3*w*h
                   rBlur += input[3 * (tid + i - 4)] * kernel[i];
                   gBlur += input[3 * (tid + i - 4) + 1] * kernel[i];
                   bBlur += input[3 * (tid + i - 4) + 2] * kernel[i];
               }
        }
        */
        //printf("%d %d %d\n",rBlur,gBlur,bBlur);
        //if (tid < 3 * w * h) {
            output[3 * tid] = rBlur / 16;
            output[3 * tid + 1] = gBlur / 16;
            output[3 * tid + 2] = bBlur / 16;
        //}
    }
}

void bruxeliseCPU(RGBImage *input,unsigned char *output){
    int idx,r,g,b;

    for(int i=0;i<input->height;i++){
        for(int j=0; j<input->width;j++){
            idx = i * input->width + j;
            //here channels is 3
            r = input->data[3 * idx];
            g = input->data[3 * idx + 1];
            b = input->data[3 * idx + 2];
            output[idx] = (0.21 * r + 0.71 * g + 0.07 * b);
        }
    }
}

int main(int argc, char* argv[]) {
    RGBImage *h_img,*h_res;
    unsigned char *d_res,*d_data,*h_data;
    char* filename="./sample_640_426.ppm", *dstname="./blurred.ppm";
    int h,w, threadnumber=1024,blocknumber=267*3;

    h_img=readPPM(filename);

    h=h_img->height;
    w=h_img->width;
    //blocknumber=h*w/threadnumber;

    h_res= createPPM(w,h);

    cudaMalloc(&d_data,3*w*h*sizeof(unsigned char));  //sizeof(h_img->data)

    cudaMalloc((void**) &d_res,3*w*h*sizeof(unsigned char));

    cudaMemcpy(d_data,h_img->data,3*w*h*sizeof(unsigned char),cudaMemcpyHostToDevice);
     /*
    h_data=(unsigned char*) malloc(3*w*h*sizeof(unsigned char));
    gaussBlurCPU(h_img->data,h_data,w,h,blocknumber,threadnumber);
    h_res->data=h_data;
      */
    gaussBlur3x3<<<blocknumber,threadnumber>>>(d_data,d_res,w,h);

    cudaMemcpy(h_res->data,d_res,3*w*h*sizeof(unsigned char),cudaMemcpyDeviceToHost);

    writePPM(dstname,h_res);

    //cudaFree(d_data);
    //cudaFree(d_res);

    return 0;
}


