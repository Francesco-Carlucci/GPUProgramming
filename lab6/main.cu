#include <iostream>
#include "img.h"

__global__ void bruxelise(unsigned char *input,unsigned char *output){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    double r,g,b;
    r= input[3*tid];
    g= input[3*tid+1];
    b= input[3*tid+2];

    output[tid]=0.21*r+0.71*g+0.07*b;
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
    printf("final idx: %d",idx);
}


int main(int argc, char* argv[]) {
    RGBImage *h_img;
    GrayImage *h_gray;
    unsigned char *d_gray,*h_data,*d_data;
    char* filename="./WP_20150710_22_14_21_Pro.ppm", *dstname="./gray.pgm";
    int h,w, threadnumber=1024,blocknumber;

    h_img=readPPM(filename);

    h=h_img->height;
    w=h_img->width;
    blocknumber=h*w/threadnumber;

    cudaMalloc(&d_data,3*w*h*sizeof(unsigned char));  //sizeof(h_img->data)

    cudaMalloc((void**) &d_gray,w*h*sizeof(unsigned char));

    h_gray= createPGM(w,h);

    cudaMemcpy(d_data,h_img->data,3*w*h*sizeof(unsigned char),cudaMemcpyHostToDevice);

    //h_data=(unsigned char*) malloc(w*h*sizeof(unsigned char));
    //bruxeliseCPU(h_img,h_data);
    //h_gray->data=h_data;

    bruxelise<<<blocknumber,threadnumber>>>(d_data,d_gray);

    cudaMemcpy(h_gray->data,d_gray,w*h*sizeof(unsigned char),cudaMemcpyDeviceToHost);

    writePGM(dstname,h_gray);

    cudaFree(d_data);
    cudaFree(d_gray);

    return 0;
}
