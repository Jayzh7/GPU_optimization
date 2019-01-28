#include "convolution.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "logging.h"
/*
The maximum number of threads in the block is limited to 1024. This is the product of whatever your threadblock dimensions are (x*y*z). For example (32,32,1) creates a block of 1024 threads. (33,32,1) is not legal, since 33*32*1 > 1024.
grid
x: 2^31 - 1
y,z: 65535
thread
x,y: 1024
z: 64

960
1000
1280
12544
*/
__global__ void gpu_conv_1(float * img, float * weight, float * out, int i_w, int i_h, int w_w, int w_h, int o_w, int o_h, int i_d, int g, int o, int i, int Kx, int Ky, int Sx, int Sy, int group) {
    
    unsigned int m = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
   
    if (Kx == 1) {
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy)*i_w + n*Sx] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w];
    } else {
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy)*i_w + n*Sx] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w];
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy)*i_w + n*Sx+1] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 1];
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy)*i_w + n*Sx+2] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 2];
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy+1)*i_w + n*Sx] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + Kx];
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy+1)*i_w + n*Sx+1] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + Kx+1];
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy+1)*i_w + n*Sx+2] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + Kx+2];
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy+2)*i_w + n*Sx] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 2*Kx];
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy+2)*i_w + n*Sx+1] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 2*Kx+1];
        out[o*o_w*o_h + m*o_w + n] +=
            img[i*i_h*i_w + (m*Sy+2)*i_w + n*Sx+2] * 
            weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 2*Kx+2];
    }
}
    
__global__ void gpu_conv_2(float * img, float * weight, float * out, int i_w, int i_h, int w_w, int w_h, int o_w, int o_h, int Kx, int Ky, int Sx, int Sy, int group, int o_d, int i_d) {
    unsigned int g = blockIdx.y;
    unsigned int o = g*(o_d/group)+blockIdx.x;

    int m = threadIdx.y;
    int n = threadIdx.x;

    for(int i=g*(i_d/group);i<(g+1)*(i_d/group);i++) 
        if (Kx == 1) {
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy)*i_w + n*Sx] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w];
        } else {
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy)*i_w + n*Sx] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w];
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy)*i_w + n*Sx+1] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 1];
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy)*i_w + n*Sx+2] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 2];
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy+1)*i_w + n*Sx] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + Kx];
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy+1)*i_w + n*Sx+1] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + Kx+1];
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy+1)*i_w + n*Sx+2] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + Kx+2];
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy+2)*i_w + n*Sx] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 2*Kx];
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy+2)*i_w + n*Sx+1] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 2*Kx+1];
            out[o*o_w*o_h + m*o_w + n] +=
                img[i*i_h*i_w + (m*Sy+2)*i_w + n*Sx+2] * 
                weight[o*w_w*w_h + (i-(g*(i_d/group)))*w_w + 2*Kx+2];
        }
        
}
//add padding to blob
BLOB* pad(BLOB* in, int pad){

    //create output blob
    BLOB* out = blob_calloc(in->d, in->h+2*pad, in->w+pad*2);

    //copy non-padded input into output blob
    for(int z=0;z<in->d;z++)
       for(int y=0;y<in->h;y++)
          for(int x=0;x<in->w;x++)
              blob_data(out,z,y+pad,x+pad)= blob_data(in,z,y,x);

    //return pointer to padded blob
    return out;
}


BLOB* load_weights(BLOB* b, conv_param_t* p){

    //open weights file for reading
    FILE* fp = fopen(p->weights, "rb");
    if(fp==NULL)
        error("could not open file %s for reading\n",p->weights);

    //for fully connected layers the kernel size is equal to the input size
    int Ky=(p->fc)?b->h:p->Ky;
    int Kx=(p->fc)?b->w:p->Kx;

    //allocate 3D blob, and emulate 4D in KxKy later
    BLOB* w=blob_alloc(p->num_out, b->d/p->group, Ky*Kx);

    //fill 4D weight structure
    for(int g=0;g<p->group;g++)
        for(int o=g*(p->num_out/p->group);o<(g+1)*(p->num_out/p->group);o++)
            for(int i=g*(b->d/p->group);i<(g+1)*(b->d/p->group);i++)
                //note: each output map has only  b->d/p->group input maps. Hence the absolute index of i is subtracted when storing in w!
                if((int)fread( &(blob_data(w,o,i-g*(b->d/p->group),0)),sizeof(float),Ky*Kx, fp)!=Ky*Kx)
                    error("loading weights from file %s\n", p->weights);

    //close file
    fclose(fp);

    //return weight blob
    return w;
}

float* load_1d(const char* fname, size_t num){

    //open file for reading
    FILE* fp = fopen(fname, "rb");
    if(fp==NULL)
        error("could not open file %s for reading\n",fname);

    //read in array
    float* arr= (float*) malloc(sizeof(float)*num);
    if(fread(arr,sizeof(float), num, fp)!=num)
        error("loading data from file %s\n", fname);

    //close file
    fclose(fp);

    return arr;
}

//convolution, NOTE: destructive of BLOB* in. duplicate if further required!
BLOB* convolution(BLOB* input, conv_param_t* p){
    //use local pointer
    BLOB* in = input;
    BLOB* out;
    static bool output_reuse = false;

    //padding of input if required
    if(p->pad!=0)
        in = pad(in, p->pad);

    //if fully connected, the kernel size is set to the image size
    int Ky=(p->fc)?in->h:p->Ky;
    int Kx=(p->fc)?in->w:p->Kx;

    //create blob to hold output
    int height=(int)floor(((float)in->h - (float)Ky)/(float)p->Sy)+1;
    int width =(int)floor(((float)in->w - (float)Kx)/(float)p->Sx)+1;
    

    //load bias if required
    if(p->bias==NULL){
        //zero init
        out = blob_calloc(p->num_out, height, width);
    }else{
        //not required to calloc
        out = blob_alloc(p->num_out, height, width);

        //load bias values from file
        float* bias =load_1d(p->bias, p->num_out);

        //set bias or init with zeroes
        for(int o=0;o<out->d;o++)
            for(int m=0;m<out->h;m++)
                for(int n=0;n<out->w;n++)
                    blob_data(out,o,m,n)=bias[o];

        //cleanup bias
        free(bias);
    }

    //load weights
    BLOB* w = load_weights(in, p);

    float *in_gpu, *w_gpu;
    static float *out_gpu;
    dim3 block, grid;

    if (!output_reuse){
        blob2gpu(in_gpu, in);
    }
    else{
        in_gpu = out_gpu;
    }
    
    blob2gpu(w_gpu, w);
    // Allocs memory for the output of the conv in the ouput
    out_gpu = 
    cudaCheckError(cudaMalloc(&out_gpu, blob_bytes(out)));
    blob2gpu(out_gpu, out);

    if(out->w*out->h > 1023){
        grid = dim3(4, 4, 1);
        block = dim3(out->w/4, out->h/4);
                
        int o, g, i;
        //perform convolution
        for( g=0;g<p->group;g++) {
            for( o=g*(out->d/p->group);o<(g+1)*(out->d/p->group);o++) {
                for( i=g*(in->d/p->group);i<(g+1)*(in->d/p->group);i++) {
                    gpu_conv_1 <<<grid, block>>> (in_gpu, w_gpu, out_gpu, in->w , in->h, w->w , w->h, out->w, out->h, in->d, g, o, i, Kx, Ky, p->Sx, p->Sy, p->group);
                }
            }
        }
    }
    else{
        block = dim3(out->w, out->h);
        grid = dim3(out->d/p->group, p->group);
        // for( g=0;g<p->group;g++) {/
        gpu_conv_2 <<< grid, block >>> (in_gpu, w_gpu, out_gpu, in->w , in->h, w->w , w->h, out->w, out->h, Kx, Ky, p->Sx, p->Sy, p->group, out->d, in->d);
        // }
    }

    gpu2blob(out, out_gpu);

    if (!output_reuse){
        cudaCheckError(cudaFree(in_gpu));
    }
    else{
        output_reuse = false;
    }

    cudaCheckError(cudaFree(w_gpu));
    output_reuse = true;
    // for(int g=0;g<p->group;g++)
    //     for(int o=g*(out->d/p->group);o<(g+1)*(out->d/p->group);o++)
    //         for(int i=g*(in->d/p->group);i<(g+1)*(in->d/p->group);i++)
    //             for(int m=0;m<out->h;m++)
    //                 for(int n=0;n<out->w;n++)
    //                     for(int k=0;k<Ky;k++)
    //                         for(int l=0;l<Kx;l++)
    //                             //note: absolute starting i is subtracted for the weights, see load_weights function for more info
    //                             blob_data(out,o,m,n)+=blob_data(in, i, m*p->Sy+k, n*p->Sx+l) * blob_data(w, o, i-(g*(in->d/p->group)), k*Kx + l);

    //free weights
    blob_free(w);

    //done with padded blob, free
    if(p->pad!=0)
        blob_free(in);

    //perform batchnorm if needed
    if(p->bn_mean!=NULL){


        //load batchnorm mean and variance
        float* mean = load_1d(p->bn_mean, out->d);
        float* var  = load_1d(p->bn_var, out->d);

        //batchnorm
        for(int o=0;o<out->d;o++)
            for(int m=0;m<out->h;m++)
                for(int n=0;n<out->w;n++)
                    blob_data(out,o,m,n)= (blob_data(out,o,m,n) - mean[o])/sqrtf(var[o]+p->bn_eps);

        //free mean and variance
        free(mean);
        free(var);
    }

    //perform scale if needed
    if(p->scale!=NULL){
        //load scale parameters
        float* scale = load_1d(p->scale, out->d);
        float* scale_bias = load_1d(p->scale_bias, out->d);

        //scale
        for(int o=0;o<out->d;o++)
            for(int m=0;m<out->h;m++)
                for(int n=0;n<out->w;n++)
                    blob_data(out,o,m,n) = blob_data(out,o,m,n)*scale[o] + scale_bias[o];

        //free parameters
        free(scale);
        free(scale_bias);
    }

    //perform relu
    if(p->relu==true)
        for(int i=0;i<blob_size(out); i++)
            out->data[i] =  fmax(0.0f, out->data[i]);

    //return output
    return out;
}
