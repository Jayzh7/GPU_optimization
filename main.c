#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image_util.h"
#include "network.h"
#include "mobilenetv2.h"
#include "logging.h"

void preprocess(IMG* img){
    //Subtract mean RGB values, scale with 0.017, and swap RGB->BGR
    for(int y=0;y<img->height;y++)
        for(int x=0;x<img->width;x++){
            float R            = (img->data[0][y][x]-123.680f)*0.017f; //R
            img->data[1][y][x] = (img->data[1][y][x]-116.779f)*0.017f; //G
            img->data[0][y][x] = (img->data[2][y][x]-103.939f)*0.017f; //B
            img->data[2][y][x] = R;
         }
}

int argmax(BLOB* b){
    //find index of channel that is maximum
    float m=b->data[0][0][0];
    int i=0;
    for(int z=1;z<b->d;z++)
        if(b->data[z][0][0] > m ){
            m=b->data[z][0][0];
            i=z;
        }
    return i;
}

int main(int argc, char* argv[]){
    if (argc!=2 || !strcmp(argv[1],"-h") || !strcmp(argv[1],"--help"))
        error("Usage: %s <input.png>\n", argv[0]);

#ifndef SILENT
    //set stdout to line buffered such that piping to "tee" does not delay
    setlinebuf(stdout);
#endif

    //read png into image structure
    info("Loading image %s\n",argv[1]);
    IMG* img = read_png(argv[1]);

    //Do preprocessing of the image
    info("Preprocessing image\n");
    preprocess(img);

    //perform inference
    info("Performing inference\n");
    BLOB* out = network(&mobilenetv2, img);

    //get class index of maximum
    int class_idx=argmax(out);
    if(!(class_idx>=0 && class_idx<=999))
        error("provided class index (%d) is out of bounds!\n", class_idx);

    //print the class index
    printf("Detect class: %d\n", class_idx);

    //print the class label
    switch(class_idx){
        case 281:
            printf("\033[0;33mRobot finds kitteh!\033[0m\n");
        break;
        case 285:
            printf("\033[0;33mRobot finds \033[0;35mEgyptsjun\033[0;33m kitteh - so exotic!\033[0m\n");
        break;
        case 291:
            printf("\033[0;33mRobot finds \033[0;34mKing of Kittehs\033[0;33m- much wow!\033[0m\n");
        break;
        default:
            printf("\033[1;37mIz not kitteh. Itz a \"%s\"\033[0m\n", labels[class_idx]);
        break;
    }

    //cleanup output
    free_blob(out);

    //cleanup image
    destroy_img(img);

    return 0;
}
