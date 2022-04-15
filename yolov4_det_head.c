#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "opencv/highgui.h"
#include "COCO_names.h"

//darknet.h
typedef struct boxabs {
    float left, right, top, bot;
} boxabs;
typedef struct box {
    float x, y, w, h;
} box;
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
    float *uc;  //Gaussian YOLOv3 - tx,ty,tw,th 
    int points; //bit-0 - center, bit-1 - top-left-corner, bit2 - bottom-right-corner
}detection;
int nms_comparator_v3(const void *pa, const void *pb)
{
    //src/box.c
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class]; // there is already: prob = objectness*prob
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}
float overlap(float x1, float w1, float x2, float w2)
{
    //src/box.c
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}
float box_intersection(box a, box b)
{
    //src/box.c
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    //src/box.c
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    //src/box.c
    //return box_intersection(a, b)/box_union(a, b);

    float I = box_intersection(a, b);
    float U = box_union(a, b);
    if (I == 0 || U == 0) {
        return 0;
    }
    return I / U;
}

static int entry_index(int wid, int hei, int class_n, int batch, int location, int entry)
{
    //src/yolo_layer.c

    //int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
    //location n*l.w*l.h entry 4
    int n =   location / (wid*hei);
    int loc = location % (wid*hei);
    return n*wid*hei*(4+class_n+1) + entry*wid*hei + loc;
}


// Converts output of the network to detection boxes
// w,h: image width,height
// netw,neth: network width,height
// relative: 1 (all callers seems to pass TRUE)
void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    // network height (or width)
    int new_w = 0;
    // network height (or width)
    int new_h = 0;
    // Compute scale given image w,h vs network w,h
    // I think this "rotates" the image to match network to input image w/h ratio
    // new_h and new_w are really just network width and height
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    // difference between network width and "rotated" width
    float deltaw = netw - new_w;
    // difference between network height and "rotated" height
    float deltah = neth - new_h;
    // ratio between rotated network width and network width
    float ratiow = (float)new_w / netw;
    // ratio between rotated network width and network width
    float ratioh = (float)new_h / neth;
    for (i = 0; i < n; ++i) {

        box b = dets[i].bbox;
        // x = ( x - (deltaw/2)/netw ) / ratiow;
        //   x - [(1/2 the difference of the network width and rotated width) / (network width)]
        b.x = (b.x - deltaw / 2. / netw) / ratiow;
        b.y = (b.y - deltah / 2. / neth) / ratioh;
        // scale to match rotation of incoming image
        b.w *= 1 / ratiow;
        b.h *= 1 / ratioh;

        // relative seems to always be == 1, I don't think we hit this condition, ever.
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }

        dets[i].bbox = b;
    }
}
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
                            // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
                            // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i) {
            // printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}
boxabs box_c(box a, box b) {
    boxabs ba = { 0 };
    ba.top = fmin(a.y - a.h / 2, b.y - b.h / 2);
    ba.bot = fmax(a.y + a.h / 2, b.y + b.h / 2);
    ba.left = fmin(a.x - a.w / 2, b.x - b.w / 2);
    ba.right = fmax(a.x + a.w / 2, b.x + b.w / 2);
    return ba;
}
float box_diounms(box a, box b, float beta1)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, beta1);
    float diou_term = u;
    // printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
    return iou - diou_term;
}
// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
float box_diou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, 0.6);
    float diou_term = u;
    // printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
    return iou - diou_term;
}

void diounms_sort(detection *dets, int total, int classes, float thresh, int nms_kind, float beta1)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i)
        {
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh && nms_kind == 3)
                {
                    float sum_prob = pow(dets[i].prob[k], 2) + pow(dets[j].prob[k], 2);
                    float alpha_prob = pow(dets[i].prob[k], 2) / sum_prob;
                    float beta_prob = pow(dets[j].prob[k], 2) / sum_prob;
                    dets[j].prob[k] = 0;
                }
                else if (box_diou(a, b) > thresh && nms_kind == 1) {
                    dets[j].prob[k] = 0;
                }
                else {
                    if (box_diounms(a, b, beta1) > thresh && nms_kind == 2) {
                        dets[j].prob[k] = 0;
                    }
                }
            }

            //if ((nms_kind == CORNERS_NMS) && (dets[i].points != (YOLO_CENTER | YOLO_LEFT_TOP | YOLO_RIGHT_BOTTOM)))
            //    dets[i].prob[k] = 0;
        }
    }
}

int yolov4_det(float** onnx_out, char* file_name, int img_sz){
       
    int letter_box = 0; 
    int relative = 1;

    int classes = 80;

    int img_w = img_sz;    //real image
    int img_h = img_sz;    //real image

    int net_w = img_sz;    //net input w
    int net_h = img_sz;    //net input h

    float anchors[18] = 
                        //yolov3
                        // {  10,13,   16, 30,   33, 23,  
                        //    30,61,   62, 45,   59,119,  
                        //   116,90,  156,198,  373,326 };
                        //yolov4
                        { 142,110, 192,243, 459,401,
                           36, 75,  76, 55,  72,146, 
                           12, 16,  19, 36,  40, 28};
    float thresh      = .25f;
    float hier_thresh = .5f;
    float nms         = .45f;
    int   nms_kind    = 1;
    float nms_beta    = .6f;

    int yolo_layer_n = 3;
    
    int yolo_layer_info[9]= 
                            //yolov3
                            // { 3,13,13, 
                            //   3,26,26, 
                            //   3,52,52 };
                            //yolov4
                            { 3,52,52,
                              3,26,26,
                              3,13,13 };
    int** yolo_layer  = (int**)malloc( yolo_layer_n *sizeof(int*));
    float** yolo_data = (float**)malloc(yolo_layer_n * sizeof(float*));
    int data_index = 0;
    printf("===============\n");
    for(int i = 0; i<yolo_layer_n; i++){
        yolo_layer[i] = &yolo_layer_info[i*3];

        yolo_data[i]  = onnx_out[i];
        data_index = yolo_layer[i][0] * (classes + 5) * yolo_layer[i][1] * yolo_layer[i][2];
        printf("%d-yolo layer dim : %dx%dx%dx%d = %d\n",
                i,yolo_layer[i][0], (classes + 5), yolo_layer[i][1], yolo_layer[i][2],
                data_index);
    }
    printf("===============\n");

    int nboxes = 0;

    for (int i = 0; i < yolo_layer_n; ++i) {    
        int count = 0;
        int channel = yolo_layer[i][0];
        int width   = yolo_layer[i][1];
        int height  = yolo_layer[i][2];
        for (int j = 0; j < width*height; ++j){
            for(int n = 0; n < channel; ++n){
                int obj_index  = entry_index(width, height, classes, 0, n*width*height + j, 4);
                if(yolo_data[i][obj_index] > thresh){
                    ++count;
                }
            }
        }
        nboxes += count;
    }
    printf("Debug : box count : %d\n",nboxes);
    if(nboxes != 0){
        detection* dets = (detection*)calloc(nboxes, sizeof(detection));
        // printf("detection_sz:%d\n",sizeof(detection));

        for (int i = 0; i < nboxes; ++i) {
            dets[i].prob = (float*)calloc(classes, sizeof(float));
            // tx,ty,tw,th uncertainty
            // dets[i].uc = (float*)calloc(4, sizeof(float)); // Gaussian_YOLOv3
            // yolov2 region layer
            // if (l.coords > 4) {
            //     dets[i].mask = (float*)xcalloc(l.coords - 4, sizeof(float));
            // }
        }
        
        int count = 0;
        for(int i = 0; i < yolo_layer_n; i++){
            int channel = yolo_layer[i][0];
            int width   = yolo_layer[i][1];
            int height  = yolo_layer[i][2];
            
            float *predictions = yolo_data[i];
            // This snippet below is not necessary
            // Need to comment it in order to batch processing >= 2 images
            //if (l.batch == 2) avg_flipped_yolo(l);    
            for (int ii = 0; ii < width*height; ++ii){
                int row = ii / width;
                int col = ii % width;
                for(int n = 0; n < channel; ++n){
                    int obj_index  = entry_index(width, height, classes, 0, n*width*height + ii, 4);
                    float objectness = predictions[obj_index];
                    //if(objectness <= thresh) continue;    // incorrect behavior for Nan values
                    if (objectness > thresh) {
                        // printf("\n objectness = %f, thresh = %f, i = %d, n = %d \n", objectness, thresh, ii, n);
                        int box_index = entry_index(width, height, classes, 0, n*width*height + ii, 0);
                        int mask_index = (2-i) * 3 + n;
                        dets[count].bbox = get_yolo_box(predictions, anchors, mask_index, box_index, col, row, width, height, net_w, net_h, width*height);
                        dets[count].objectness = objectness;
                        dets[count].classes = classes;
                        for (int j = 0; j < classes; ++j) {
                            int class_index = entry_index(width, height, classes, 0, n*width*height + ii, 4 + 1 + j);
                            float prob = objectness*predictions[class_index];
                            dets[count].prob[j] = (prob > thresh) ? prob : 0;
                        }
                        ++count;
                    }
                    
                }
            }
            correct_yolo_boxes(dets, count, width, height, img_w, img_h, relative, letter_box);
        }
        printf("Debug : fill-boxes\n");
        // for(int i = 0; i < nboxes; i++){
        //     printf("(%.2f)", dets[i].objectness);
        //     for(int j = 0; j<classes; j++){
        //         if(dets[i].prob[j] > thresh)  
        //         printf("[%d:%.2f]",j, dets[i].prob[j]);
        //         // printf("[%.2f]", dets[i].prob[j]);
        //     }
        //     printf("\n");
        // }

        //yolov3
        // do_nms_sort(dets, nboxes, classes, nms);
        //yolov4
        diounms_sort(dets, nboxes, classes, nms, 1, nms_beta);
        printf("Debug : non-maximal-Suppression : result boxes : %d\n",nboxes);


        IplImage *input_img = cvLoadImage(file_name);
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 2);
                          //fontface, Hscale, Vscale, Shear, Thickenss  

        int x = input_img -> width;
        int y = input_img -> height;
        int new_w,   new_h;
        int new_ltx=0, new_lty=0;

        //if crop, display the crop range 
        if(x != y){
            if(x > y){
                new_w = y;
                new_h = y;
                new_ltx = (x-new_w)/2;
            }else if(y > x){
                new_w = x;
                new_h = x;
                new_lty = (y-new_h)/2;
            }
            cvRectangle(input_img, 
                        cvPoint(new_ltx, new_lty), 
                        cvPoint(new_ltx+new_w, new_lty+new_h),
                        CV_RGB(0,0,255));
            char text[100];
            sprintf(text,"<ROI : input image>");
            cvPutText(input_img, text, cvPoint(new_ltx, new_lty+15), &font, 
                                    CV_RGB(0,0,255));
        }else{
            new_w = x;
            new_h = y;
        }
        // printf("img w,h : %d,%d, net w,h : %d,%d, roi w,h : %d,%d\n",
        //         x,y,net_w,net_h,new_w,new_h);
        
        
        for(int i = 0; i < nboxes; i++){
            for(int j = 0; j<classes; j++){
                if(dets[i].prob[j] > thresh){
                    printf("(%.2f)", dets[i].prob[j]);
                    printf("[%-20s(%3d)]",COCO_names[j],j);
                    // printf("box : x,y,w,h(%.3f x %.3f , %.3f , %.3f)",
                    //         dets[i].bbox.x,dets[i].bbox.y,
                    //         dets[i].bbox.w,dets[i].bbox.h);
                    int lt_x = (dets[i].bbox.x - dets[i].bbox.w/2) * new_w + new_ltx;
                    int lt_y = (dets[i].bbox.y - dets[i].bbox.h/2) * new_h + new_lty;
                    int rb_x = (dets[i].bbox.x + dets[i].bbox.w/2) * new_w + new_ltx;
                    int rb_y = (dets[i].bbox.y + dets[i].bbox.h/2) * new_h + new_lty;
                    cvRectangle(input_img, 
                        cvPoint(lt_x, lt_y), 
                        cvPoint(rb_x, rb_y),
                        CV_RGB(255,0,0));
                    char text[100];
                    sprintf(text,"%s(%.2f)",COCO_names[j], dets[i].prob[j]);
                    cvPutText(input_img, text, cvPoint(lt_x,lt_y-10), &font, 
                                               CV_RGB(255,0,0));

                    printf("box : x,y,w,h(%4d, %4d x %4d, %4d)\n",
                            lt_x,lt_y,
                            rb_x,rb_y);
                    
                }
            }
        }
        printf("Debug : drawing done\n");
        cvSaveImage("Predictions.jpg",input_img);
        printf("Debug : saving the predicted image -> Predictions.jpg\n");
        // cvShowImage("Object Detection",input_img);
        // cvWaitKey(0);
    }
    return 0;
}