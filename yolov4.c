// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <math.h>
#include "onnxruntime_c_api.h"

#include "opencv/highgui.h"
#include "yolov4_det_head.h"

const OrtApi* g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                             \
    do {                                                       \
        OrtStatus* onnx_status = (expr);                         \
        if (onnx_status != NULL) {                               \
            const char* msg = g_ort->GetErrorMessage(onnx_status); \
            fprintf(stderr, "%s\n", msg);                          \
            g_ort->ReleaseStatus(onnx_status);                     \
            abort();                                               \
        }                                                        \
    } while (0);

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
static void hwc_to_chw(unsigned char* input, size_t h, size_t w, float** output, size_t* output_count) {
    size_t stride = h * w;
    *output_count = stride * 3;
    float* output_data = (float*)malloc(*output_count * sizeof(float));
    for (size_t i = 0; i != stride; ++i) {
        for (size_t c = 0; c != 3; ++c) {
            output_data[c * stride + i] = input[i * 3 + c];
        }
    }
    *output = output_data;
}

/**
 * convert input from HWC Int8(char) to Float
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
static void i8_to_f32(unsigned char* input, size_t h, size_t w, float** output, size_t* output_count) {
    size_t stride = h * w;
    *output_count = stride * 3;
    float* output_data = (float*)malloc(*output_count * sizeof(float));
    for (size_t i = 0; i < *output_count; ++i) {
        output_data[i] = (float)input[i];
    }
    *output = output_data;
}
/**
 * convert input from  anchors(3)*boxes*H*W  to  H*W*anchors*Boxes
 * \param input A yolo's raw matrix. 
 * \param A Row dims
 * \param B col dims
 */
void yolov4_transpose(float* input, int A, int B){
    float* temp_mat = (float*)malloc(A*B*sizeof(float));
    for(int i = 0; i<A*B; i++){
        temp_mat[i] = input[i];
    }
    for(int h = 0; h < A; h++){
        for(int w = 0; w < B; w++){
            input[w*A + h] = temp_mat[h*B + w];
        }
    }
    free(temp_mat);
}

/**
 * \param out should be freed by caller after use
 * \param output_count Array length of the `out` param
 */
static int read_img_file(const char* input_file,int model_in_sz, size_t* height, size_t* width, float** out, size_t* output_count) {
    IplImage *input_img;
    IplImage *readimg;
    
    int one_map_sz = model_in_sz * model_in_sz;
    int channel_sz = 3;

    unsigned char *imgs;

    input_img = cvLoadImage(input_file);
    readimg = cvCreateImage(cvSize(model_in_sz,model_in_sz),IPL_DEPTH_8U,3);
    int x = input_img -> width;
    int y = input_img -> height;
    if(x > y){
        int new_x = (x-y)/2;
        cvSetImageROI(input_img,cvRect(new_x,0,x-new_x*2,y));
    }else if(y > x){
        int new_y = (y-x)/2;
        cvSetImageROI(input_img,cvRect(0,new_y,x,y-new_y));
    }
    cvResize(input_img, readimg, 0);
    unsigned char* buffer = (unsigned char*)readimg->imageData;
    
    *output_count = one_map_sz * channel_sz;
    // hwc_to_chw(buffer, model_in_sz, model_in_sz, out, output_count);
    i8_to_f32(buffer, model_in_sz, model_in_sz, out, output_count);

    for(int hwc = 0; hwc < one_map_sz*channel_sz; hwc++){
        out[0][hwc] = out[0][hwc] / 255.f;
    }

    // free(buffer);

    *width = model_in_sz;
    *height = model_in_sz;
    return 0;
}

static void usage() { printf("usage: <model_path> <input_file> [cpu|cuda] \n"); }

int run_inference(OrtSession* session, const ORTCHAR_T* input_file, int img_sz) {
    size_t input_height;
    size_t input_width;
    float* model_input;
    size_t model_input_ele_count;

    const char* input_file_p = input_file;

    if (read_img_file(input_file_p, img_sz, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
        return -1;
    }

    OrtMemoryInfo* memory_info;
    ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    
    //Ort 입력/출력 노드 찾기(capi)
    size_t in_count,out_count;
    OrtAllocator* allocator;
    ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &in_count));
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &out_count));

    char** input_names = (char**)malloc(in_count*sizeof(char*));
    char** output_names = (char**)malloc(out_count*sizeof(char*));

    for(int i = 0; i < in_count; i++){
        char* name;
        ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &name));
        input_names[0] = name;
    }
    for(int i = 0; i < out_count; i++){
        char* name;
        ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &name));
        output_names[i] = name;
    }
    printf("===============\n");
    //print i/o names
    printf("in_count : %d\n",(int)in_count);
    printf("out_count : %d\n",(int)out_count);
    for(int i = 0; i<in_count; i++){  
        printf("in_names : ");
        printf("%s ",input_names[i]);
    }
    printf("\n");
    
    printf("out_names : ");
    for(int i = 0; i<out_count; i++){
        printf("%s ",output_names[i]);
        if( (i+1) != out_count) printf(",");
    }
    printf("\n");

    // const int64_t input_shape[] = {1, 3, img_sz, img_sz};
    const int64_t input_shape[] = {1, img_sz, img_sz, 3};
    const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
    const size_t model_input_len = model_input_ele_count * sizeof(float);

    OrtValue* input_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, model_input, model_input_len, input_shape,
                                                            input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                            &input_tensor));
    assert(input_tensor != NULL);

    int is_tensor;
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(input_tensor, &is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    //Ort output 노드 개수 많을 때...
    OrtValue** output_tensors = (OrtValue**)malloc(out_count * sizeof(OrtValue*));
    for(int i = 0; i< out_count; i++){
        output_tensors[i] = NULL;
    }

    double timer = get_time();

    ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, out_count,
                                    (OrtValue**)output_tensors));
    assert(output_tensors[0] != NULL);
    printf("===============\n");
    int ret = 0;
    printf("\nORT Run API Latency : %lfs\n\n",get_time()-timer);
    struct OrtTensorTypeAndShapeInfo* shape_info;
    size_t dim_count;
    int64_t dims[out_count][5];
    float** out = (float**)malloc(out_count*sizeof(float*));
    printf("===============\n");
    for(int n = 0; n<out_count; n++){
      ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensors[n], &shape_info));
      ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(shape_info, &dim_count));
      ORT_ABORT_ON_ERROR(g_ort->GetDimensions(shape_info, dims[n], sizeof(dims[n]) / sizeof(dims[n][0])));
      ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensors[n], (void**)&out[n]));
      printf("output : dims[%d] [1][2][3][4][5] : %ld,%ld,%ld,%ld,%ld\n",n,dims[n][0],dims[n][1],dims[n][2],dims[n][3],dims[n][4]);
      fflush(stdout);
    }
     


    int index = 0;
    for(int n = 0; n < out_count; n++){
        yolov4_transpose(out[n], dims[n][1]*dims[n][2], dims[n][3]*dims[n][4]);
        index = dims[n][1] * dims[n][2];
        for(int c = 0; c < dims[n][3]*dims[n][4]; c++){
            for(int i = 0; i<index; i++){
                // if( ((c+1) % 85 != 3) && ((c+1) % 85 != 4) ){
                if( ((c+1) % 85 == 1) || ((c+1) % 85 == 2) ){
                    out[n][c*index + i] = sigmoid(out[n][c*index + i]);
                }
            }
        }
    }

    //YOLO 실행  
    yolov4_det(out, (char*)input_file_p, img_sz);  
    

    //Free
    for(int i = 0; i<out_count; i++){
        g_ort->ReleaseValue(output_tensors[i]);
    }
    // free(output_tensors);
    g_ort->ReleaseValue(input_tensor);
    
    free(model_input);
    free(input_names);
    free(output_names);
    free(out);
    
    return ret;
}

int enable_cuda(OrtSessionOptions* session_options) {
  // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
  OrtCUDAProviderOptions o;
  // Here we use memset to initialize every field of the above data struct to zero.
  memset(&o, 0, sizeof(o));
  // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
  // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
  o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  o.gpu_mem_limit = SIZE_MAX;
  OrtStatus* onnx_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
  if (onnx_status != NULL) {
    const char* msg = g_ort->GetErrorMessage(onnx_status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(onnx_status);
    return -1;
  }
  return 0;
}


int main(int argc, char* argv[]) {
  //Yolov3_Darknet_PyTorch_Onnx_Converter
  //https://github.com/matankley/Yolov3_Darknet_PyTorch_Onnx_Converter

    if (argc < 3) {
        usage();
        return -1;
    }

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
        return -1;
    }

    ORTCHAR_T* model_path = argv[1];
    ORTCHAR_T* input_file = argv[2];

    printf("===============\n\n");
    printf("ORT MODEL : %s\n\n",model_path);

    // By default it will try CUDA first. If CUDA is not available, it will run all the things on CPU.
    // But you can also explicitly set it to DML(directml) or CPU(which means cpu-only).
    ORTCHAR_T* execution_provider = (argc >= 4) ? argv[3] : NULL;
    OrtEnv* env;
    ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
    assert(env != NULL);
    int ret = 0;
    OrtSessionOptions* session_options;
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

    if (execution_provider) {
        if (strcmp(execution_provider, ORT_TSTR("cpu")) == 0) {
            // Nothing; this is the default
        } else {
            usage();
            puts("Invalid execution provider option.");
            return -1;
        }
    } else {
        printf("Try to enable CUDA first\n");
        ret = enable_cuda(session_options);
        if (ret) {
            fprintf(stderr, "CUDA is not available\n");
        } else {
            printf("CUDA is enabled\n");
        }
    }

    OrtSession* session;
    ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
    
    int img_size = 416;
    ret = run_inference(session, input_file, img_size);
    
    // g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);
    

    
    if (ret != 0) {
        fprintf(stderr, "fail\n");
    }
    return ret;
}
