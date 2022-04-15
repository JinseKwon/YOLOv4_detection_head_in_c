#include <math.h>
#include <float.h>
#include <time.h>

struct timespec u_time;
double get_time() {
    clock_gettime(CLOCK_REALTIME, &u_time);
    return (u_time.tv_sec) + (u_time.tv_nsec) * 1e-9;
}

int yolov4_det(float** onnx_out, char* file_name, int img_sz);

inline float sigmoid(float x){
  return 1.f/(1.f + expf(-x));
}