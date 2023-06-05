#include "trapezoidal/trapezoidal_gpu.h"
#include "trapezoidal/trapezoidal_cpu.h"

int main(){
    DS_timer timer(4);

    // timer setting
    timer.setTimerName(SERIAL, (char *)"serial Compute");
    timer.setTimerName(OPENMP, (char *)"openmp Compute");
    timer.setTimerName(CUDA_BASIC, (char *)"cuda Compute");
    timer.setTimerName(CUDA_OPTIMIZING, (char *)"cuda Optimizing Compute");

    double h = double(END - START) / double(SECTION);
    double serial_res = 0;
    double omp_res = 0;
    double cuda_res;

    // print parameter
    printf("start : %d, end : %d, section : %d\n\n", START, END, SECTION);

    // serial trapezoidal
    serialTrapezoidal(serial_res, h, timer, SERIAL);
    printf("serial result : %lf\n", serial_res);

    // openMP trapezoidal
    ompTrapezoidal(omp_res, h, timer, OPENMP);
    printf("openMP result : %lf\n", omp_res);

    // cuda basic trapezoidal
    kernelCall(cuda_res, h, timer, CUDA_BASIC);
    printf("cuda basic result : %lf\n", cuda_res);

    // cuda optimizing trapezoidal
    kernelCall(cuda_res, h, timer, CUDA_OPTIMIZING);
    printf("cuda optimizing result : %lf\n", cuda_res);

    timer.printTimer();
    return 0;
}