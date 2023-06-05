#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "trapezoidal/trapezoidalGPU.h"
#include "trapezoidal/trapezoidalCPU.h"

#include "parameter/parameter.h"

#include "timer/DS_timer.h"
#include "timer/DS_definitions.h"

int main(){
    DS_timer timer(6);

    // timer setting
    timer.setTimerName(0, (char *)"serial Compute");
    timer.setTimerName(1, (char *)"openmp Compute");
    timer.setTimerName(2, (char *)"cuda Compute");
    timer.setTimerName(3, (char *)"cuda Optimizing Compute");
    timer.setTimerName(4, (char *)"host -> device");
    timer.setTimerName(5, (char *)"device -> host");

    float h = float(END - START) / float(SECTION);
    float d_res = 0;
    float serial_res = 0;
    float omp_res = 0;
    float cuda_res = 0;

    // serial trapezoidal
    timer.onTimer(0);
    serialTrapezoidal(serial_res, h);
    timer.offTimer(0);

    // openMP trapezoidal
    timer.onTimer(1);
    ompTrapezoidal(omp_res, h);
    timer.offTimer(1);

    printf("serial result : %f\n", serial_res);
    printf("openMP result : %f\n", omp_res);
    timer.printTimer();
    return 0;
}