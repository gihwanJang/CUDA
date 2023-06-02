#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void kernelCall(float*d_a, float*d_b, float*d_res, int n, int m, int k, int block_size, int func);