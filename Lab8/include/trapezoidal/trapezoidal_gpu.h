#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer/DS_timer.h"
#include "timer/DS_definitions.h"

#include "parameter/parameter.h"

void kernelCall(double&cuda_res, double h, DS_timer&timer, int mode);