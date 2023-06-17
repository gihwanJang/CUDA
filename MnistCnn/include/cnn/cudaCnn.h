
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer/DS_timer.h"
#include "timer/DS_definitions.h"

#include "cnn/cudaLayer.h"
#include "cnn/cudaNeuralNet.h"

void cudaCNN(DS_timer&timer);

void loadData();

void forward_propagation(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer, double data[28][28], int mode);

void back_propagation(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer);

void learn(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer);

unsigned int classify(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer, double data[28][28]);

void test(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer);