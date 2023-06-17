
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer/DS_timer.h"
#include "timer/DS_definitions.h"

#include "cnn/layer.h"
#include "cnn/neuralNet.h"

void CNN(DS_timer&timer);

void loadData();

void forward_propagation(Layer&l_input, Layer&l_c, Layer&l_p, Layer&l_f, DS_timer&timer, double data[28][28], int mode);

void back_propagation(Layer&l_input, Layer&l_c, Layer&l_p, Layer&l_f, DS_timer&timer);

void learn(Layer&l_input, Layer&l_c, Layer&l_p, Layer&l_f, DS_timer&timer);

unsigned int classify(Layer&l_input, Layer&l_c, Layer&l_p, Layer&l_f, DS_timer&timer, double data[28][28]);

void test(Layer&l_input, Layer&l_c, Layer&l_p, Layer&l_f, DS_timer&timer);