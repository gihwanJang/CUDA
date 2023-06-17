#include <cstdlib>
#include <vector>
#include <memory>
#include <math.h>
#include <string.h>

#define dt (1.0E-01f) // 학습 률
#define threshold (1.0E-02f) // 임계 값

// math functions
float activ_func(float v);
void apply_activ_func(float *input, float *output, int N);
void update_error(float *err, float *output, unsigned int Y, int N);
void update_grad(float *output, float *grad, int N);

// forward propagation
void fp_preact_conv(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
void fp_bias_conv(float preact[6][24][24], float bias[6]);

void fp_preact_poll(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]);
void fp_bias_poll(float preact[6][6][6], float bias[1]);

void fp_preact_full(float input[6][6][6], float preact[10], float weight[10][6][6][6]);
void fp_bias_full(float preact[10], float bias[10]);

// back propagation
void bp_weight_full(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
void bp_bias_full(float bias[10], float d_preact[10]);

void bp_output_poll(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
void bp_preact_poll(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
void bp_weight_poll(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]);
void bp_bias_poll(float bias[1], float d_preact[6][6][6]);

void bp_output_conv(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
void bp_preact_conv(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
void bp_weight_conv(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
void bp_bias_conv(float bias[6], float d_preact[6][24][24]);