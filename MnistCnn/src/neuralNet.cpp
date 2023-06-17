#include "cnn/neuralNet.h"

#define THREAD_NUM 4

// math functions
float activ_func(float v)
{
    return 1 / (1 + exp(-v));
}

void apply_activ_func(float *input, float *output, int N)
{
    for (int i = 0; i < N; ++i)
        output[i] = activ_func(input[i]);
}

void update_error(float *err, float *output, unsigned int Y, int N)
{
    for (int i = 0; i < N; ++i)
        err[i] = ((int(Y) == i ? 1.0f : 0.0f) - output[i]);
}

void update_grad(float *output, float *grad, int N)
{
    for (int i = 0; i < N; ++i)
        output[i] += dt * grad[i];
}

void fp_preact_conv(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 24; ++i2)
        {
            for (int i3 = 0; i3 < 24; ++i3)
            {
                for (int i4 = 0; i4 < 5; ++i4)
                {
                    for (int i5 = 0; i5 < 5; ++i5)
                    {
                        preact[i1][i2][i3] += weight[i1][i4][i5] * input[i2 + i4][i3 + i5];
                    }
                }
            }
        }
    }
}

void fp_bias_conv(float preact[6][24][24], float bias[6])
{
    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 24; ++i2)
        {
            for (int i3 = 0; i3 < 24; ++i3)
            {
                preact[i1][i2][i3] += bias[i1];
            }
        }
    }
}

void fp_preact_poll(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
    for (int i3 = 0; i3 < 6; ++i3)
    {
        for (int i4 = 0; i4 < 6; ++i4)
        {
            for (int i5 = 0; i5 < 6; ++i5)
            {
                for (int i1 = 0; i1 < 4; ++i1)
                {
                    for (int i2 = 0; i2 < 4; ++i2)
                    {
                        preact[i3][i4][i5] += weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2];
                    }
                }
            }
        }
    }
}

void fp_bias_poll(float preact[6][6][6], float bias[1])
{
    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 6; ++i2)
        {
            for (int i3 = 0; i3 < 6; ++i3)
            {
                preact[i1][i2][i3] += bias[0];
            }
        }
    }
}

void fp_preact_full(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
    for (int i1 = 0; i1 < 10; ++i1)
    {
        for (int i2 = 0; i2 < 6; ++i2)
        {
            for (int i3 = 0; i3 < 6; ++i3)
            {
                for (int i4 = 0; i4 < 6; ++i4)
                {
                    preact[i1] += weight[i1][i2][i3][i4] * input[i2][i3][i4];
                }
            }
        }
    }
}

void fp_bias_full(float preact[10], float bias[10])
{
    for (int i = 0; i < 10; ++i)
        preact[i] += bias[i];
}

void bp_weight_full(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
    for (int i1 = 0; i1 < 10; ++i1)
    {
        for (int i2 = 0; i2 < 6; ++i2)
        {
            for (int i3 = 0; i3 < 6; ++i3)
            {
                for (int i4 = 0; i4 < 6; ++i4)
                {
                    d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
                }
            }
        }
    }
}

void bp_bias_full(float bias[10], float d_preact[10])
{
    for (int idx = 0; idx < 10; ++idx)
        bias[idx] += dt * d_preact[idx];
}

void bp_output_poll(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
    for (int i1 = 0; i1 < 10; ++i1)
    {
        for (int i2 = 0; i2 < 6; ++i2)
        {
            for (int i3 = 0; i3 < 6; ++i3)
            {
                for (int i4 = 0; i4 < 6; ++i4)
                {
                    d_output[i2][i3][i4] += n_weight[i1][i2][i3][i4] * nd_preact[i1];
                }
            }
        }
    }
}

void bp_preact_poll(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 6; ++i2)
        {
            for (int i3 = 0; i3 < 6; ++i3)
            {
                float o = activ_func(preact[i1][i2][i3]);
                d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
            }
        }
    }
}

void bp_weight_poll(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
    int d = pow(6, 3);

    for (int i1 = 0; i1 < 1; ++i1)
    {
        for (int i2 = 0; i2 < 4; ++i2)
        {
            for (int i3 = 0; i3 < 4; ++i3)
            {
                for (int i4 = 0; i4 < 6; ++i4)
                {
                    for (int i5 = 0; i5 < 6; ++i5)
                    {
                        for (int i6 = 0; i6 < 6; ++i6)
                        {
                            d_weight[i1][i2][i3] += d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3];
                        }
                    }
                }
            }
        }
    }
}

void bp_bias_poll(float bias[1], float d_preact[6][6][6])
{
    int d = pow(6, 3);

    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 6; ++i2)
        {
            for (int i3 = 0; i3 < 6; ++i3)
            {
                bias[0] += dt * d_preact[i1][i2][i3] / d;
            }
        }
    }
}

void bp_output_conv(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 6; ++i2)
        {
            for (int i3 = 0; i3 < 6; ++i3)
            {
                for (int i4 = 0; i4 < 4; ++i4)
                {
                    for (int i5 = 0; i5 < 4; ++i5)
                    {
                        for (int i6 = 0; i6 < 4; ++i6)
                        {
                            d_output[i1][i2 * 4 + i4][i3 * 4 + i5] += n_weight[0][i4][i5] * nd_preact[i1][i2][i3];
                        }
                    }
                }
            }
        }
    }
}

void bp_preact_conv(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 24; ++i2)
        {
            for (int i3 = 0; i3 < 24; ++i3)
            {
                float o = activ_func(preact[i1][i2][i3]);
                d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
            }
        }
    }
}

void bp_weight_conv(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
    float d = pow(24.0f, 2.0f);

    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 5; ++i2)
        {
            for (int i3 = 0; i3 < 5; ++i3)
            {
                for (int i4 = 0; i4 < 24; ++i4)
                {
                    for (int i5 = 0; i5 < 24; ++i5)
                    {
                        d_weight[i1][i2][i3] += d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d;
                    }
                }
            }
        }
    }
}

void bp_bias_conv(float bias[6], float d_preact[6][24][24])
{
    float d = pow(24.0f, 2.0f);

    for (int i1 = 0; i1 < 6; ++i1)
    {
        for (int i2 = 0; i2 < 24; ++i2)
        {
            for (int i3 = 0; i3 < 24; ++i3)
            {

                bias[i1] += dt * d_preact[i1][i2][i3] / d;
            }
        }
    }
}