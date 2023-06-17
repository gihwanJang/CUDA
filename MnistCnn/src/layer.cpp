#include "cnn/layer.h"

Layer::Layer(int M, int N, int O)
{
    this->M = M;
    this->N = N;
    this->O = O;

    output = new float[sizeof(float) * O];
    preact = new float[sizeof(float) * O];
    bias = new float[sizeof(float) * N];
    weight = new float[sizeof(float) * M * N];

    d_output = new float[sizeof(float) * O];
    d_preact = new float[sizeof(float) * O];
    d_weight = new float[sizeof(float) * M * N];

    for (int i = 0; i < N; ++i)
    {
        bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

        for (int j = 0; j < M; ++j)
            weight[i * N + j] = 0.5f - float(rand()) / float(RAND_MAX);
    }
}

Layer::~Layer()
{
    delete output;
    delete preact;
    delete bias;
    delete weight;
    delete d_output;
    delete d_preact;
    delete d_weight;
}

void Layer::setOutput(float *data)
{
    memcpy(output, data, sizeof(float) * O);
}

void Layer::clear()
{
    memset(output, 0, sizeof(float) * O);
    memset(preact, 0, sizeof(float) * O);
}

void Layer::bp_clear()
{
    memset(d_output, 0, sizeof(float) * O);
    memset(d_preact, 0, sizeof(float) * O);
    memset(d_weight, 0, sizeof(float) * M * N);
}
