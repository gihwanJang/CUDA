#include "cnn/neuralNet.h"

class Layer
{
public:
    int M, N, O;

    float *output;
    float *preact;

    float *bias;
    float *weight;

    float *d_output;
    float *d_preact;
    float *d_weight;

    Layer(int M, int N, int O);

    ~Layer();

    void setOutput(float *data);
    void clear();
    void bp_clear();
};