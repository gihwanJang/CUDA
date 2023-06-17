#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include <cuda.h>
#include <stdio.h>
#include <time.h>

#include "mnist.h"
#include "layer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

mnist_data *train_set, *test_set;
unsigned int train_cnt, test_cnt;

Layer l_input = Layer(0, 0, 28 * 28);
Layer l_c = Layer(5 * 5, 6, 24 * 24 * 6);
Layer l_s = Layer(4 * 4, 1, 6 * 6 * 6);
Layer l_f = Layer(6 * 6 * 6, 10, 10);

void loadData();
void learn();
unsigned int classify(double data[28][28]);
void test();
double forward_propagation(double data[28][28]);
double back_propagation();

int main(int argc, char const *argv[])
{
	srand(time(NULL));
	loadData();
	learn();
	test();
	return 0;
}

void loadData()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", &train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt);
}

double forward_propagation(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c.clear();
	l_s.clear();
	l_f.clear();

	clock_t start, end;
	start = clock();

	l_input.setOutput((float *)input);
	
	fp_preact_c<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c.preact, (float (*)[5][5])l_c.weight);
	fp_bias_c<<<64, 64>>>((float (*)[24][24])l_c.preact, l_c.bias);
	apply_activ_func<<<64, 64>>>(l_c.preact, l_c.output, l_c.O);

	fp_preact_s<<<64, 64>>>((float (*)[24][24])l_c.output, (float (*)[6][6])l_s.preact, (float (*)[4][4])l_s.weight);
	fp_bias_s<<<64, 64>>>((float (*)[6][6])l_s.preact, l_s.bias);
	apply_activ_func<<<64, 64>>>(l_s.preact, l_s.output, l_s.O);

	fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
	fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
	apply_activ_func<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

double back_propagation()
{
	clock_t start, end;

	start = clock();

	bp_weight_f<<<64, 64>>>((float(*)[6][6][6])l_f.d_weight, l_f.d_preact, (float(*)[6][6])l_s.output);
	bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

	bp_output_s<<<64, 64>>>((float(*)[6][6])l_s.d_output, (float(*)[6][6][6])l_f.weight, l_f.d_preact);
	bp_preact_s<<<64, 64>>>((float(*)[6][6])l_s.d_preact, (float(*)[6][6])l_s.d_output, (float(*)[6][6])l_s.preact);
	bp_weight_s<<<64, 64>>>((float(*)[4][4])l_s.d_weight, (float(*)[6][6])l_s.d_preact, (float(*)[24][24])l_c.output);
	bp_bias_s<<<64, 64>>>(l_s.bias, (float(*)[6][6])l_s.d_preact);

	bp_output_c<<<64, 64>>>((float(*)[24][24])l_c.d_output, (float(*)[4][4])l_s.weight, (float(*)[6][6])l_s.d_preact);
	bp_preact_c<<<64, 64>>>((float(*)[24][24])l_c.d_preact, (float(*)[24][24])l_c.d_output, (float(*)[24][24])l_c.preact);
	bp_weight_c<<<64, 64>>>((float(*)[5][5])l_c.d_weight, (float(*)[24][24])l_c.d_preact, (float(*)[28])l_input.output);
	bp_bias_c<<<64, 64>>>(l_c.bias, (float(*)[24][24])l_c.d_preact);

	update_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	update_grad<<<64, 64>>>(l_s.weight, l_s.d_weight, l_s.M * l_s.N);
	update_grad<<<64, 64>>>(l_c.weight, l_c.d_weight, l_c.M * l_c.N);

	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

void learn()
{
	cublasHandle_t blas;
	cublasCreate(&blas);

	float err;
	int iter = 50;

	double time_taken = 0.0;

	printf("Learning\n");

	while (iter < 0 || iter-- > 0)
	{
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i)
		{
			float tmp_err;

			time_taken += forward_propagation(train_set[i].data);

			l_f.bp_clear();
			l_s.bp_clear();
			l_c.bp_clear();

			update_error<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
			err += tmp_err;

			time_taken += back_propagation();
		}

		err /= train_cnt;
		printf("error: %e, time_on_gpu: %lf\n", err, time_taken);

		if (err < threshold)
		{
			printf("Training complete, error less than threshold\n\n");
			break;
		}
	}

	printf("\n Time - %lf\n", time_taken);
}

unsigned int classify(double data[28][28])
{
	float res[10];
	unsigned int max = 0;

	forward_propagation(data);

	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i)
		if (res[max] < res[i])
			max = i;

	return max;
}

void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i)
		if (classify(test_set[i].data) != test_set[i].label)
			++error;

	printf("Error Rate: %.2lf%%\n", double(error) / double(test_cnt) * 100.0);
}