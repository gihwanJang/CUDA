#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mnist/mnist.h"

#include "timer/DS_timer.h"
#include "timer/DS_definitions.h"

#include "cnn/layer.h"
#include "cnn/neuralNet.h"

//timer
DS_timer timer(4);

//MNIST 데이터
mnist_data *train_set, *test_set;
unsigned int train_cnt, test_cnt;


/*
* 입력 레이어
* -
* -
* 28 x 28인 이미지 사이즈 입력
*/
Layer l_input = Layer(0, 0, 28 * 28);
/*
* 합성곱 레이어
* 5 x 5 크기의 필터 
* 6개의 출력 채널
* 24 * 24 * 6의 사이즈 입력
*/
Layer l_c = Layer(5 * 5, 6, 24 * 24 * 6);
/*
* 풀링 레이어
* 4 x 4 크기의 필터
* 1개의 출력 채널
* 6 * 6 * 6의 사이즈 입력
*/
Layer l_s = Layer(4 * 4, 1, 6 * 6 * 6);\
/*
* 합성곱 레이어
* 풀링 레이어 크기의 입력(6 * 6* 6)
* 10개의 출력 채널(0 ~ 9)
* ==
*/
Layer l_f = Layer(6 * 6 * 6, 10, 10);

void loadData();
void learn();
unsigned int classify(double data[28][28]);
void test();
void forward_propagation(double data[28][28], int mode);
void back_propagation();

int main(int argc, char const *argv[])
{
	timer.setTimerName(0, (char *)"total Learning");
    timer.setTimerName(1, (char *)"kernel Learning");
	timer.setTimerName(2, (char *)"total classify");
    timer.setTimerName(3, (char *)"kernel classify");

	loadData();

	timer.onTimer(0);
	learn();
	timer.offTimer(0);

	timer.onTimer(2);
	test();
	timer.offTimer(2);

	timer.printTimer();
	return 0;
}

// MNIST 데이터 로드
void loadData()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", &train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt);
}

// 입력 이미지에 대한 forward propagation 수행
void forward_propagation(double data[28][28], int mode)
{
	float input[28][28];

	// 입력 이미지 데이터 타입 변환 double -> float
	for (int i = 0; i < 28; ++i)
		for (int j = 0; j < 28; ++j)
			input[i][j] = data[i][j];

	// 이미지에 대한 각 레이어의 입력과 출력을 초기화
	l_input.clear();
	l_c.clear();
	l_s.clear();
	l_f.clear();

	timer.onTimer(mode);

	// 입력 레이어에 데이터 설정
	l_input.setOutput((float *)input);
	
	// convolution layer의 forward propagation
	fp_preact_c<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c.preact, (float (*)[5][5])l_c.weight);
	fp_bias_c<<<64, 64>>>((float (*)[24][24])l_c.preact, l_c.bias);
	apply_activ_func<<<64, 64>>>(l_c.preact, l_c.output, l_c.O);

	// subSampling layer의 forward propagation
	fp_preact_s<<<64, 64>>>((float (*)[24][24])l_c.output, (float (*)[6][6])l_s.preact, (float (*)[4][4])l_s.weight);
	fp_bias_s<<<64, 64>>>((float (*)[6][6])l_s.preact, l_s.bias);
	apply_activ_func<<<64, 64>>>(l_s.preact, l_s.output, l_s.O);

	// fully connected layer의 forward propagation
	fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
	fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
	apply_activ_func<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
	
	timer.offTimer(mode);
}

// back propagation 수행
void back_propagation()
{
	timer.onTimer(1);

	// fully connected layer의 back propagation
	bp_weight_f<<<64, 64>>>((float(*)[6][6][6])l_f.d_weight, l_f.d_preact, (float(*)[6][6])l_s.output);
	bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

	// subSampling layer의 back propagation
	bp_output_s<<<64, 64>>>((float(*)[6][6])l_s.d_output, (float(*)[6][6][6])l_f.weight, l_f.d_preact);
	bp_preact_s<<<64, 64>>>((float(*)[6][6])l_s.d_preact, (float(*)[6][6])l_s.d_output, (float(*)[6][6])l_s.preact);
	bp_weight_s<<<64, 64>>>((float(*)[4][4])l_s.d_weight, (float(*)[6][6])l_s.d_preact, (float(*)[24][24])l_c.output);
	bp_bias_s<<<64, 64>>>(l_s.bias, (float(*)[6][6])l_s.d_preact);

	// convolution layre의 back propagation
	bp_output_c<<<64, 64>>>((float(*)[24][24])l_c.d_output, (float(*)[4][4])l_s.weight, (float(*)[6][6])l_s.d_preact);
	bp_preact_c<<<64, 64>>>((float(*)[24][24])l_c.d_preact, (float(*)[24][24])l_c.d_output, (float(*)[24][24])l_c.preact);
	bp_weight_c<<<64, 64>>>((float(*)[5][5])l_c.d_weight, (float(*)[24][24])l_c.d_preact, (float(*)[28])l_input.output);
	bp_bias_c<<<64, 64>>>(l_c.bias, (float(*)[24][24])l_c.d_preact);

	// 가중치 업데이트
	update_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	update_grad<<<64, 64>>>(l_s.weight, l_s.d_weight, l_s.M * l_s.N);
	update_grad<<<64, 64>>>(l_c.weight, l_c.d_weight, l_c.M * l_c.N);

	timer.offTimer(1);
}

// 학습 데이터에 대한 학습
void learn()
{
	cublasHandle_t blas;
	cublasCreate(&blas);

	float err; // 현재 반복에서의 오차
	int iter = 50; // 학습 반복 횟수

	printf("Learning\n");

	for(int t = 1; t <= iter; ++t)
	{
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i)
		{
			float tmp_err;

			// 입력 이미지에 대한 forward_propagation
			forward_propagation(train_set[i].data, 1);

			l_f.bp_clear();
			l_s.bp_clear();
			l_c.bp_clear();

			// 오차에 대한 갱신
			update_error<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
			err += tmp_err;

			back_propagation();
		}

		err /= train_cnt;
		printf("%d . error: %e\n", t, err);
	}
}

// 이미지에 대한 분류
unsigned int classify(double data[28][28])
{
	float res[10];
	unsigned int max = 0;

	// 해당 이미지에 대한 foward_popagation
	forward_propagation(data, 3);

	// 출력 값을 device -> host로 복사
	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	// 0~9까지 중 확률이 가장 큰 값 선택
	for (int i = 1; i < 10; ++i)
		if (res[max] < res[i])
			max = i;

	return max;
}

// 모델 테스트
void test()
{
	printf("testing\n");

	int error = 0;

	// 테스트 이미지에 대하여 분류
	for (int i = 0; i < test_cnt; ++i){
		int clfy = classify(test_set[i].data); // 분류된 값
		int ans = test_set[i].label; // 실재 값

		// 분류된 값과 실재 값에 대한 오차 누적
		error += (clfy != ans);

		printf("classify : %d, answer : %d\n", clfy, ans);
	}

	// 최종 오차 출력
	printf("Error Rate: %.2lf%%\n", double(error) / double(test_cnt) * 100.0);
}