#define USE_MNIST_LOADER
#define MNIST_DOUBLE

#include "cnn/cudaCnn.h"
#include "mnist/mnist.h"

mnist_data *train_set, *test_set;
unsigned int train_cnt, test_cnt;

// MNIST 데이터 로드
void loadData()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", &train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt);
}

void cudaCNN(DS_timer&timer)
{
    CudaLayer l_input = CudaLayer(0, 0, 28 * 28);
    CudaLayer l_c = CudaLayer(5 * 5, 6, 24 * 24 * 6);
    CudaLayer l_p = CudaLayer(4 * 4, 1, 6 * 6 * 6);
    CudaLayer l_f = CudaLayer(6 * 6 * 6, 10, 10);

	loadData();

	timer.onTimer(0);
	learn(l_input, l_c, l_p, l_f, timer);
	timer.offTimer(0);

	timer.onTimer(2);
	test(l_input, l_c, l_p, l_f, timer);
	timer.offTimer(2);
}

// 입력 이미지에 대한 forward propagation 수행
void forward_propagation(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer, double data[28][28], int mode)
{
	float input[28][28];

	// 입력 이미지 데이터 타입 변환 double -> float
	for (int i = 0; i < 28; ++i)
		for (int j = 0; j < 28; ++j)
			input[i][j] = data[i][j];

	// 이미지에 대한 각 레이어의 입력과 출력을 초기화
	l_input.clear();
	l_c.clear();
	l_p.clear();
	l_f.clear();

	timer.onTimer(mode);

	// 입력 레이어에 데이터 설정
	l_input.setOutput((float *)input);
	
	// convolution layer의 forward propagation
	cu_fp_preact_conv<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c.preact, (float (*)[5][5])l_c.weight);
	cu_fp_bias_conv<<<64, 64>>>((float (*)[24][24])l_c.preact, l_c.bias);
	cu_apply_activ_func<<<64, 64>>>(l_c.preact, l_c.output, l_c.O);

	// subSampling layer의 forward propagation
	cu_fp_preact_poll<<<64, 64>>>((float (*)[24][24])l_c.output, (float (*)[6][6])l_p.preact, (float (*)[4][4])l_p.weight);
	cu_fp_bias_poll<<<64, 64>>>((float (*)[6][6])l_p.preact, l_p.bias);
	cu_apply_activ_func<<<64, 64>>>(l_p.preact, l_p.output, l_p.O);

	// fully connected layer의 forward propagation
	cu_fp_preact_full<<<64, 64>>>((float (*)[6][6])l_p.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
	cu_fp_bias_full<<<64, 64>>>(l_f.preact, l_f.bias);
	cu_apply_activ_func<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
	
	timer.offTimer(mode);
}

// back propagation 수행
void back_propagation(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer)
{
	timer.onTimer(1);

	// fully connected layer의 back propagation
	cu_bp_weight_full<<<64, 64>>>((float(*)[6][6][6])l_f.d_weight, l_f.d_preact, (float(*)[6][6])l_p.output);
	cu_bp_bias_full<<<64, 64>>>(l_f.bias, l_f.d_preact);

	// subSampling layer의 back propagation
	cu_bp_output_poll<<<64, 64>>>((float(*)[6][6])l_p.d_output, (float(*)[6][6][6])l_f.weight, l_f.d_preact);
	cu_bp_preact_poll<<<64, 64>>>((float(*)[6][6])l_p.d_preact, (float(*)[6][6])l_p.d_output, (float(*)[6][6])l_p.preact);
	cu_bp_weight_poll<<<64, 64>>>((float(*)[4][4])l_p.d_weight, (float(*)[6][6])l_p.d_preact, (float(*)[24][24])l_c.output);
	cu_bp_bias_poll<<<64, 64>>>(l_p.bias, (float(*)[6][6])l_p.d_preact);

	// convolution layre의 back propagation
	cu_bp_output_conv<<<64, 64>>>((float(*)[24][24])l_c.d_output, (float(*)[4][4])l_p.weight, (float(*)[6][6])l_p.d_preact);
	cu_bp_preact_conv<<<64, 64>>>((float(*)[24][24])l_c.d_preact, (float(*)[24][24])l_c.d_output, (float(*)[24][24])l_c.preact);
	cu_bp_weight_conv<<<64, 64>>>((float(*)[5][5])l_c.d_weight, (float(*)[24][24])l_c.d_preact, (float(*)[28])l_input.output);
	cu_bp_bias_conv<<<64, 64>>>(l_c.bias, (float(*)[24][24])l_c.d_preact);

	// 가중치 업데이트
	cu_update_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	cu_update_grad<<<64, 64>>>(l_p.weight, l_p.d_weight, l_p.M * l_p.N);
	cu_update_grad<<<64, 64>>>(l_c.weight, l_c.d_weight, l_c.M * l_c.N);

	timer.offTimer(1);
}

// 학습 데이터에 대한 학습
void learn(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer)
{
	cublasHandle_t blas;
	cublasCreate(&blas);

	float err; // 현재 반복에서의 오차
	int iter = 50; // 학습 반복 횟수

	printf("Learning\n");
	printf("learnig data : %d\n", train_cnt);

	for(int t = 1; t <= iter; ++t)
	{
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i)
		{
			float tmp_err;

			// 입력 이미지에 대한 forward_propagation
			forward_propagation(l_input, l_c, l_p, l_f, timer, train_set[i].data, 1);

			l_f.bp_clear();
			l_p.bp_clear();
			l_c.bp_clear();

			// 오차에 대한 갱신
			cu_update_error<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
			err += tmp_err;

			// 가중치 업데이트를 위한 back_propagation
			back_propagation(l_input, l_c, l_p, l_f, timer);
		}

		err /= train_cnt;
		printf("%d . error: %e\n", t, err);
	}
}

// 이미지에 대한 분류
unsigned int classify(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer, double data[28][28])
{
	float res[10];
	unsigned int max = 0;

	// 해당 이미지에 대한 foward_popagation
	forward_propagation(l_input, l_c, l_p, l_f, timer, data, 3);

	// 출력 값을 device -> host로 복사
	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	// 0~9까지 중 확률이 가장 큰 값 선택
	for (int i = 1; i < 10; ++i)
		if (res[max] < res[i])
			max = i;

	return max;
}

// 모델 테스트
void test(CudaLayer&l_input, CudaLayer&l_c, CudaLayer&l_p, CudaLayer&l_f, DS_timer&timer)
{
	printf("testing\n");

	int error = 0;

	// 테스트 이미지에 대하여 분류
	for (int i = 0; i < test_cnt; ++i){
		int clfy = classify(l_input, l_c, l_p, l_f, timer, test_set[i].data); // 분류된 값
		int ans = test_set[i].label; // 실재 값

		// 분류된 값과 실재 값에 대한 오차 누적
		error += (clfy != ans);

		printf("classify : %d, answer : %d\n", clfy, ans);
	}

	printf("\ntotal test img : %d, error test img %d\n", test_cnt, error);

	// 최종 오차 출력
	printf("Error Rate: %.2lf%%\n", double(error) / double(test_cnt) * 100.0);
}
