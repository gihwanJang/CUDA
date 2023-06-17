#include "cnn/neuralNet.h"

// layer 클래스
class Layer
{
public:
    int M, N, O; // 입력, 출력, 뉴런의 갯수

    float *bias; // 편향
    float *weight; // 가중치

    float *output; // 출력
    float *preact; // 입력 -> 활성화

    float *d_output; // 출력 기울기
    float *d_preact; // 활성화 입력의 기울기
    float *d_weight; // 가중치의 기울기

    Layer(int M, int N, int O); // 생성자
    ~Layer(); // 소멸자

    void setOutput(float *data); // 출력 설정
    void clear(); // 출력, 활성화 입력 초기화
    void bp_clear(); // 기울기 값 초기화
};