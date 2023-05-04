# C++ 컴파일러를 설정합니다.
CXX := g++
# C++ 컴파일러 플래그를 설정합니다.
CXXFLAGS := -std=c++11 -Wall
# nvcc 컴파일러를 설정합니다.
NVCC := nvcc
# nvcc 컴파일러 플래그를 설정합니다.
NVCCFLAGS := -std=c++11 -arch=sm_75
# include 경로를 설정합니다.
INCLUDE_PATH = -I./include

# 소스코드 폴더와 빌드 폴더 경로를 설정합니다.
SRC_DIR := ./src
BUILD_DIR := ./build

# 소스파일 목록을 설정합니다.
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)
# 오브젝트 파일 목록을 설정합니다.
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS)))

# 빌드할 바이너리 파일을 설정합니다.
TARGET := program

# "all" 규칙: 바이너리 파일을 빌드합니다.
all: $(TARGET)

# 오브젝트 파일 생성 규칙: C++ 파일을 컴파일합니다.
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH)  -c $< -o $@

# 오브젝트 파일 생성 규칙: CUDA 파일을 컴파일합니다.
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_PATH) -c $< -o $@

# 바이너리 파일 생성 규칙: 오브젝트 파일을 링크합니다.
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# "clean" 규칙: 빌드된 오브젝트 파일과 바이너리 파일을 삭제합니다.
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
