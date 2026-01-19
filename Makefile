NVCC = nvcc
CXX = g++
CUDA_PATH ?= /usr/local/cuda

# Architecture flags for typical modern GPUs (Ampere, Turing, Volga, Pascal)
ARCH_FLAGS = -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_60,code=sm_60

NVCC_FLAGS = -O3 -std=c++17 $(ARCH_FLAGS) -lcublas
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

TARGET = nn_cuda_bench
SRC = nn_cuda_bench.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
