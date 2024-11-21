CXX = g++
CUDA_DIR = /usr/local/cuda
CUDA_COMPILER = $(CUDA_DIR)/bin/nvcc

CXXFLAGS = -O3 -w -Wall -Wextra -g -ffast-math -funroll-loops -ftree-vectorize $(BS)
CUDA_FLAGS = $(addprefix -Xcompiler ,$(CXXFLAGS)) -arch=sm_86 #-arch=sm_60

INCLUDE_PATH = -I /usr/local/cuda/include
LIBRARY_PATH = -L $(CUDA_DIR)/lib -L $(CUDA_DIR)/lib64

.DEFAULT_GOAL := all

all: clean adi3d

original:
	gcc -O0 -w -g adi3d.c -o original

adi3d: adi3d.o adi3d_gpu.o
	$(CXX) adi3d.o adi3d_gpu.o $(LIBRARY_PATH) -lcudart -o prog

adi3d.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c main.cpp -o adi3d.o

adi3d_gpu.o: adi3d.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) -c adi3d.cu -o adi3d_gpu.o

clean:
	rm -f *.o prog original
