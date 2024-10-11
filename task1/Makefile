CXX = g++
CUDA_DIR = /usr/local/cuda
CUDA_COMPILER = $(CUDA_DIR)/bin/nvcc

CXXFLAGS = -O3 -w -g -ffast-math -funroll-loops -ftree-vectorize $(BS)
CUDA_FLAGS = $(addprefix -Xcompiler ,$(CXXFLAGS)) -arch=native #-arch=sm_86

INCLUDE_PATH = -I /usr/local/cuda/include
LIBRARY_PATH = -L $(CUDA_DIR)/lib -L $(CUDA_DIR)/lib64

.DEFAULT_GOAL := all

all: clean jac3d

original:
	gcc -O0 -w -g jac3d.c -o original

jac3d: jac3d.o jac3d_gpu.o
	$(CXX) jac3d.o jac3d_gpu.o $(LIBRARY_PATH) -lcudart -o prog

jac3d.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) -c main.cpp -o jac3d.o

jac3d_gpu.o: jac3d.cu
	$(CUDA_COMPILER) $(CUDA_FLAGS) -c jac3d.cu -o jac3d_gpu.o

clean:
	rm -f *.o prog original
