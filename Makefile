CFLAGS	:=	-O3 -Wall -Wextra -g -ffast-math -funroll-loops -ftree-vectorize
FLAGS	:=	-arch native $(addprefix -Xcompiler ,$(CFLAGS))

all: clean cpu gpu

cpu:
	g++ $(CFLAGS) jac3d.cpp -o prog_cpu

gpu:
	nvcc $(FLAGS)  jac3d.cu -o prog_gpu

clean:
	rm -f prog_cpu prog_gpu
