CFLAGS	:=	-O3 -Wall -Wextra -g -fPIC
FLAGS	:=	-arch native -G

all: clean cpu gpu

cpu:
	gcc $(CFLAGS) jac3d.c -o prog_cpu

gpu:
	nvcc $(CLFAGS) jac3d.cu -o prog_gpu

clean:
	rm -f prog_cpu prog_gpu
