CFLAGS	:=	-O3 -Wall -Wextra -g -ffast-math -funroll-loops -ftree-vectorize
FLAGS	:=	$(addprefix -Xcompiler ,$(CFLAGS)) #-Xptxas -dlcm=cg

FLAGS += $(BS)

all: clean gpu

cpu:
	gcc $(CFLAGS) jac3d.c -o prog_orig

gpu:
	nvcc $(FLAGS) jac3d.cu -o prog

clean:
	rm -f prog prog_orig
