CPPFLAGS += -I -W -Wall -Wno-unused-parameter
LDLIBS += -lgsl -lgslcblas 

LDLIBS += -O3 -DNDEBUG
CUDA_LIBS += /data/NVIDIA_CUDA_SDK/lib
CUDA_INC += /data/NVIDIA_CUDA_SDK/common/inc
CUDA_INSTALL_PATH = /usr/local/cuda

CUDA_SDK_PATH = /data/NVIDIA_CUDA_SDK
CUDA_INSTALL_PATH = /usr/local/cuda
CUDA_SDK_LIBS = $(CUDA_SDK_PATH)/lib
CUDA_SDK_INCLUDE = $(CUDA_SDK_PATH)/common/inc


all: two_block_wallace
	g++ -o test test.cpp $(LDLIBS) $(CPPFLAGS) obj/cuda_rng_options.o -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_LIBS) -lcuda -lcudart -lGL -lGLU -lcutilD

two_block_wallace:
	nvcc --compile -g cuda/cuda_rng_options.cu -D_DEBUG -o obj/cuda_rng_options.o -Xcompiler "-m32" -Xcompiler "" -I. -Icuda/ -I/usr/local/cuda/include -I$(CUDA_SDK_INCLUDE) -DUNIX -O3

clean:
	rm -f obj/*
	rm -f test
