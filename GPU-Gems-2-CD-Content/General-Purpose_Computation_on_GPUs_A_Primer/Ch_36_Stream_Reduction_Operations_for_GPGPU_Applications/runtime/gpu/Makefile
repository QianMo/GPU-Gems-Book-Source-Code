
ROOTDIR = ../..
STATIC_LIBRARY = gpu
FILES = gpukernel gpuruntime gpustream gpuiterator

include ../../config/DetectOS.mk

ifeq ($(OS), Windows_NT)
SUBDIRS := ogl dx9
else 
ifeq ($(OS), Linux)
SUBDIRS :=  ogl
endif
endif

include ../../common.mk






