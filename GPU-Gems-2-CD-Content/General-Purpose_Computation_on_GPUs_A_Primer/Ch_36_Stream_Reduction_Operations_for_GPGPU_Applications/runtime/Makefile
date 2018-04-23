ROOTDIR = ..
STATIC_LIBRARY = brook
FILES = brt profiler logger

include ../config/DetectOS.mk

ifeq ($(OS), Windows_NT)
SUBDIRS := cpu gpu
LIBRARIES := cpu gpu ogl gpu-dx9
else 
ifeq ($(OS), Linux)
SUBDIRS :=  cpu gpu
LIBRARIES := cpu gpu ogl
else
SUBDIRS := cpu
endif
endif

include ../common.mk
