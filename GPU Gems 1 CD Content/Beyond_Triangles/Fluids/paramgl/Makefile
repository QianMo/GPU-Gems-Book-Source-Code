ifeq ("$(shell uname)", "Linux") 
  OS="Linux"
endif

ifeq ("$(shell uname)", "Darwin")
  OS="MacOSX"
endif

ifndef OS
  $(error Unsupported operating system)
endif

CC=g++
CFLAGS= -Wall -g
INCLUDES= -I../../inc
LFLAGS=
SRCS = param.cpp paramgl.cpp

ifeq ($(OS), "Linux")
  MAIN=../../lib/libparamgl.so
  CFLAGS += -DUNIX
  LFLAGS += -shared -lGL -lGLU -lglut
endif

ifeq ($(OS), "MacOSX")
  MAIN=../../lib/libparamgl.dylib
  CFLAGS += -DMACOS
  LFLAGS += -dynamiclib -framework OpenGL -framework GLUT
endif

OBJS = $(SRCS:.cpp=.o)

$(MAIN):  $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o $(MAIN)
.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

clean:
	rm -f *.o *~ $(MAIN)
