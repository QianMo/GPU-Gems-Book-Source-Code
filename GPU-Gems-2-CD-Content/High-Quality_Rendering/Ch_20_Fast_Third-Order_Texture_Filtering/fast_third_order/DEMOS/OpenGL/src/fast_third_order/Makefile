ifeq ("$(shell uname)", "Darwin")
  OS="MacOSX"
endif

ifeq ("$(shell uname)", "Linux")
  OS="Unix"
endif

ifeq ("$(shell uname)", "FreeBSD")
  OS="Unix"
endif

ifndef OS
  $(error Unsupported operating system)
endif

MAIN=fast_third_order
CC=g++
CFLAGS= -Wall -g
INCLUDES= -I../../inc -I../../../../inc -I/usr/X11R6/include
LFLAGS=
SRCS = simple_draw_buffers.cpp ../shared/read_text_file.cpp ../shared/data_path.cpp

ifeq ($(OS), "Unix")
  CFLAGS += -DUNIX
  LFLAGS += -L/usr/X11R6/lib -lXmu -lXi -lglut -lGLU -lGL
endif

ifeq ($(OS), "MacOSX")
  CFLAGS += -DMACOS
  LFLAGS += -L/usr/lib -lobjc -framework OpenGL -framework GLUT
endif

OBJS = $(SRCS:.cpp=.o)

$(MAIN):  $(OBJS)
	$(CC) $(OBJS) -o $(MAIN) $(LFLAGS)
ifeq ($(OS), "MacOSX")
	perl ../build_app.pl $(MAIN)
endif
.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS)
	rm -f *~ $(MAIN)
ifeq ($(OS), "MacOSX")
	rm -rf $(MAIN).app
endif
