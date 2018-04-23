PROGRAM = Sort_Linux

SOURCE = GLSLShader.cpp RenderTexture.cpp main.cpp

HFILES = GLSLShader.h RenderTexture.h defines.h

TARGETS = $(PROGRAM)
OBJECTS = $(patsubst %.cpp,%.o,$(SOURCE))
LDFLAGS = -lglut -lGLU -lGL -L./Linux/lib -lGLEW -L/usr/X11R6/lib -lXmu -lX11

default: $(TARGETS)

.cpp.o: $(SOURCE)
	g++ -c -O2 -I. -I./include -o $@ $<
#	g++ -c -g -DDEBUG -I. -I./include -o $@ $<

$(TARGETS): $(OBJECTS)
	g++ -o $@ $(OBJECTS) $(LDFLAGS)

clean:
	rm -f *.o $(TARGETS) *~ GPUGems.ncb BuildLog.xml
