# Change this location to where you installed Sh
# E.g. /usr/local
SH_INSTALL_DIR = /usr/local

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
DEPENDS = $(SOURCES:.cpp=.d)
LIBS = -lglut -lGL -lGLU -lsh
LDFLAGS = -L$(SH_INSTALL_DIR)/lib $(LIBS)
INCLUDES = -I$(SH_INSTALL_DIR)/include -I../OpenGL/include -I../GLUT/include
CPPFLAGS = $(INCLUDES)
CXXFLAGS = $(CPPFLAGS) -Wall -g
CXX = g++
MAIN = bin/dispmap

all: $(MAIN)

depend: $(DEPENDS)

clean:
	rm -f *.o *.d $(MAIN)

$(MAIN): $(OBJECTS)
	libtool --mode=link $(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $<

%.d: %.cpp
	set -e; $(CC) -M $(CPPFLAGS) $< \
                  | sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
                [ -s $@ ] || rm -f $@
	$(CXX) $(CXXFLAGS) -M $< > $@


include $(DEPENDS)

