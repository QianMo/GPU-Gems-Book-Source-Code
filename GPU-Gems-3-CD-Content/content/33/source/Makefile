# CUDA compiler flags
CUDADIR = /usr/local/cuda

# system flags
FLAGS  = -D_REENTRANT -DLINUX
CUDAFLAGS =

# CUDA emulation mode
ifeq ($(emu),1)
	CUDAFLAGS += -deviceemu
	EMUBUILD = emu
endif

# generic compiler flags
ifeq ($(dbg),1)
	FLAGS += -g -Wall -DDEBUG -DOUTLINE
	CUDAFLAGS += -g -D_DEBUG -DUNIX -DLINUX
	BUILDVARIANT = debug
else 
	FLAGS += -O -Wall
	CUDAFLAGS += -O -DUNIX -DLINUX -Xocg "-mem"
	BUILDVARIANT = release
endif

# feature: run the solver such that you can observe the solution steps
ifeq ($(seq),1)
	FLAGS += -DSEQUENTIAL_SOLVER_STEPPING
	CUDAFLAGS += -DSEQUENTIAL_SOLVER_STEPPING
	SEQBUILD = seq
endif

EXE     = Linux/bin/LCPDemo.$(EMUBUILD)$(BUILDVARIANT)$(SEQBUILD)

COMPILER  = g++ -pipe -fPIC
INCPATHS  = -Iinclude -Igridlib -I.
LIBS      = -lglut -lGLEW -lGLU -lGL -lcuda -lcudart -lXmu -lXi -lX11

LIBPATHS  = -LLinux/lib -L/usr/X11R6/lib -L$(CUDADIR)/lib

CPPSRC = gridlib/GLCamera \
	gridlib/GbMath \
	gridlib/GbVec3 \
	gridlib/GbVec3i \
	gridlib/GbVec4 \
	gridlib/GbMatrix3 \
	gridlib/GbMatrix4 \
	gridlib/GbQuaternion \
	gridlib/GbPlane \
	gridlib/GoRigidBody \
	CUDASolver \
	RigidShape \
	RigidTetrahedron \
	RigidHexahedron \
	RigidObject \
	bank_checker \
	main.glut

CUDASRC = solver

OBJ = $(CUDASRC:=.o) $(CPPSRC:=.o)

%.o: %.cu
	$(CUDADIR)/bin/nvcc -o $@ -c $< -Xcompiler "-m32" $(INCPATHS) -I$(CUDADIR)/include $(CUDAFLAGS)

.PRECIOUS: $(CUDAGEN)

.C.o:
	$(COMPILER) $(FLAGS) $(INCPATHS) -c $< -o $@

.cpp.o:
	$(COMPILER) $(FLAGS) $(INCPATHS) -c $< -o $@

.c.o:
	gcc $(FLAGS) $(INCPATHS) -c $< -o $@

all: $(EXE)

$(EXE): $(OBJ)
	$(COMPILER) $(FLAGS) $(LIBPATHS) $(OBJ) $(LIBS) -o $(EXE)

clean:
	rm -f core *~ gridlib/*~ $(OBJ) $(EXE) *.stackdump *.log *.cpp?.i* *.ptx *.cubin *.fatbin.c *.cudafe* *.cu.c

