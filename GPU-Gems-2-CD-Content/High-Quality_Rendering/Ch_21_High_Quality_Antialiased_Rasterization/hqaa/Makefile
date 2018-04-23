SRC := ${wildcard *.cpp}
OBJ := ${patsubst %.cpp,%.o,${SRC}}

CPPFLAGS += -I../include -I../include/linux -I../gpu -I../util -DLINUX

LDFLAGS += -lGL -lGLU

hqaa : ${OBJ} ../gpu/libgpu.a ../util/libutil.a
	gcc ${LDFLAGS} -o $@ $?

clean :
	rm -f *.o hqaa
