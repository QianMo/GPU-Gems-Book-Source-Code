SRC := ${wildcard *.cpp}
OBJ := ${patsubst %.cpp,%.o,${SRC}}

CPPFLAGS += -I../include -I../include/linux -DLINUX

libutil.a : ${OBJ}
	ar r $@ $?

clean :
	rm -f *.o *.a
