
ROOTDIR = ..
EXECUTABLE = brcc
FILES = context decl express parseenv project stemnt symbol token \
	location brtdecl lexer gram main getopt brtexpress codegen \
	subprocess brtstemnt b2ctransform brtkernel b2creduce \
	b2cindex brtreduce brtgather brtvout fxc cgc \
        ps2arb ps2arb_lexer ps2arb_gram ps2arb_intermediate print_arb

SUBDIRS = splitting
LIBRARIES = brccsplitting

CPP_PATH = /usr/bin/cpp
SLOP = lexer.cpp gram.cpp gram.output gram.h \
       ps2arb_lexer.cpp ps2arb_gram.cpp ps2arb_gram.h

ADDITIONAL_DEPENDANCIES = gram.h ps2arb_gram.h

include ../common.mk

ps2arb_gram.cpp ps2arb_gram.h: ps2arb_gram.y
	bison -pps2arb_ -d  ps2arb_gram.y
	-mv -f ps2arb_gram.tab.c ps2arb_gram.cpp
	-mv -f ps2arb_gram.tab.h ps2arb_gram.h

ps2arb_lexer.cpp: ps2arb_lexer.l ps2arb_gram.h
	flex -Pps2arb_ -ops2arb_lexer.cpp ps2arb_lexer.l

lexer.cpp: lexer.l gram.h
	flex $<
	rm -rf lexer.cpp
	-mv lex.yy.c lexer.cpp

gram.cpp gram.h:  gram.y
	bison -d -b gram $<
	-mv -f gram.tab.c gram.cpp
	-mv -f gram.tab.h gram.h
# DO NOT DELETE
