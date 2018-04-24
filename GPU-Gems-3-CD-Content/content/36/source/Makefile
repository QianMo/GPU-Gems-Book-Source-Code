TARGET = xfeedback_aes
ASM_TARGET = aes.vp

CPP = gcc -E -x c++ -P -Wall -Wundef
DEFINE = -D MODE=0 -D NUM_KEY=1 -D NUM_ROUND=2 -D NUM_BLOCK=3 -D ENC_KEY_BEGIN=11 -D ENC_KEY_END=25 -D DEC_KEY_BEGIN=26 -D DEC_KEY_END=40

SRCS = xfeedback_aes.cc rijndael.c

$(TARGET): $(SRCS) $(ASM_TARGET)
	gcc -Wall -Wundef -O3 -g $(DEFINE) $(SRCS) -o $@ -lglut -lGLU -lGL -L/usr/X11R6/lib -lXmu -lXext -lX11 -lm

$(ASM_TARGET): aes.vpt
	$(CPP) $(DEFINE) $< -o $@

clean:
	rm -rf $(TARGET) $(ASM_TARGET)
