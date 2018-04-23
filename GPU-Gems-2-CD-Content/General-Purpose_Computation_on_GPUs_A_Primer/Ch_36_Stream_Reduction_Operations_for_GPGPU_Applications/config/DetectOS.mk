ifndef OS

ARCH=$(shell uname | sed -e 's/-.*//g')

ifeq ($(ARCH), CYGWIN)
OS=Windows_NT
else
ifeq ($(ARCH), Linux)
OS=Linux
else
OS=Default
endif
endif

endif

