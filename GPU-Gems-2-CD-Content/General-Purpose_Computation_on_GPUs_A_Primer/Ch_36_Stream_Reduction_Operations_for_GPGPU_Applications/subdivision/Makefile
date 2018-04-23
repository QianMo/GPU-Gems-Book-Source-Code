all:
	make -f Makefile.nv all
	make -f Makefile.ati all
clean: 
	make -f Makefile.nv clean
	make -f Makefile.ati clean
run: 
	-../bin/subdivision > data
	-../bin/subdivision > data.liberal
	-../bin/subdivision > data.conservative
	-../bin/subdiv_ati > data.cracks
	-../bin/modelviewer
#	-../bin/subdivision.exe > data
#	-../bin/subdivision.exe > data.liberal
#	-../bin/subdivision.exe > data.conservative
#	-../bin/subdivati.exe > data.cracks


