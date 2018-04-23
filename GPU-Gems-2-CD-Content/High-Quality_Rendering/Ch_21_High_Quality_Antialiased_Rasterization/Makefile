test : hqaa/hqaa
	hqaa/hqaa scenes/grids.pyg

clean :
	make -C hqaa clean
	make -C gpu clean
	make -C util clean

hqaa/hqaa : gpu/libgpu.a util/libutil.a
	make -C hqaa

gpu/libgpu.a :
	make -C gpu

util/libutil.a :
	make -C util


