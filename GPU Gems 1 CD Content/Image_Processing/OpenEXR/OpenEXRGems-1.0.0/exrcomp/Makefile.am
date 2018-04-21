## Process this file with automake to produce Makefile.in

bin_PROGRAMS = exrcomp

exrcomp_CXXFLAGS = -DPKG_DATA_DIR=\"$(pkgdatadir)\" $(AM_CXXFLAGS)

INCLUDES = @OPENEXR_INCLUDES@ @CG_CFLAGS@ -I$(top_builddir) \
	   -I$(top_builddir)/Glut

LDADD = @OPENEXR_LIBS@ @CG_LIBS@ -L$(top_builddir)/Glut -lGlut -lglut

exrcomp_SOURCES = main.cpp clo.cpp clo.h GlFloatPbuffer.cpp GlFloatPbuffer.h \
		  Comp.h Comp.cpp

EXTRA_DIST = main.cpp clo.cpp GlFloatPbuffer.cpp GlFloatPbuffer.h Comp.h \
	     Comp.cpp CompIn.cg CompOut.cg CompOver.cg

pkgdata_DATA = CompIn.cg CompOut.cg CompOver.cg
