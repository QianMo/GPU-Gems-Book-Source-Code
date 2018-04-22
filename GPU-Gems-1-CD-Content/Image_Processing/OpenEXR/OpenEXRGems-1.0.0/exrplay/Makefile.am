## Process this file with automake to produce Makefile.in

bin_PROGRAMS = exrplay

exrplay_CXXFLAGS = -DPKG_DATA_DIR=\"$(pkgdatadir)\" $(AM_CXXFLAGS)

INCLUDES = @OPENEXR_INCLUDES@ @CG_CFLAGS@ -I$(top_builddir) \
	  -I$(top_builddir)/Glut

LDADD = @OPENEXR_LIBS@ @CG_LIBS@ -L$(top_builddir)/Glut -lGlut -lglut

exrplay_SOURCES = main.cpp Display.cpp Display.h DisplayLut.cpp DisplayLut.h \
		  clo.cpp clo.h FrameRateCounter.cpp FrameRateCounter.h

noinst_HEADERS = Display.h DisplayLut.h clo.h FrameRateCounter.h

EXTRA_DIST = main.cpp Display.cpp Display.h Display.cg DisplayLut.cpp \
	     DisplayLut.h DisplayLut.cg clo.cpp clo.h \
	     FrameRateCounter.cpp FrameRateCounter.h filmlook.lut \
	     options.xml

pkgdata_DATA = Display.cg DisplayLut.cg
