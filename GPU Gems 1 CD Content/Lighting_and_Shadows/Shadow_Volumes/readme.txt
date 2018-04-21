There are two demos, with source, for the Shadow
Volume chapter of GPU Gems.

Double-click the .exe file in the appropriate
sub-directory to run them.

The Full demo is from the NVIDIA site.  It gives
detailed, raw OpenGL code for a full implementation
of the shadowing method (with almost all
optimizations) that appears in the chatper.

The Simple demo uses the G3D library to abstract
most of the rendering and greatly simplify the
code so that the algorithm is more clear.  The
Simple demo only implements shadow casting from
directional lights and does not have any
optimizations.

Both demos require the contents of the SDL-1.2.5
and g3d-6_00-b12 directories to compile.  These
are two Open Source libraries.  Although limited
versions are installed in the g3d and SDL
directories, you can download the full releases
of these libraries from:

http://www.libsdl.org
http://g3d-cpp.sf.net

