GPUGems3 Demo: LCP Algorithms for Collision Detection using CUDA
----------------------------------------------------------------


Usage: 
------

Windows: run Windows/LCPDemo_(config).exe
Linux:   run Linux/bin/LCPDemo.(config)

where (config) is an execution configuration as follows:

release     CPU optimized, solver running on GPU
debug       CPU debug, solver running on GPU
emurelease  CPU optimized, solver running in emulation on CPU
emudebug    CPU debug, solver running in emulation on CPU

The GLUT command line arguments are accepted. You need to have the CUDA
toolkit version 0.8 beta installed. You can obtain it from the NVIDIA web
pages or from this CDROM. For the release and debug configurations, you need a
NVIDIA G80 series hardware supported by the CUDA toolkit. The emulated
versions do not need a GPU (for other than rendering).

Keys:
  <enter>  toggle simulation auto run
  <0>      single step simulation
  <+><->   move front plane
  <w>      toggle wireframe rendering
  <n>      toggle normals
  <b>      toggle draw broadphase
  <q><esc> quit

Mouse:
  left          roll the box
  middle/wheel  zoom
  right         move sideways


Have fun !


Compiling the source:
---------------------

Windows: 
  - Use the VisualStudio.NET project file LCPDemo.vcproj
  - You need a working OpenGL and CUDA installation. Download driver and the
    CUDA toolkit from NVIDIAs web pages if necessary.
  - Compile the different execution configurations using VCs build settings
    selector

Linux:
  - Use the provided Makefile
  - You need a working OpenGL and CUDA installation. Download driver and the
    CUDA toolkit from NVIDIAs web pages if necessary.
  - If you have installed CUDA in a non-default location, you need to put the
    path in the CUDADIR variable on top of the Makefile.
  - Compile the different execution configurations using arguments to make:
    dbg=1   turn on debug mode, else release mode
    emu=1   run in emulator, else run on GPU


The LCP solver source features a very instructive "sequential" execution
mode. This is intended for exploring the solver, not for regular use. To
enable it, you need to define the SEQUENTIAL_SOLVER_STEPPING preprocessor
token. On Linux, add seq=1 to the make command line. On Windows, you need to
add the define manually to the preprocessor defines using the build properties
dialog for all files in the root project folder. You get the most instructive
output by compiling the solver then in debug mode.


Misc issues:
------------

If the console output in debug mode shows weird characters around class names,
your console does not support ANSI escape sequences. This does not influence
program operation. On Windows, load ansi.sys for the DOS box. On Linux, get a
better shell :) 

In emulation mode, collisions can be missed in rare circumstances. The cause
of this is not yet known and is probably due to the way thread shared memory
is set up in the emulator.

Note that the rigid body system used has no friction.

This code has been developed with the latest CUDA toolkit version available to
the production date of the CDROM. This is CUDA toolkit version 0.8 beta. The
final release version of the toolkit and future releases will provide a
slightly changed API. The solver may therefore not compile on CUDA versions
other than 0.8 beta. Please check www.kipfer.de for updates.


More info:
----------


Check for more GPGPU stuff:
http://www.kipfer.de
http://www.gpgpu.org


Contact:
--------

Peter Kipfer
Havok, Dublin
email: peter@kipfer.de
