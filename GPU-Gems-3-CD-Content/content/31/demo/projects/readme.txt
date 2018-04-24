*** Fast N-Body Simulation with CUDA *** 

This sample code accompanies the GPU Gems 3 article "Fast N-body
Simulation with CUDA", by Lars Nyland, Mark Harris, and 
Jan F. Prins.

To use the code, you will need The NVIDIA CUDA Toolkit version 
1.0 or later, available at http://developer.nvidia.com/cuda

You will also need an NVIDIA 8-Series or later GPU, such as a 
GeForce 8800, 8600, or 8500, or a Quadro FX 4600 or 5600.

*** Building the Code ***
Windows:
1. Go to the "common" directory and open the cutil_vc7.sln, and 
   build both the Debug and Release configurations.  Exit. Open
   the paramgl_vc7.sln, and build both the Debug and Release 
   configurations.  Exit.
2. Go to the projects/nbody directory and open the nbody_vc7.sln.
   Build all configurations.
   
Linux:
1. Go to the common directory and type "Make; Make dbg=1".  Then
   type "Make -f Makefile_paramgl; Make -f Makefile_paramgl dbg=1".
2. Go to the projects/nbody directory and type
   "Make; Make dbg=1; Make emu=1; Make emu=1 dbg=1"


:::NOTE:::
The code is designed to work with CUDA 1.0, which will be released
before this book is released.  If you have an earlier version of 
cuda, the code may not compile. This can probably be fixed by 
changing line 50 of the file bodysystemcuda.cu (there is a comment
there that explains how).  This is due to an API change between 
CUDA 0.8 and CUDA 1.0.

*** Running the N-Body Simulator ***

You can either run the code from inside visual studio, or go to 
the bin/win32/release directory and type "nbody".

There are three execution modes: interactive mode, benchmark mode,
and test mode, described below.  There are also a number of 
command line options.

*** Command Line Options ***

--benchmark: Runs in benchmark mode, as described below.
--compare:   Runs in test mode, as described below.

Note: If neither benchmark or interactive mode is selected, 
      interactive mode will be run.

--n=<NUM>: Sets the number of bodies in the simulation.  This 
           should be a multiple of 256.  The default is 1024.
           
--p=<NUM>: Sets the width of the tile used in the simulation.
           The default is 256.
--q=<NUM>: Sets the height of the tile used in the simulation.
           The default is 1.
           
Note: q is the number of threads per body, and p*q should be 
      less than or equal to 256.
           
--i=<NUM>: Sets the number of iterations to run in benchmark mode.
           The default is 100.

*** Interactive Mode ***

In interactive mode, you will be able to visualize the bodies in
the simulation like stars in a galaxy.  There are several keyboard
and mouse controls.

Mouse Buttons:
Left: Rotate the view.
CTRL+Left: Translate the view forward and back
SHIFT+Left: Pan the view in the image plane

Keyboard:
SPACE: Pause / Un-pause the simulation
q:     Quit.
`:     Toggle sliders
p:     Cycle textured / colored / point body display
d:     Enable / Disable rendering
1:     Reset the bodies in randomized rotating "shell" 
       configuration.
2:     Reset the bodies in randomized spherical configuration 
       with randomized velocities.
3:     Reset the bodies in randomized configuration with 
       "expanding" velocity.
       
Sliders:
Velocity Damping: Controls the damping factor applied to the 
                  velocity each time step.
Softening Factor: Controls the softening factor epsilon in the 
                  gravitational force calculation.
Time step:        Controls the simulation step size.
Cluster Scale:    This is the relative size of the starting cluster.  
                  This affects the initial conditions when you press
                  '1', '2', or '3'.
Velocity Scale:   This is the scale of the initial velocities of the 
                  starting cluster. This affects the initial conditions 
                  when you press '1', '2', or '3'.

      
*** Benchmark Mode ***

In benchmark mode, the simulation is run for a number iterations
without rendering and timed, and the simulator reports the total
time, average time, and average body-body interactions per second
and GFLOP/s rates.

*** Test Mode ***

In test mode, the simulation is run on both the CPU and GPU and 
a comparison is done.  If all positions in the GPU simulation are
within a tolerance of the CPU simulation, the simulator reports
"Test PASSED".  Otherwise it reports "Test FAILED".


