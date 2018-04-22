------------------------------------------------------------------------------
                        "Flo": a 2D GPU Fluid Simulator
                               by Mark J. Harris
------------------------------------------------------------------------------
                                 version 1.1
                               January 30, 2004
------------------------------------------------------------------------------

- DESCRIPTION -

This program implements a stable, two-dimensional simulation of viscous, 
incompressible fluid flow.  It performs all computation on the GPU.  It includes
support for vorticity confinement and arbitrary interior boundaries (obstacles).  
The user can experiment with the fluid by pushing the flow around with the mouse, 
injecting "ink", and drawing obstacles in the flow.

The demo and source code support the chapter "Fast Fluid Dynamics Simulation 
on the GPU", by Mark J. Harris, in the book "GPU Gems".  Please see the 
chapter for a detailed description of how the simulation works, as well as 
how it is implemented on an NVIDIA GeForce FX GPU.

The demo requires and NVIDIA GeForce FX GPU.

- QUICK START -

Go into the "flo" directory.  Double click "flo128.bat".  Flo should launch
an OpenGL window.  

Now you can play with the fluid.  Try clicking and dragging in the window.  
You will draw "ink" into the fluid and it will mix and swirl.  If you click 
and drag with the right mouse button the fluid will be pushed, but no ink will 
be added.

Now press the '`' key (The "back-quote".  It's the same key as the '~' on US 
Keyboards, and the same key used for the console in most games.)

This will bring up a set of sliders.

If the demo does not run, check the REQUIREMENTS section of this document.

NOTE: If you have Antialiaising or Anisotropic Filtering enabled in the driver control
panel, Flo will not run as fast as possible. (These settings will not affect the 
rendering quality in Flo.)



- KEYBOARD CONTOLS -

[`]:		(tilde key) Display parameter sliders

[t]:		Cycle the display mode (Velocity, Scalar, Pressure, Vorticity).

[T]:		Toggle display on/off (turn it off for more accurate simulation timing.

[L]:		Time 100 iterations and print out the results.

[p]:		Pause the fluid simulation.

[.]:		Take a single simulation step (when paused).

[r]:		Reset the fluid simulation.

[b]:		Toggle bilinear interpolation via fragment program.
		(High Quality, Lower Performance)

[m]:		Toggle interpolated texture generation.
		(Lower Quality, Higher Performance)

[v]:            Toggle vorticity confinement.

[a]:	        Toggle arbitrary boundaries. To draw boundaries, see "MOUSE CONTROLS".

[c]:		Toggle clearing of pressure each iteration.
		With this off, the solution requires fewer iterations, but
		it has more oscillatory behavior.  (It "jiggles" too much.)

[q]/[ESC]:	Quit




- MOUSE CONTROLS -

Left Mouse Button:       Inject ink and perturb the fluid.
Right Mouse Button:      Perturb the fluid w/o injecting ink.
CTRL + Left Button:      Draw boundaries. (must be enabled with the 'a' key.)



- SLIDERS (press '`' to display them) -

"Solver Iterations":     the number of iterations of the Jacobi solver executed
			 each time step of the simulation.

"Time Step":		 The size of the time step taken each simulation step.

"Grid Scale":		 The physical size of the grid cells (they are square).

"Vort. Conf. Scale":     Vorticity Confinement Scale.  Raising this increases the
                         amount of vorticity confinement.  Vorticity confinement
                         is a modeling technique which attempts to restore vorticity
                         (rotational flow) that is lost due to numerical dissipation
                         on a coarse grid. To toggle vorticity confinement (enabled by
                         default), press 'v'.

"Viscosity":		 The "thickness" of the fluid.  Increasing this will make the
                         fluid flow more like mollases than water.

"Brush Radius":		 The size of the "brush" used to push and draw in the
                         fluid with the mouse.

"Ink Red                 The color of the "Ink" injected into the fluid with
     Green               the mouse.
     Blue"



- PERFORMANCE -

Enabling and disabling different options will affect performance, because they result
in more or fewer computations (and rendering passes).  The fastest possible performance 
is obtained with these settings:

[a] arbitrary boundaries: off (default)
[v] vorticity confinement: off
[m] 2D texture generation: on  (this will reduce quality slightly)

If you can tolerate a slightly less accurate, and visibly "jiggly" simulation, you can
greatly increase speed using

[c] Toggle pressure clearing: off

Then turn on the sliders [`] and set them to 10-20 iterations.  This can give quite a
performance boost.  The mathematics of why this approximation may be undesirable (when
accuracy is needed) are beyond the scope of this readme file.

By setting all of the settings above we have seen performance of over 500 fps on a 
64x64 grid and 150 fps on a 128x128 grid running on an NVIDIA GeForce FX 5950.



- COMMAND LINE PARAMETERS -

Flo accepts one optional command line parameter, the resolution of the square
fluid simulation grid.  

Syntax: "Flo <resolution>"

This is an integer value greater than zero.  The default value of resolution
is 64.

Suggested values of resolution are 64, 128, and 256.  The larger the value, 
the slower the simulation will run.  Non-power-of-two dimensions will work, 
but the 'm' key command (above) will result in a white image on GeForce FX 
(because 2D textures must be power-of-two-dimensioned).



- HARDWARE REQUIREMENTS -

Flo requires an NVIDIA GeForce FX or Quadro FX Graphics Card. 
(www.nvidia.com)  

Flo has not been tested on other graphics cards.



- SOFTWARE REQUIREMENTS -

Flo requires the Cg runtime to be installed in order to run, and the Cg
libraries and compiler must be installed in order to build Flo.  Cg
can be obtained from the NVIDIA developer website:

http://developer.nvidia.com/object/cg_toolkit.html


Flo depends on the following external software libraries, which have been
included for your convenience: 
GLEW (http://glew.sourceforge.net) "The GL Extension Wrangling Library"
GLUT (http://www.xmission.com/~nate/glut.html) "The GL Utility Toolkit"

Flo also uses the ParamGL OpenGL slider library written by Simon Green.  
This library is included, but is also available as part of the NVIDIA SDK.



- BUILDING FLO -

There is a precompiled executable included in the Flo library.  To build 
the executable, load either the Microsoft Visual Studio 6.0 workspace, 
"flo.dsw", or the Microsoft Visual Studio .Net 2003 solution, "flo.sln".

Once loaded, select either "Release" or "Debug" configuration, and choose 
"Rebuild All" from the "Build" menu (For .Net, select "Rebuild Solution").

NOTE: If you receive compiler errors about not being to locate <cg/cggl.h> or 
linker errors about not being able to locate cg.lib or cggl.lib, then you
need to add the cg paths to your visual studios directories.  Select
Tools->Options->Directories (tab).  Select "Include Files" from the "Show
directories for:" combo box, and enter the path to the Cg include files
(typically c:\program files\nvidia corporation\cg\include).  Then change the
same combo box to "Library Files" and enter the path to the Cg libraries
(typically c:\program files\nvidia corporation\cg\lib).

NOTE: (Visual Studio .Net) It seems that GLUT has not been updated for MSVC 
.Net.  There is a conflict between the Windows and GLUT definitions of the 
exit() function.  This is easily fixed by replacing the GLUT exit() prototype 
(in glut.h) with the windows prototype.  I have already made this change in 
the included version of GLUT.  If you change to a new version of GLUT, you 
will also need to make this change.



- LICENSE -

----------------------------------------------------------------------------
Flo is Copyright 2004 Mark J. Harris and
The University of North Carolina at Chapel Hill
----------------------------------------------------------------------------

Permission to use, copy, modify, distribute and sell this software and its 
documentation for any purpose is hereby granted without fee, provided that 
the above copyright notice appear in all copies and that both that 
copyright notice and this permission notice appear in supporting 
documentation. Binaries may be compiled with this software without any 
royalties or restrictions. 

The author and The University of North Carolina at Chapel Hill make no 
representations about the suitability of this software for any purpose. 
It is provided "as is" without express or implied warranty.
----------------------------------------------------------------------------

The above permission and copyright apply only to Flo and do not apply to 
ParamGL, GLEW, or GLUT, included with Flo. GLEW and GLUT have separate 
license / copyright:

----------------------------------------------------------------------------
GLEW (http://glew.sourceforge.net) is licensed under the modified BSD license, 
the SGI Free Software License B, and the GLX Public License.
----------------------------------------------------------------------------

----------------------------------------------------------------------------
The GLUT COPYRIGHT: (http://www.xmission.com/~nate/glut.html)

The OpenGL Utility Toolkit distribution for Win32 (Windows NT &
Windows 95) contains source code modified from the original source
code for GLUT version 3.3 which was developed by Mark J. Kilgard.  The
original source code for GLUT is Copyright 1997 by Mark J. Kilgard.
GLUT for Win32 is Copyright 1997 by Nate Robins and is not in the
public domain, but it is freely distributable without licensing fees.
It is provided without guarantee or warrantee expressed or implied.
It was ported with the permission of Mark J. Kilgard by Nate Robins.

THIS SOURCE CODE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

OpenGL (R) is a registered trademark of Silicon Graphics, Inc.
----------------------------------------------------------------------------
