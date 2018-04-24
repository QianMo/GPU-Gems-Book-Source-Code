::StochasticGPU Notes::

While the code may look large, a majority of it is supporting the underlying GUI for designing BRDFs.  The main files to observe are the shaders: lafortune.cg, ward.cg, svlafortune.cg, and the sequence generation code, sequencegen.cpp.  Many of the calculations performed here can be placed into a look-up table in the shader to minimize the amount of external code needed for GPU-based importance sampling.

The BRDF designing interface uses the following mouse commands:

Over the object:
Left-Mouse Button – Rotates the camera
Middle-Mouse Button – Moves the camera closer or farther away from the object
Right-Mouse Button – Rotates the object

Over the BRDF graph:
Left-Mouse Button – Rotate the outgoing angle over the azimuthal axis
Right-Mouse Button – Rotate the outgoing angle over the polar axis
Ctrl + Right-Mouse Button – Zoom in/out on the graph

In addition, a few non-obvious features are available through the keyboard via the following input.

i/I – Toggles the visibility of the BRDF design interface 
p/P – Prints the current statistics out on all the widget options
s/S – Outputs the current screen to a 16-bit floating point FBO and saves to an HDR image file. Note: this will be upside down since OpenGL stores images from the bottom up.  Also, if you wish to output non-tonemapped files, please see the comment in common.cg.

Interface Notes:
* When using random sequences, such as Possion Disk, Best Candidate, Penrose, results will vary each time they are selected

* Operations that require the Cg source to recompile, such as changing the number of samples, require approximately one second to execute.  In addition, the first time different material types are loaded, such as Ward, there is a pause.

* If you wish to see the performance of the algorithm, just add an idle callback function that calls the glutPostRedisplay function.

Compilation Notes:
For compilation, Cg version 1.5 or higher must be installed as well as GLEW.  The DLLs included in this project for running this application independent of these libraries should be removed if you are attempting to compile the code yourself.

Acknowledgements:
This code contains freely available code from other graphics researchers, including the spherical harmonic convolution code by Ravi Ramamoorthi, radical inversion code by Matt Pharr and Greg Humphries, sampling code by Daniel Dunbar, and RGBE parsing code by Bruce Walter.  We thank these researchers for their open source code.  We also thank the Stanford scanning repository for the Happy Buddha model.  Lastly, we thank NSF, the I2 Lab, and the Media Convergence Lab for partially supporting the development of this code base.
