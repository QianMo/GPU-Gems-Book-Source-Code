High Quality Antialiased Rasterization Source Code
GPU Gems II, Chapter 21

December, 2004


Code that implements tiled supersampled rasterization with GPU accelerated
downsample and filtering.  Based on code from NVIDIA Gelato film renderer.

Please see README.html for full documentation.


To compile and test in Windows:

  - Open hqaa/hqaa.vcproj
  - Build and run the Release target

To compile in Linux:

  - Just type "make" at the top level


This should compile the source code for the hqaa application and
run one of the tests, scenes/grid.pyg, which will render a killeroo
scene to an onscreen window.

