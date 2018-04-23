
===============================================================
Tri-Cubic Filtering and Real-Time Implicit Curvature Demo
Copyright (c) 2004 by VRVis Research Center and ETH Zurich.
===============================================================
Markus Hadwiger, Christian Sigg, and Henning Scharsach.
http://www.vrvis.at/                  http://graphics.ethz.ch/
===============================================================

This demo renders an isosurface from a 128^3 distance field
volume generated from the well-known dragon model, computes the
curvature of this implicit surface, and displays a color-coding
of the magnitude of the maximum principal curvature.

The isovalue of the isosurface to render can be changed at
any time, there is no surface geometry at all in this demo!
The rendered surface is determined on a per-pixel basis via
ray casting on the GPU.

Implicit surface curvature can be computed from the first and
second derivatives of the distance field. These derivatives
can be computed via direct convolution with appropriate
filter kernels. However, this demo does not use any precomputed
information and computes all derivatives on-the-fly.

For computing curvature in this way, tri-cubic filters are
the lowest-order filters that achieve sufficient quality,
and especially the cubic B-spline and its derivatives allow
to obtain a quality that is comparable with filters of much
higher order (Kindlmann et al. 2003).

This demo first computes an image of ray/isosurface intersection
positions by casting rays into the volume, which is done
here by exploiting looping and branching in the fragment
shader (shader model 3.0), but can also be done by slicing the
volume.

These intersection positions are then used to apply NINE
different tri-cubic filters for each pixel of the output
image. Each of these filters requires a neighborhood
of 64 samples of the distance field volume, but the filtering
process can be optimized considerably by applying the method
illustrated in the "Fast Third-Order Filtering" chapter.

These nine filters are:
* three for reconstructing the gradient (first derivatives)
* six for reconstructing the Hessian (second derivatives)

When all the derivatives have been computed for each pixel,
the curvature can be computed from them and mapped to a color
via a 1D lookup table (texture).

Requirements
------------
This demo only runs on Geforce 6xxx models and above, because
it requires the nv_fragment_program2 OpenGL extension (and
similar extensions) for accessing shader model 3.0.

Usage
-----
Use the mouse to rotate the object or the light source when
the left button is pressed, or to zoom when the right button
is pressed.

Keys:

'+': increase the iso value.
'-': decrease the iso value.
'L': turn on/off lighting.
'a': switch between rotating the object or the light source.
'p': show the ray/isosurface intersection image instead of
     the computed curvature magnitudes.
'[': scale the 1D color table for showing curvature magnitude.
'b': show the bounding box of the volume.
'r': reset the view.
'f': toggle frame rate display.

