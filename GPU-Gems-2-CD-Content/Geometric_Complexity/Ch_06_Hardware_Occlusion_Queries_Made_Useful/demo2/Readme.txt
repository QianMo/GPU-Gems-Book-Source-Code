HARDWARE OCCLUSION QUERIES MADE USEFUL
======================================

This programn is a demonstration of the Coherent Hierarchical Culling algorithm 
described in the chapter "Hardware Occlusion Queries Made Useful" of the book GPU 
Gems II. Additional information and sample images are available at

http://www.cg.tuwien.ac.at/research/vr/chcull/

Updates to this demo can can be found on the same webpage.

Copyright and Disclaimer:

This code is copyright Vienna University of Technology, 2004.


Please feel FREE to COPY and USE the code to include it in your own work,
provided you include this copyright notice.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Author of this demo:

Oliver Mattausch (matt@cg.tuwien.ac.at)

Authors of the book chapter:

Michael Wimmer (wimmer@cg.tuwien.ac.at)
Jiri Bittner (bittner@cg.tuwien.ac.at)

Date: December 18, 2004

----------
Input
----------

Help for keybord and mouse is available with F1.

You can switch between the different occlusion modes described in the book chapter using the 
SPACE key (view-frustum culling, hierarchical stop and wait, coherent hierarchical culling).

A visualization of the culling algorithm from can be shown by pressing '1' on the keyboard. Note that the 
classification of nodes usually changes in consecutive frames, so that the hierarchy seems to oscillate. 
This is expected, since the classification of previously visible interior nodes is based on the classification 
of their children, which might be more accurate (leading to the interior node to become occluded) than the 
query for the node itself.

----------
The scene
----------

The scene used in this program consists of basic objects randomly placed in a box. The extent of the box and the 
number and size of objects can be changed (see F1).

The hierarchical structure is a kd-tree with objects intersecting a split plane placed into multiple leaf nodes.

By playing with the parameters and moving around, the user can explore situtations where occlusion culling is extremly 
useful, and where it is not (because the objects are too sparse or there is too much to see from this viewpoint).

The hierarchical coherent culling algorithm is compared to simple view-frustum culling and the hierarchical stop and wait 
algorithm (which also uses hardware occlusion queries, but in a less sophisticated way).

----------
Installation
----------

A binary for Win32 is included.

The program should compile under Windows and Linux, a CMAKE makefile for multi-platform
compilation is included (see www.cmake.org for details).

For Linux, you need to have a working GLUT and GLEW installation.

----------
Structure
----------

This demo is written in C++ and partly C and uses OpenGL.
It has few external needs like limits.h, stdlib.h, math.h, string.h, stdio.h, or time.h.
We also make heavy use of stl classes, like priority_queue.

The program expects the GL_ARB_occlusion_query extension.

The program and especially the mathematical routines are not designed for speed or
efficiency, but for simplicity and clarity.

RenderTraverser.cpp: implements the core algorithm described in the paper

occquery.cpp: contains the OpenGl setup code, glut stuff, and sets up of the scene hierarchy
HierarchyNode.cpp: contains a kd-tree hierarchy implementation, which can be traversed by the culling algorithm
Geometry.cpp: represents simple drawable geometry, so that we have something to cull for the algorithm

Timers.cpp: contains exact timer functions in order to accurately measure the frame render time
stdafx.cpp: source file that includes just the standard includes
glInterface.h: includes gl, glut and glew headers
DataTypes.c: defines basic geometric datatypes
MathStuff.c: declares some handy mathematical routines
teapot.h: contains the teapot geometry

If you find any problem with the code or have any comments, we would like to hear from you!
