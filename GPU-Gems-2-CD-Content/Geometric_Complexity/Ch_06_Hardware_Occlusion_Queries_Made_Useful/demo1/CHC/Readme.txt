HARDWARE OCCLUSION QUERIES MADE USEFUL
======================================

This programn is a demonstration of the Coherent Hierarchical Culling algorithm described in the chapter "Hardware Occlusion Queries Made Useful" of the book GPU Gems II. Additional information and sample images are available at 

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

Daniel Scherzer (scherzer@cg.tuwien.ac.at)

Authors of the book chapter:

Michael Wimmer (wimmer@cg.tuwien.ac.at)
Jiri Bittner (bittner@cg.tuwien.ac.at)

Date: December 18, 2004

----------
Input
----------

Help for keybord and mouse is available with F1.

You can switch between the different occlusion modes described in the book chapter using the SPACE key (view-frustum culling, hierarchical stop and wait, coherent hierarchical culling).

A visualization of the culling algorithm from can be shown by pressing 'X'. Note that the classification of nodes usually changes in consecutive frames, so that the hierarchy seems to oscillate. This is expected, since the classification of previously visible interior nodes is based on the classification of their children, which might be more accurate (leading to the interior node to become occluded) than the query for the node itself. The demo also includes the optimization to use actual geometry as occluders for previously visible leaves. For a similar reason, this makes oscillation even stronger.

----------
The scene
----------

This demo uses a randomly generated terrain scene with a large number of futuristic objects placed on the terrain.

The hierarchical structure is a loose kd-tree where each object occurs in only one leaf node.

By playing with the parameters and moving around, the user can explore situtations where
occlusion culling is extremly useful, and where it is not (because the objects are too
sparse or there is too much to see from this viewpoint).

Note that this demo evolved from a demo showing the Light Space Perspective Shadow Map (LiSPSM) algorithm, therefore shadows can be enabled with F4. If you are interested in the shadow algorithm, please visit http://www.cg.tuwien.ac.at/research/vr/lispsm/

There is also a light view, where the view frustum is shown as a semitransparent body and the view vector as a red line emanating from the near plane.


----------
Installation
----------

A binary for Win32 is included.

The program should compile under Windows, a CMAKE makefile for multi-platform compilation is included (see www.cmake.org for details). However, parts of the program make use of pbuffers, therefore Linux compilation will require some adaptation.

For Linux, you need to have a working GLUT and GLEW installation.

----------
Structure
----------

This demo is written in C/C++ and uses OpenGL in order to keep it as universal as possible.

The program expects the ARB_Occlusion_query extension.
The program expects the ARB_Shadow extension (GeForce3/Radeon 9500 or higher for hardware support) for shadow support.

The program and especially the mathematical routines are not designed for speed or efficiency, but for simplicity and clarity.


If you find any problem with the code or have any comments, we would like to hear from you!
