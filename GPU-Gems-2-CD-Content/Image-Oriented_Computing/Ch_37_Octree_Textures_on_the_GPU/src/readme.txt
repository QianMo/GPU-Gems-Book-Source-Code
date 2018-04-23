----------------------------------------------
Octree Textures on the GPU - GPUGems 2 release
----------------------------------------------
2004-11-18

IMPORTANT:

* Please make sure that Cg is installed in 
  "C:\Program Files\NVIDIA Corporation\Cg" 
  before compiling.

* This source code compiles with Visual Studio .NET
  It is also possible to compile it under Linux, but
  the Makefile is left as an exercise to the reader ;-)

* The output files are copied into the "exe" directory
  where they can be launched using the "bat" files.
  (there are a few command line parameters)

OPTIONS:

* If you are using the Cg compiler >= v1.3 you can uncomment 
  the "CG_V1_2" define in "liboctreegpu/src/config.h"

* "paint" project: 
   - to change maximum tree depth increase PAINT_MAX_DEPTH in "CPaintNode.h"
   - to change default subdivision level edit PAINT_DEFAULT_DEPTH in "CPaintNode.h"

* "simul" project:
   - to change simulation resolution edit SIMULTREE_MAX_DEPTH in "simul\config.h"

REMARKS:

* If you would like to use octree textures into your
  own applications, the best is to start by looking
  at the CPaintTree and CPaintNode classes from the
  paint project.

* Please excuse the lack of comments and poor quality 
  of some parts of the code. This was not designed at 
  all as a commercial quality project and is for most
  part "deadline in 2 hours" research code ... 

* If you make some nice effects, please let me know ! 
  It is great to know that a piece of code has been 
  useful to someone :-)

LIBS:

The liboctreegpu library uses free source code from 
Magic Software, Inc. (http://www.magic-software.com)
The paint and simul applications use the glut, gluX, 
libtexture and lib3ds libraries.

--
Please visit http://www.aracknea.net/octreetex for updates
--
(c) Sylvain Lefebvre 2004 - all rights reserved
--
