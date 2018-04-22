
This program is based on the "Fast, Practical and Robust Shadows" paper by:

Morgan McGuire  (Brown University, morgan@cs.brown.edu)
John F. Hughes  (Brown University, jfh@cs.brown.edu)
Kevin T. Egan	(Brown University, ktegan@cs.brown.edu)
Mark Kilgard    (NVIDIA Corporation, mjk@nvidia.com)
Cass Everitt    (NVIDIA Corporation, ceveritt@nvidia.com)

All other credits are listed at the bottom of this file.



KEY BINDINGS:

q, ESCAPE:      quit program
mouse:          adjust look direction (quake controls)
UP, DOWN
or w, s:	move camera forwards/backwards
LEFT, RIGHT:
or a, d:	move camera left/right
=:              take a screenshot (saved in "screenshots" sub-directory)
w:              draw all objects in wireframe
n, m:			rotate all objects around their own z-axes
p:              pause animating models (on/off)

z:              draw shadows (on/off)
v:              draw shadow volumes into frame buffer (on/off)
t:              make drawn shadow volumes transparent (on/off)
o:              highlight occluding objects that need z-fail method (on/off)
x:		use shadow optimizations (on/off)

0-4:            select light 0 through 4
SPACE:          light on (on/off)
j, l:           move light left/right along x-axis
y, h:           move light up/down along y-axis
i, k:           move light forwards/back along z-axis
g:              directional light (on/off)
a:              attenuation (on/off)
b:              area light (on/off)
e:              reduce area light radius
r:              expand area light radius
c:		move light with camera

5-9:            select preset scene 5 - 9





WALKTHROUGH

The following is a series of steps that can let you quickly test out a
variety of the demo's features.

Start the program.  After loading for a while you should see a cathedral
with 8 animated characters doing all kinds of strange movements.  Note
the shadows of the characters are cast against the ground and change
as the characters move.  Use the arrow keys to move left or right,
or advance forwards or backwards.  Moving the mouse will change the
angle at which you are looking (the look vector).  The red dot located 
slightly to the upper right of the screen is the current light position.
You can use the 'l' and 'j' keys to move the light left and right and
see how the surfaces and shadows react to the changing light position.
See the above list of key bindings to find all the manipulations
you can do to the light.

After moving around and changing light positions if you ever want
to come back to this original setup you can press the '5' key.  Pressing
keys '5' through '9' will load preset scenes that show some features
of the demo.

Preset 6: For this you can move the light around and see how cubes cast
shadows on the back wall and each other.  Try pressing 'v' to visually see
what the shadow volumes look like.  Also if you go around so that the
cubes cast shadows on to the eye point you can press the 'm' key to see
which cubes must use the slower z-fail method (these will be cubes that
partially obscure your view to the camera).  Then if you turn off shadow
optimizations by pressing the 'x' key you can see that no all cubes are
rendered using the slower method.

Preset 7: This shows the scissor optimization.  Shadow volumes are being
drawn visually but they are being clipped so as to next extend past areas
that the light can reach.

Preset 8: This shows a variety of shadow interactions by looking through
the dome of the cathedral onto the cubes and floor.  A directional light
is setup to shine through the dome of the cathedral.  You can
change the directional light direction by pressing 'j', 'l', or any
of the other light movement keys.

Preset 9: This is setup much like the very first preset scene except that
multiple lights of different colors have been turned on.  In general
you can turn on other lights by selecting a different light (pressing a
key from '0' to '4' selects the corresponding light), and then pressing
the space bar.  By light 0 is always on by default.






OPTIMIZATIONS

Implemented:
 - z-pass/z-fail selection per model (if shadowOptimizations on)
 - back cap fan optimization (if shadowOptimizations on)
 - light attenuation culling (if shadowOptimizations on)
 - model attenuation culling (if shadowOptimizations on)
 - light attenuation clipping (if shadowOptimizations on)
 - frustum culling for shadow front cap, extrusions, and back cap
   (if shadowOptimizations on)
 - extrusion caching (if shadowOptimizations on)
 - no back caps for directional lights (if shadowOptimizations on)

Not Implemented (often for clarity):
 - setting depth bounds
 - occlusion culling (checkOcclusionCull() included but not used)
 - doubling vertex buffer and extruding shadow volumes in a
   vertex program, (Extrude.vp file included but not used)
 - enabling GL_STENCIL_TWO_SIDE_EXT to use glActiveStencilFaceEXT()
   for setting the stencil region with one rendering pass
 - vertex/tex-coord/index arrays (currently use glVertex())
 - minimal GL state transitions
 - Letting one model have multiple instances with different
   transformations





FAQ

Q. Sometimes the drawing visual shadow volumes produces a blinking
   behavior when I make small movements or small changes in direction.
   Is this a bug?

A. No that is probably not a bug.  This is most likely because
   the scissor region optimization is cutting off the shadow.  At
   the moment the scissor region is set based on an axis-aligned box
   around the light that contains all surfaces that can receive 10/255 
   the full light energy.  Sometimes small changes in direction make
   the axis-aligned box and near clip plane act in somewhat jerky ways.
   See receivesIllumination() and getScreenBoundingRectangle().


Q. The cathedral never seems to be highlighted as needing z-fail, why
   is that?

A. Highlighting all of the edges in the cathedral causes so many lines
   to be drawn that nothing else can be seen.  For this reason we do
   not highlight the cathedral if it needs edges (in finalPass() the
   for loop starts at 1 when highlighting occluders which skips the
   cathedral).


Q. There seems to be code to load in Quake 3 level files (.bsp) but
   the code is never called, why is that?

A. All of the level files that we could find had t-junctions or other
   geometry attributes that were not compatible with the way edges are
   extracted from models in this program.
 

Q. What is the total polygon count?

A. This is the total number of triangles sent to the graphics card.
   Polygons drawn to the stencil buffer would be included in this count
   even though they are not drawn to the frame buffer (try pressing
   the 'v' key if you do want to see the shadow volume polygons drawn
   to the frame buffer).


Q. What is the visible polygon count?

A. This is the total number of triangles drawn to the frame buffer.


Q. How come when I draw shadows/visible shadow volumes the polygon count
   goes up so high?

A. This is because each polygon is drawn once with back face culling,
   and once with front face culling.  For opaque visible shadow volumes the
   polygons are actually drawn again in wireframe to highlight edges
   (see the note about GL_STENCIL_TWO_SIDE_EXT under "Optimizations").


Q. Is a Linux version available?

A. We haven't compiled one, but the SDL, Graphics3D libraries should
   support Linux.


Q. I have a question which isn't addressed here, who should I contact?

A. Kevin Egan, ktegan@cs.brown.edu


Q. I think I have found a bug in the program, who should I contact?

A. Kevin Egan, ktegan@cs.brown.edu


Q. I think I have found a flaw in the paper, who should I contact?

A. Morgan McGuire, morgan@cs.brown.edu





COMPILING

The first gotcha is that to run the program you need a copy of the
Simple DirectMedia Layer library (SDL.dll) IN YOUR CURRENT DIRECTORY.
When compiling you also need to make sure that the compiler knows what
directories to search through for the SDL and graphics3D library
and header files.  If you download and unzip the latest versions of
the SDL or graphics3D library you should find "lib" and "include"
sub-directories that store libraries and include files respectively.
It is these directories that you need to tell your compiler about.
To tell Visual C++ 6.0 where to look for libraries or include files
go to the toolbar and open Tools -> Options -> Directories and then
select either the "include files" or "library files"  item from the
"Show Directories" list.  If you decide to compile this yourself make
sure that you have the latest release of the graphics3D library.
If you want to compile this for Linux you will need to make your own
Makefile.





MODEL LOADING

The shadow volume algorithm needs silouhette edges to function
properly. Because of this many of the models publicly available will
cause debugAssert() failures in computeEdges() while trying to load
(or just corrupt the model if you are running in Release mode). The
problem arises when a model has t-junctions, extremely small triangles
(BasicModell::compact() will try to fix this), or more then two polygons
meeting at an edge, or an edge with only one polygon. For this reason
use Debug mode when loading models for the first time and don't expect
any sort of complicated levels to cast shadows because t-junctions are
what levels are all about.





CREDITS

This program is based on the "Fast, Practical and Robust Shadows" paper by:

Morgan McGuire  (Brown University, morgan@cs.brown.edu)
John F. Hughes  (Brown University, jfh@cs.brown.edu)
Kevin T. Egan	(Brown University, ktegan@cs.brown.edu)
Mark Kilgard    (NVIDIA Corporation, mjk@nvidia.com)
Cass Everitt    (NVIDIA Corporation, ceveritt@nvidia.com)

The paper and this demo are available at:
http://www.cs.brown.edu/research/graphics/games/

The library depends on the G3D library if you want to compile it directly (http://g3d-cpp.sf.net)

Most of the implementation for this demo was by Kevin Egan
(ktegan@cs.brown.edu) and Seth Block (smblock@cs.brown.edu).  Some
code along with useful support was provided by Morgan McGuire
(morgan@cs.brown.edu). Some updates were done by Peter Sibley(pgs@cs.brown.edu).
The source code for the demo is released under the BSD license (http://www.opensource.org/licenses/bsd-license.php),
and is freely distributable.

This demo uses the Simple Direct MediaLayer Library
(version 1.2.5, http://www.libsdl.org/index.php) and the
Graphics3D library (http://g3d-cpp.sourceforge.net/).
Graphics3D version 5.0 or later is necessary for this to compile.
See the corresponding web sites for licensing issues.

There were two main resources used when trying to decipher the quake3
model (md3) and level (bsp) formats.  They were the "Unofficial Quake
3 Map Specs" by Kekoa Proudfoot (kekoa@graphics.stanford.edu).  This
document can be downloaded at at http://graphics.stanford.edu/~kekoa/q3/.
Also the tutorials for loading md3 and bsp files by Ben Humphrey
(digiben@gametutorials.com) were very useful.  These can be found at
http://www.gametutorials.com/Tutorials/opengl/OpenGL_Pg5.htm. 

The mountain skybox textures came from another tutorial on skyboxes by
Ben Humphrey (digiben@gametutorials.com).  The tutorial can be found at
http://www.gametutorials.com/Tutorials/opengl/OpenGL_Pg2.htm.

"The Tick" model is a Quake 3 model made by Carl "casManG" Schell
(carl@cschell.com), you can download the model for yourself
(which contains extra textures and .skin files) at:
http://www.planetquake.com/polycount/downloads/index.asp?model=270.
Please look at the "TheTick.txt" file for licensing information
for this model.

The cathedral model is by Sam Howell (sam@themightyradish.com) and is used
with permission.

Quake 3 is a trademark of id Software (http://www.idsoftware.com).

"The Tick" character is a trademark of New England Comics
(http://www.newenglandcomics.com/).



