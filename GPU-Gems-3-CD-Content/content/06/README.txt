GPU GENERATED, PROCEDURAL WIND ANIMATION FOR TREES
--------------------------------------------------

This sample illustrates procedural wind animation technique for trees as
described in the chapter "GPU Generated, Procedural Wind Animation for Trees"
of the book GPU Gems III. Technique implementation for both DirectX 9 and 
DirectX 10 are provided.


Author: Renaldas Zioma (rej@scene.lt)


Help
----

Information about controls is available by pressing F1.

Sample allows you to choose between different tree and wind types.
You can see how performance is affected by changing depth of hierarchy (SLOD)
and altering number of tree instances.
NOTE: Hide Info and UI panes (pressing F1 and Tab) for better performance

You can alter procedural animation parameter using controls on the right
side of the sample window.

Global and trunk related parameters can be found in the top right corner.
Branch related parameters are situated below. Branch parameters are divided
into 3 horizontal groups: [Front], [Back] and [Side]. Each represents spatial
relation between branch and wind (see book chapter for more details).

Parameter groups have 2 columns each. Rightmost column defines base behavior,
if tree is in the upright position, while leftmost one - modifies behavior,
once tree is leaning away from the wind (due to load). Inertia propagation
and inertia delay defines how and when behaviors will be swapped to simulate
inertia phenomena.


Requirements
------------

Compiling sample:
  . Microsoft Visual Studio 2005
  . Microsoft DirectX 9.0 SDK (April 2007)

Running sample: 
  . Microsoft DirectX DirectX (April 2007) Runtime
  . Video card with Shader Model 2 support or DirectX 10 compatible



If you find any problem with the code or have any comments, don't hesitate
to contact the author!
