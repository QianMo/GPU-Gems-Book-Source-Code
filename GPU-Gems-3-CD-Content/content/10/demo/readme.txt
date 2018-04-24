Parallel-Split Shadow Maps demo
- - - - - - - - - - - - - - - -

Requirements:
-------------

DX9 version:
- Shader model 2.0 support for multipass rendering
- Shader model 3.0 support for DX9-level rendering

OpenGL version:
- Framebuffer object support.
- GLSL support.
- Depth textures and hardware shadow mapping support.

DX10 version:
- Windows Vista.
- DirectX SDK April 2007 or later.
- Runs on reference rasterizer if no DX10 hardware available.

Controls:
---------
W/S/A/D + mouse: move the camera
Arrow keys: rotate the light
Page up / Page down / Insert / Delete: modify the scene

Many other options are available in the menus.

Compiling:
----------
Project files are included for Visual Studio 2003 and Visual
Studio 2005 Express. With VS 2005 Express, the Windows Platform
SDK is required. The source code has also been tested to compile
with the MinGW compiler.

For compiling the DX10 demo, DirectX SDK April 2007 or newer is
required. For the DX9 demo, earlier versions of the DirectX SDK
should work as well.

About:
------
This demo is based on the GPU Gems 3 chapter 'Parallel-Split
Shadow Maps on Programmable GPUs' by Fan Zhang, Hanqui Sun
and Oskari Nyman.

Programming by Oskari Nyman
Artwork by Jurica Dolic

The source code is not restricted by any license, you may
freely utilize it for any purpose.

For more information, visit the websites
http://www.cse.cuhk.edu.hk/~fzhang/
http://www.hax.fi/asko/
