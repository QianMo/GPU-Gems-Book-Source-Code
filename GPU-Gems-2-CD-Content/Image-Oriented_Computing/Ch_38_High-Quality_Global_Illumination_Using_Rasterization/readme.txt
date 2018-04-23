[----------Parthenon brief description----------]

1. System Requirements
   Parthenon requires a video card which supports vertex / pixel shader 2.0 (or later) and
   floating-point buffer in DirectX.
   You also need DX9ab.dll and DXErr9ab.dll which can be downloaded from this website.
   (Parthenon_with_dll.zip contains these dlls.)

   "Clootie_DX90_dlls.zip"
   http://clootie.narod.ru/delphi/download_dx90.html

2. Loading model data
   This program currently support MQO file only.
   (MQO is original format of Metasequoia. http://www21.ocn.ne.jp/~mizno/main_e.html)
   All models should be tessellated with triangles before loading.
   Click "File - Load Scene" to load model data.

   If two or more file is chosen, it will automatically render all file by the same 
   setup (include IBL setting) continuously. (batch rendering)

   Material parameters are converted automatically so that a material can be used for
   global illumination calculation correctly.

   If material has emission of 1.0, it's treated as a light source.
   The color of light is determined using diffuse color.
   (The diffuse color is also used as a base color of other material.)

   If emission is smaller than 1.0 and alpha (opacity) is not 1.0, it becomes a glass like material.
   When alpha is exactly 1.0, material is converted based on specular value.

   If specular value is 0.0, it becomes like clay, and if specular value is 1.0, it becomes a material like metal.
   Other specular value is converted into material like plastic.

   If power value is less than or equal to 50, its reflection is blurred based on the value (smaller is blurry), 
   and the value is larger than 50, its reflection is like perfect mirror reflection.

   Although Metasequoia has alpha / bump / decal texture data, this program only supports decal texture for now.

   The size of model is set by the size of a default preview of Metasequoia.

3. Previewing and Camera Position
   After the scene file is loaded, a preview screen is displayed from the view point of default camera.
   This preview merely shows material diffuse color.
   To set camera parameters, click "Camera - Setting", enter new camera position and gazing point position,
   and click "Set" button to apply new camera parameters.
　
4. Setup of Rendering
    < Lighting Environment >
      Image Based Lighting
        Check this if you want to use Image Based Lighting.
        The coordinate mapping of image is "Light Probe".
        It cannot deal with IBL and local lights (which are defined within the scene file) simultaneously.
        Parthenon can load .hdr format. (but some file may not be able to load).

      Scaling Factor
        Scaling factor of image brightness.

      Batch Mode Sample 
        Each rendering process in batch rendering mode is terminated when the 
        number of samples is reached this value. When rendering of current scene is 
        terminated, its image is saved automatically and it will carry out the 
        rendering of the next file.

    < Photon Mapping > 
      Emitting
        The number of the photons to store is specified.
        The more you use photons, the better the accuracy of rendering becomes.
        However, in general, using default number is often sufficient. 

　　　Gathering
        The number of the photons to use in radiance estimation is specified.
        Global Photon Map doesn't need a high quality photon map, so the 
        default number is also sufficient.	

　　　Caustics
        Currently unavailable.

　　< Final Gathering >
　　　Buffer Size
　　　　This value is currently fixed.

　　　Depth Bias
        Adjust this value when you get wrong indirect illumination.

　　< Shadow Mapping >　
　　　Buffer Size
　　　　This value is currently fixed.

　　　Depth Bias
        Adjust this value when you get wrong direct illumination.
　
5. Rendering and Saving Image
    Click "Rendering - Run" to start rendering.
    The process of rendering consists of precomputation and progressive rendering.
    When caustics is enabled, progress of caustics rendering is showed as "Caustics ** %" in caption .

    To change the resolution of a image, please execute the a program with a parameter as follows.

　　　Parthenon.exe horizontal_resolution vertical_resolution（example : Parthenon.exe 1024 768）
　　
    Known issue is that higher resolution more than 512 x 512 will be exceptionally slow.
    I have already found its cause and I'm implementing new algorithm now.

    Parthenon doesn't have explicit termination of rendering. After the rendered image has gotten 
    enough quality, click "File - Save Image" to save the result.
    Don't cancel rendering before you save the image.
    
    In batch rendering mode, the image is automatically saved as "(model file name).bmp", after  the 
    number of samples reach the value of "Batch Mode Sample".
    
    If you render same scene with different view point, the last photon map data is used again 
    and precomputation is omitted. Changing photon map parameters cause precomputation.

6. FAQ
　　Q : This program cannot run correctly!
　　A : Please make sure that there is D3DX9ab.dll and DXErr9ab.dll in current folder (the folder which program is in) or 
        a path folder. Next, check whether Vertex Shader 2.0 and Pixel Shader 2.0 are supported.
        In addition, Parthenon requires a GPU with floating point buffer support in DirectX.

　　Q : When rendering is begun, an image gets blurred like flowing...
　　A : This problem is caused by "alternate pixel center" option in video card preference.
        Moreover, anisotropic filtering also causes problem in combination with this.
        Some results of combination had been reported. 

        anisotropic filtering on  + alternate pixel center off : bluring
        anisotropic filtering on  + alternate pixel center on : excessive aliasing
        anisotropic filtering off + alternate pixel center off : correct renedering

　　　  (Thank you for your report, N.Kazu!)

　　Q : Strange black cross bar is appeared on rendered image.
　　A : It is bug in ray-tracing engine. Move gazing point or camera position slightly. 

　　Q : Antialiasing with caustics doesn't seem to work correctly.
　　A : It's just a bug. However, caustics are often blurred in many case, so its jaggies are hardly noticed. 

Parthenon is still under development. 
There are many bugs and a lot of stuff are not implemented yet. 
Feel free to send your comment, idea and rendered image!