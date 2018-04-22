/**
  @file RenderDevice.h

  Graphics hardware abstraction layer (wrapper for OpenGL).

  You can freely mix OpenGL calls with RenderDevice, just make sure you put
  the state back the way you found it or you will confuse RenderDevice.

  @maintainer Morgan McGuire, morgan@graphics3d.com
  @created 2001-05-29
  @edited  2003-12-07
*/

#ifndef GLG3D_RENDERDEVICE_H
#define GLG3D_RENDERDEVICE_H

#include "graphics3D.h"
#include "GLG3D/Texture.h"
#include "GLG3D/Milestone.h"
#include "GLG3D/VertexProgram.h"
#include "GLG3D/PixelProgram.h"
#include "GLG3D/VARArea.h"

typedef unsigned int uint;

namespace G3D {

/**
 Number of hardware texture units to track state for.
 */
#define MAX_TEXTURE_UNITS 8


/**
 Used by RenderDevice::init
 */
class RenderDeviceSettings {
public:
    int     width;
    int     height;

    /* The number of bits in <B>each</B> color channel of the frame buffer.
       5, <B>8</B>.*/
    int     rgbBits;

    /* The number of bits in the alpha channel of the frame buffer. 0, 1, <B>8</B> */
    int     alphaBits;

    /** 16, <B>24</B>, 32 */
    int     depthBits;

    /** <B>8</B> */
    int     stencilBits;

    /** Number of samples per pixel for anti-aliasing purposes.  <B>1</B> (none), 4, 8 */
    int     fsaaSamples;

    /** Will you accept a software rendering pipeline? */
    bool    hardware;

    bool    fullScreen;

    /** Should buffer flips be un-hitched from refresh rate?  <B>true</B>, false.  True
        generally gives higher frame rates.*/
    bool    asychronous;

    /** Allocate a stereo display context. true, <B>false</B> */
    bool    stereo;

    /** Specify the value at which lighting saturates
     before it is applied to surfaces.  1.0 is the default OpenGL value,
     higher numbers increase the quality of bright lighting at the expense of
     color depth.Default is 1.0.  Set
        to 2.0 to make a Color3::WHITE light 50% of the maximum brightness. */
    double  lightSaturation;

    /** In cycles/sec */
    int     refreshRate;

    /**
     If true, you should set up your event loop as described in the 
     docs for RenderDevice::resize.
     */
    bool    resizable;

    /**
     When true, a window frame and title bar are present.
     */
    bool    framed;

    RenderDeviceSettings() :
        width(800),
        height(600),
        rgbBits(8),
        alphaBits(8),
        depthBits(24),
        stencilBits(8),
        fsaaSamples(1),
        hardware(true),
        fullScreen(false),
        asychronous(true),
        stereo(false),
        lightSaturation(1.0),
        refreshRate(85),
        resizable(false),
        framed(true) {}
};

class VAR;

/**
 You must call RenderDevice::init() before using the RenderDevice.
  
 Rendering interface that abstracts OpenGL.  OpenGL is a basically
 good API with some rough spots.  Three of these are addressed by
 RenderDevice.  First, OpenGL state management is both tricky and
 potentially slow.  Second, OpenGL functions are difficult to use
 because many extensions have led to an evolutionary rather than
 designed API.  For type safety, new enums are introduced for values
 instead of the traditional OpenGL GLenum's, which are just ints.
 Third, OpenGL intialization is complicated.  This interface
 simplifies it significantly.

 <P> On Windows (G3D_WIN32) RenderDevice supports a getHDC() method that
 returns the HDC for the window.

 <P> NICEST line and point smoothing is enabled by default (however,
 you need to set your alpha blending mode to see it).

 <P> glEnable(GL_NORMALIZE) is set by default.  glEnable(GL_COLOR_MATERIAL) 
     is enabled by default.

 <P> For stereo rendering, set <CODE>RenderDeviceSettings::stereo = true</CODE>
     and use glDrawBuffer to switch which eye is being rendered.  Only
     use RenderDevice::beginFrame/RenderDevice::endFrame once per frame,
     but do clear both buffers separately.

 <P> The only OpenGL calls <B>NOT</B> abstracted by RenderDevice are
     fog and texture coordinate generation.  For everything else, use
     RenderDevice.

 <P>
 Example
  <PRE>
   RenderDevice renderDevice = new RenderDevice();
   renderDevice->init(RenderDeviceSettings());
  </PRE>

  RenderDevice requires SDL and OpenGL.  

  <P>
  Example 2 (textured quad)
  <PRE>
    RenderDevice* renderDevice = new RenderDevice();
    renderDevice->init(640, 480);

    TextureRef sprite = new Texture("Grass Texture", "image.jpg");

    renderDevice->beginFrame();
    renderDevice->pushState();
    renderDevice->clear(true, true, true);
    renderDevice->setCullFace(RenderDevice::CULL_NONE);
    renderDevice->setProjectionMatrix3D(-.2, .2, -.15, .15, .2, 200);
    renderDevice->setTexture(0, sprite);
    renderDevice->setColor(Color3::WHITE);
    renderDevice->beginPrimitive(RenderDevice::QUADS);
        renderDevice->setTexCoord(0,  Vector2(0, 1));
        renderDevice->sendVertex(Vector3(-3, -3, -5));
        
        renderDevice->setTexCoord(0,  Vector2(1, 1));
        renderDevice->sendVertex(Vector3( 3, -3, -5));
        
        renderDevice->setTexCoord(0,  Vector2(1, 0));
        renderDevice->sendVertex(Vector3( 3,  3, -5));

        renderDevice->setTexCoord(0,  Vector2(0, 0));
        renderDevice->sendVertex(Vector3(-3,  3, -5));
    renderDevice->endPrimitive();
    renderDevice->popState();

    renderDevice->endFrame();

    while (true);

    renderDevice->cleanup();
    </PRE>
 */
class RenderDevice {
public:
    enum Primitive {LINES, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP,
                    TRIANGLE_FAN, QUADS, QUAD_STRIP, POINTS};

    enum RenderMode {RENDER_SOLID, RENDER_WIREFRAME, RENDER_POINTS};

    
    enum {MAX_LIGHTS = 8};

private:

    friend class VAR;
    friend class VARArea;
    friend class Milestone;

    enum Vendor {NVIDIA, ATI, ARB};

    /**
     Status and debug statements are written to this log.
     */
    class Log*                  debugLog;

    /**
     The current GLGeom generation.  Used for vertex arrays.
     */
    uint32                      generation;

    /**
     The set of supported OpenGL extensions.
     */
    Set<std::string>            extensionSet;


    void setGamma(
        double                  brightness,
        double                  gamma);

	int                         stencilBits;

    /**
     Actual number of depth bits.
     */
    int                         depthBits;

    /**
     The intensity at which lights saturate.
     */
    double                      lightSaturation;

	void setVideoMode();

    /**
     Initialize the OpenGL extensions.
     */
    void initGLExtensions();

    /**
     True if EXT_stencil_wrap is in the extension list.
     */
    bool                        stencilWrapSupported;

    /**
     True if GL_EXT_texture_rectangle is in the extension list.
     */
    bool                        textureRectangleSupported;

    /**
     True if GL_ARB_vertex_program is in the extension list.
     */
    bool                        _supportsVertexProgram;
    bool                        _supportsNVVertexProgram2;

    bool                        _supportsVertexBufferObject;
    bool                        _supportsTwoSidedStencil;

    /**
     True if GL_ARB_fragment_program is in the extension list.
     */
    bool                        _supportsFragmentProgram;
    /**
     For counting the number of beginFrame/endFrames.
     */
    int                         beginEndFrame;

    /** Sets the texture matrix without checking to see if it needs to
        be changed.*/
    void forceSetTextureMatrix(int unit, const double* m);

    /** Time at which the previous endFrame() was called */
    double                      lastTime;

    /** Sets vendor */
    void computeVendor();
    Vendor                      vendor;

    /** Exponentially weighted moving average frame rate */
    double                      emwaFrameRate;

    /** Argument to last beginPrimitive() */
    Primitive                   currentPrimitive;

    /** Number of vertices since last beginPrimitive() */
    int                         currentPrimitiveVertexCount;
   
    /** Helper for setXXXArray.  Sets the currentVARArea and
        makes some consistency checks.*/
    void setVARAreaFromVAR(const class VAR& v);

    /** The area used inside of an indexedPrimitives call. */
    VARAreaRef                  currentVARArea;

    /** Number of triangles since last beginFrame() */
    int                         triangleCount;

    double                      emwaTriangleCount;
    double                      emwaTriangleRate;

	/** Updates the polygon count based on the primitive */
	void countPrimitive(RenderDevice::Primitive primitive, int numVertices);

    std::string                 cardDescription;

    /**
     Sets the milestones on the currentVARArea.
     */
    void setVARAreaMilestone();

    /** Called by sendIndices. */
    void internalSendIndices(
        RenderDevice::Primitive primitive,
        size_t                  indexSize, 
        int                     numIndices, 
        const void*             index) const;

    ////////////////////////////////////////////////////////////////////
public:
    RenderDevice();
    ~RenderDevice();

    /**
     Checkmarks all rendering state (<B>including</B> OpenGL fog and texture
     coordinate generation).
     */
    void pushState();

    /**
     Sets all state to a clean rendering environment.
     */
    void resetState();

    /**
     Restores all state to whatever was pushed previously.  Push and 
     pop must be used in matching pairs.
     */
    void popState();

    void clear(bool clearColor, bool clearDepth, bool clearStencil);

    enum DepthTest   {DEPTH_GREATER,     DEPTH_LESS,       DEPTH_GEQUAL,  
                      DEPTH_LEQUAL,      DEPTH_NOTEQUAL,   DEPTH_EQUAL,   
                      DEPTH_ALWAYS_PASS, DEPTH_NEVER_PASS};

    enum AlphaTest   {ALPHA_GREATER,     ALPHA_LESS,       ALPHA_GEQUAL,  
                      ALPHA_LEQUAL,      ALPHA_NOTEQUAL,   ALPHA_EQUAL,  
                      ALPHA_ALWAYS_PASS, ALPHA_NEVER_PASS};

    enum StencilTest {STENCIL_GREATER,   STENCIL_LESS,     STENCIL_GEQUAL,
                      STENCIL_LEQUAL,    STENCIL_NOTEQUAL, STENCIL_EQUAL, 
                      STENCIL_ALWAYS_PASS, STENCIL_NEVER_PASS};

    enum BlendFunc   {BLEND_SRC_ALPHA,   BLEND_ONE_MINUS_SRC_ALPHA, BLEND_ONE,
                      BLEND_ZERO, BLEND_SRC_COLOR,  BLEND_DST_COLOR,  
                      BLEND_ONE_MINUS_SRC_COLOR};

    enum StencilOp   {STENCIL_INCR_WRAP, STENCIL_DECR_WRAP,
                      STENCIL_KEEP,      STENCIL_INCR,     STENCIL_DECR,
                      STENCIL_REPLACE,   STENCIL_ZERO,     STENCIL_INVERT};

    enum CullFace    {CULL_FRONT,        CULL_BACK,           CULL_NONE};

    enum ShadeMode   {SHADE_FLAT,        SHADE_SMOOTH};

    enum CombineMode {TEX_REPLACE, TEX_INTERPOLATE, TEX_ADD, TEX_MODULATE, 
                      TEX_BLEND};

    /**
     Call to begin the rendering frame.
     */
    void beginFrame();

    /**
     Call to end the current frame and swap buffers.
     */
    void endFrame();

    /**
     Returns an estimate of the number of frames rendered per second.
     The result is smoothed using an exponentially weighted moving
     average filter so it is robust to unequal frame rendering times.
     */
    double getFrameRate() const;

    /**
     Returns an estimate of the triangles rendered per second.  The
     result is smoothed using an exponentially weighted moving average
     filter.
     */
    double getTriangleRate() const;

    /**
     Returns an estimate of the triangles rendered per frame.  The
     result is smoothed using an exponentially weighted moving average
     filter.
     */
    double getTrianglesPerFrame() const;

    /**
     Use ALWAYS_PASS to shut off testing.
     */
    void setDepthTest(DepthTest test);
    void setStencilTest(StencilTest test);

    void setRenderMode(RenderMode mode);

    /**
     Sets the constant used in the stencil test and operation (if op == STENCIL_REPLACE)
     */
    void setStencilConstant(int reference);
    void setAlphaTest(AlphaTest test, double reference);

    void setDepthRange(double low, double high);

    void enableColorWrite();
    void disableColorWrite();

    void enableAlphaWrite();
    void disableAlphaWrite();

    void enableDepthWrite();
    void disableDepthWrite();

    /**
     Equivalent to glShadeModel
     */
    void setShadeMode(ShadeMode s);

    /**
     If wrapping is not supported on the device, the nearest mode is
     selected.  Unlike OpenGL, stencil writing and testing are
     independent. You do not need to enable the stencil test to use
     the stencil op.

     Use KEEP, KEEP, KEEP to disable stencil writing.  Equivalent to a
     combination of glStencilTest, glStencilFunc, and glStencilOp.


     If there is no depth buffer, the depth test always passes.  If there
     is no stencil buffer, the stencil test always passes.
     */
    void setStencilOp(
        StencilOp                       fail,
        StencilOp                       zfail,
        StencilOp                       zpass);

    /**
     When RenderDevice::supportsTwoSidedStencil is true, separate
     stencil operations can be used for front and back faces.  This
     is useful for rendering shadow volumes.
     */
    void setStencilOp(
        StencilOp                       frontStencilFail,
        StencilOp                       frontZFail,
        StencilOp                       frontZPass,
        StencilOp                       backStencilFail,
        StencilOp                       backZFail,
        StencilOp                       backZPass);

    /**
     Use BLEND_ZERO, BLEND_ONE to shut off blending.
     Equivalent to glBlendFunc.
     */
    void setBlendFunc(
        BlendFunc                       src,
        BlendFunc                       dst);

    /**
     Equivalent to glLineWidth.
     */
    void setLineWidth(
        double                          width);

    /**
     Equivalent to glPointSize.
     */
    void setPointSize(
        double                          diameter);

    /**
     This is not the OpenGL MODELVIEW matrix: it is a matrix that maps
     object space to world space.  The actual MODELVIEW matrix
     is cameraToWorld.inverse() * objectToWorld.  You can retrieve it
     with getModelViewMatrix.
     */
    void setObjectToWorldMatrix(
        const CoordinateFrame&          cFrame);

    CoordinateFrame getObjectToWorldMatrix() const;

    /**
     See RenderDevice::setObjectToWorldMatrix.
     */
    void setCameraToWorldMatrix(
        const CoordinateFrame&          cFrame);

    CoordinateFrame getCameraToWorldMatrix() const;

    Matrix4 getProjectionMatrix() const;

    /**
     cameraToWorld.inverse() * objectToWorld
     */
    CoordinateFrame getModelViewMatrix() const;

    /**
     projection() * cameraToWorld.inverse() * objectToWorld
     */
    Matrix4 getModelViewProjectionMatrix() const;


    /**
    To set a typical 3D perspective matrix, use either
     <CODE>renderDevice->setProjectionMatrix(Matrix4::perspectiveProjection(...)) </CODE>
     or call setProjectionAndCameraMatrix.
     */
    void setProjectionMatrix(const Matrix4& P);

    /**
     m is a 16-element matrix in row major order for multiplying
     texture coordinates:

     v' = M v

     All texture operations check textureUnit against the number of
     available texture units when in debug mode.

     Equivalen to glMatrixMode(GL_TEXTURE); glLoadMatrix(...);
     */
    void setTextureMatrix(
        uint                    textureUnit,
        const double*           m);

    void setTextureMatrix(
        uint                    textureUnit,
        const CoordinateFrame&  c);

    /**
     The matrix returned may not be the same as the
     underlying hardware matrix-- the y-axis is flipped
     in hardware when a texture with invertY = true is specified.
     */
    Matrix4 getTextureMatrix(uint textureUnit);

    /**
     The combine mode specifies how to combine the result of a texture
     lookup with the accumulated fragment value (e.g. the output of
     the previous combine or the constant color for the first texture
     unit).

     The initial combine op is TEX_MODULATE 
     Equivalent to glTexEnvn.
     */
    void setTextureCombineMode(
        uint                      textureUnit,
        const CombineMode         texCombine);


    /**
     Resets the matrix, texture, combine op, and constant for a texture unit.
     */
    void resetTextureUnit(
        uint                      textureUnit);

    /**
     Equivalent to glPolygonOffset
     */
    void setPolygonOffset(
        double                  offset);

    /**
     Set the vertex color (equivalent to glColor).
     */
    void setColor(const Color4& color);
    void setColor(const Color3& color);

    /**
     Equivalent to glNormal
     */
    void setNormal(const Vector3& normal);

    /**
     Equivalent to glTexCoord
     */
    void setTexCoord(uint textureUnit, const Vector4& texCoord);
    void setTexCoord(uint textureUnit, const Vector3& texCoord);
    void setTexCoord(uint textureUnit, const Vector3int16& texCoord);
    void setTexCoord(uint textureUnit, const Vector2& texCoord);
    void setTexCoord(uint textureUnit, const Vector2int16& texCoord);
    void setTexCoord(uint textureUnit, double texCoord);

    /**
     Equivalent to glCullFace
     */
    void setCullFace(CullFace f);

    /**
     Number that the use must multiply all light intensities by 
     to account for the device's brightness.
     */
    inline double getBrightScale() const {
        return brightScale;
    }

    /**
     (0, 0) is the <B>upper</B>-left corner of the screen.
     */
    void setViewport(const Rect2D& v);
    Rect2D getViewport() const;

    /**
     Vertices are "sent" rather than "set" because they
     cause action.
     */
    void sendVertex(const Vector2& vertex);
    void sendVertex(const Vector3& vertex);
    void sendVertex(const Vector4& vertex);

    void setProjectionAndCameraMatrix(const class GCamera& camera);

    /**
     Analogous to glBegin.  See the example in the detailed description
     section of this page.
     */
    void beginPrimitive(Primitive p);

    /**
     Analogous to glEnd.  See the example in the detailed description
     section of this page.
     */
    void endPrimitive();

	void beginIndexedPrimitives();
	void endIndexedPrimitives();

    /** The vertex, normal, color, and tex coord arrays need not come from
        the same VARArea. 

        The format of a VAR array is restricted depending on its use.  The
        following table (from http://oss.sgi.com/projects/ogl-sample/registry/ARB/vertex_program.txt)
        shows the underlying OpenGL restrictions:

     <PRE>

                                       Normal    
      Command                 Sizes    ized?   Types
      ----------------------  -------  ------  --------------------------------
      VertexPointer           2,3,4     no     short, int, float, double
      NormalPointer           3         yes    byte, short, int, float, double
      ColorPointer            3,4       yes    byte, ubyte, short, ushort,
                                               int, uint, float, double
      IndexPointer            1         no     ubyte, short, int, float, double
      TexCoordPointer         1,2,3,4   no     short, int, float, double
      EdgeFlagPointer         1         no     boolean
      VertexAttribPointerARB  1,2,3,4   flag   byte, ubyte, short, ushort,
                                               int, uint, float, double
      WeightPointerARB        >=1       yes    byte, ubyte, short, ushort,
                                               int, uint, float, double
      VertexWeightPointerEXT  1         n/a    float
      SecondaryColor-         3         yes    byte, ubyte, short, ushort,
        PointerEXT                             int, uint, float, double
      FogCoordPointerEXT      1         n/a    float, double
      MatrixIndexPointerARB   >=1       no     ubyte, ushort, uint

      Table 2.4: Vertex array sizes (values per vertex) and data types.  The
      "normalized" column indicates whether fixed-point types are accepted
      directly or normalized to [0,1] (for unsigned types) or [-1,1] (for
      singed types). For generic vertex attributes, fixed-point data are
      normalized if and only if the <normalized> flag is set.

  </PRE>
    
    */
	void setVertexArray(const class VAR& v);
	void setNormalArray(const class VAR& v);
	void setColorArray(const class VAR& v);
	void setTexCoordArray(unsigned int unit, const class VAR& v);

    /**
     Vertex attributes are a generalization of the various per-vertex
     attributes that relaxes the format restrictions.  There are at least
     16 attributes on any card (some allow more).  These attributes have
     special meaning under the fixed function pipeline, as follows:

    <PRE>
    Generic
    Attribute   Conventional Attribute       Conventional Attribute Command
    ---------   ------------------------     ------------------------------
         0      vertex position              Vertex
         1      vertex weights 0-3           WeightARB, VertexWeightEXT
         2      normal                       Normal
         3      primary color                Color
         4      secondary color              SecondaryColorEXT
         5      fog coordinate               FogCoordEXT
         6      -                            -
         7      -                            -
         8      texture coordinate set 0     MultiTexCoord(TEXTURE0, ...)
         9      texture coordinate set 1     MultiTexCoord(TEXTURE1, ...)
        10      texture coordinate set 2     MultiTexCoord(TEXTURE2, ...)
        11      texture coordinate set 3     MultiTexCoord(TEXTURE3, ...)
        12      texture coordinate set 4     MultiTexCoord(TEXTURE4, ...)
        13      texture coordinate set 5     MultiTexCoord(TEXTURE5, ...)
        14      texture coordinate set 6     MultiTexCoord(TEXTURE6, ...)
        15      texture coordinate set 7     MultiTexCoord(TEXTURE7, ...)
       8+n      texture coordinate set n     MultiTexCoord(TEXTURE0+n, ...)
    </PRE>

      @param normalize If true, the coordinates are forced to a [0, 1] scale
    */
    void setVertexAttribArray(unsigned int attribNum, const class VAR& v, bool normalize);

    /**
     Draws the specified kind of primitive from the current vertex array.
     */
	template<class T>
	void sendIndices(RenderDevice::Primitive primitive, int numIndices, 
                     const T* index) {
		
        internalSendIndices(primitive, sizeof(T), numIndices, index);

        // Mark all active arrays as busy.
        setVARAreaMilestone();

		countPrimitive(primitive, numIndices);
	}

    /**
     Draws the specified kind of primitive from the current vertex array.
     */
	template<class T>
	void sendIndices(RenderDevice::Primitive primitive, 
                     const Array<T>& index) {
		sendIndices(primitive, index.size(), index.getCArray());
	}

    void setStencilClearValue(int s);
    void setDepthClearValue(double d);
    void setColorClearValue(const Color4& c);

    /**
     Devices with more textures than texture units (e.g. GeForceFX)
     
     @param textureUnit >= 0
     @param texture Set to NULL to disable the unit
     */
    void setTexture(
        uint                textureUnit,
        TextureRef          texture);

    /** Returns the number of textures available.  May be higher
        than the number of texture units if the programmable
        pipeline provides more textures than the fixed function
        pipeline.*/
    uint numTextures() const;

    /** Returns the number of texture units 
        (texture + reg combiner + matrix) available.
        This only applies to the fixed function pipeline.
    */
    uint numTextureUnits() const;

    /** Returns the number of texture coordinates allowed.
        This may be greater than the number of texture matrices.*/
    uint numTextureCoords() const;

    /**
     Automatically enables vertex programs when they are set. 
     Assumes supportsVertexProgram() is true.
     @param vp Set to NULL to use the fixed function pipeline.
     */
    void setVertexProgram(const VertexProgramRef& vp);

    /**
     Sets vertex program arguments for vertex programs outputted by Cg.
     The names of arguments are read from comments.

     <PRE>
        ArgList args;
        args.set("MVP", renderDevice->getModelViewProjection());
        args.set("height", 3);
        args.set("viewer", Vector3(1, 2, 3));
        renderDevice->setVertexProgram(toonShadeVP, args);
     </PRE>


     @param args must include *all* arguments or an assertion will fail
     */
    void setVertexProgram(const VertexProgramRef& vp,
                          const GPUProgram::ArgList& args);

    /**
     (Automatically enables pixel programs when they are set.) 
     Assumes GPUProgram() is true.
     @param pp Set to NULL to use the fixed function pipeline.
     */
    void setPixelProgram(const PixelProgramRef& pp);

    /**
     It is recommended to call RenderDevice::pushState immediately before
     setting the pixel program, since the arguments can affect texture
     state that will only be restored with RenderDevice::popState.
     */
    void setPixelProgram(const PixelProgramRef& pp,
                         const GPUProgram::ArgList& args);
    
    /**
      Reads a depth buffer value (1 @ far plane, 0 @ near plane) from
      the given screen coordinates (x, y) where (0,0) is the top left
      corner of the width x height screen.  Result is undefined for x, y not
      on screen.

      The result is sensitive to the projection and camera to world matrices.
     */
    double getDepthBufferValue(
        int                 x,
        int                 y) const;


    /**
     Description of the graphics card and driver version.
     */
    std::string getCardDescription() const;


private:
    /** Call after vendor is set */
    std::string getDriverVersion();

    /**
     For performance, we don't actually unbind a texture when
     turning off a texture unit, we just disable it.  If it 
     is enabled with the same texture, we've saved a swap.
    */
    uint32               currentlyBoundTexture[MAX_TEXTURE_UNITS];

    /**
     Snapshot of the state maintained by the render device.
     */
    // WARNING: if you add state, you must initialize it in
    // the constructor and RenderDevice::init and set it in
    // setState().
    class RenderState {
    public:
        class TextureUnit {
        public:
            Vector4             texCoord;

            /** NULL if not bound */
            TextureRef          texture;
            double              textureMatrix[16];
            CombineMode         combineMode;

            TextureUnit();
        };


        Rect2D                      viewport;

        GLight                      light[MAX_LIGHTS];
        bool                        lightEnabled[MAX_LIGHTS];
        bool                        depthWrite;
        bool                        colorWrite;
        bool                        alphaWrite;

        DepthTest                   depthTest;
        StencilTest                 stencilTest;
        int                         stencilReference;
        AlphaTest                   alphaTest;
        double                      alphaReference;

        int                         stencilClear;
        double                      depthClear;
        Color4                      colorClear;               

        CullFace                    cullFace;

        StencilOp                   frontStencilFail;
        StencilOp                   frontStencilZFail;
        StencilOp                   frontStencilZPass;
        StencilOp                   backStencilFail;
        StencilOp                   backStencilZFail;
        StencilOp                   backStencilZPass;
        
        BlendFunc                   srcBlendFunc;
        BlendFunc                   dstBlendFunc;
        
        ShadeMode                   shadeMode;
    
        double                      polygonOffset;

        RenderMode                  renderMode;

        double                      specular;
        double                      shininess;

        double                      lowDepthRange;
        double                      highDepthRange;

        VertexProgramRef            vertexProgram;
        PixelProgramRef             pixelProgram;

        // Ambient light level
        Color4                      ambient;

        double                      lineWidth;
        double                      pointSize;

        bool                        lighting;
        Color4                      color;
        Vector3                     normal;
        TextureUnit                 textureUnit[MAX_TEXTURE_UNITS];
    
        CoordinateFrame             objectToWorldMatrix;
        CoordinateFrame             cameraToWorldMatrix;

        Matrix4                     projectionMatrix;

        RenderState(int width = 1, int height = 1);

    };

    GLint toGLStencilOp(RenderDevice::StencilOp op) const;

    /**
     True between beginPrimitive and endPrimitive
     */
    bool                            inPrimitive;

	bool						    inIndexedPrimitive;

    int                             _numTextureUnits;

    int                             _numTextures;

    int                             _numTextureCoords;

    /**
     Called from the various setXLight functions.
     @param force When true, OpenGL state is changed
     regardless of whether RenderDevice thinks it is up to date.
     */
    void setLight(int i, const GLight* light, bool force);

    /**
     Current render state.
     */
    RenderState                     state;

    /**
     Old render states
     */
    Array<RenderState>              stateStack;

    void setState(
        const RenderState&          newState);

    /**
     Amount to multiply colors by due to gamma.
     */
    double                          brightScale;

    bool                            _initialized;

    RenderDeviceSettings            settings;

public:

    bool supportsOpenGLExtension(const std::string& extension) const;

    /**
      When true, the 6-argument version of RenderDevice::setStencilOp
      can set the front and back operations to different values.
      */
    bool supportsTwoSidedStencil() const {
        return _supportsTwoSidedStencil;
    }

    /**
     When true, Texture::DIM_2D_RECT textures can be created.
     */
    bool supportsTextureRectangle() const {
        return textureRectangleSupported;
    }

    bool supportsVertexProgram() const {
        return _supportsVertexProgram;
    }

    /**
     When true, NVIDIA Vertex Program 2.0 vertex programs can
     be loaded by VertexProgram.
     */
    bool supportsVertexProgramNV2() const {
        return _supportsNVVertexProgram2;
    }

    bool supportsPixelProgram() const {
        return _supportsFragmentProgram;
    }

    /**
     When true, VAR arrays will be in video, not main memory,
     and much faster.
     */
    bool supportsVertexBufferObject() const { 
        return _supportsVertexBufferObject;
    }

    /**
     Returns a value that you should DIVIDE light intensities by
     based on the gamma.  This is automatically handled if you
     use setLight()
     */
    inline double getLightSaturation() const {
        return lightSaturation;
    }

    void push2D();

    /**
     Set up for traditional 2D rendering (origin = upper left, y increases downwards).
     */
    void push2D(const Rect2D& viewport);
    void pop2D();

    /**
     It is often useful to keep track of the number of polygons
     rendered in a scene for profiling purposes.
     */
    int                 polygonCount;
 
    bool init(const RenderDeviceSettings& settings, class Log* log = NULL);

    /** Returns true after RenderDevice::init has been called. */
    bool initialized() const;

	/**
	 Shuts down the system.  This should be the last call you make.
	 */
    void cleanup();

    /**
     Set the titlebar caption
     */
    void setCaption(const std::string& caption);

    /**
     Takes a JPG screenshot of the front buffer and saves it to a file.
     Returns the name of the file that was written.
     Example: renderDevice->screenshot("screens/"); 
     */
    std::string screenshot(const std::string& filepath) const;

    /**
     Notify RenderDevice that the window size has changed.  
     Called in response to a user resize event:
     <PRE>
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch(event.type) {
            case SDL_VIDEORESIZE:
                {
                    renderDevice->notifyResize(event.resize.w, event.resize.h);
                    Rect2D full(0, 0, renderDevice->getWidth(), renderDevice->getHeight());
                    renderDevice->setViewport(full);
                }
                break;
            }
        }

     </PRE>
     */
    void notifyResize(int w, int h);

    /**
     Takes a screenshot of the front buffer and
     puts the data into the G3D::GImage dest variable.
     */
    void screenshotPic(GImage& dest) const;

	/**
	 Pixel dimensions of the OpenGL window interior
	 */
    int getWidth() const;

	/**
	 Pixel dimensions of the OpenGL window interior
	 */
    int getHeight() const;


	inline int getStencilBitDepth() const {
		return stencilBits;
	}


	inline int getZBufferBitDepth() const {
		return depthBits;
	}

    /**
     You must also enableLighting.  Ambient light is handled separately.
     Lighting is automatically adjusted to the lightSaturation value.

     Lights are specified in <B>world space</B>-- they are not affected
     by the camera or object matrix.  Unlike OpenGL, you do not need to
     reset lights after you change the camera matrix.

     setLight(i, NULL) disables a light.
     */
    void setLight(int num, const GLight& light);
    void setLight(int num, void*);

    /**
     Sets the current specular coefficient used in the lighting equation.
     Should be on the range 0 (perfectly diffuse) to 1 (bright specular
     highlight).
     */
    void setSpecularCoefficient(double s);

    /**
     Sets the current shininess exponent used in the lighting equation.
     On the range 0 (large highlight) to 255 (tiny, focussed highlight).
     */
    void setShininess(double s);

    /**
     You must also RenderDevice::enableLighting.
     */
    void setAmbientLightColor(
        const Color3&        color);

    void setAmbientLightColor(
        const Color4&        color);

    /**
     Equivalent to glEnable(GL_LIGHTING).

     On initialization, RenderDevice configures the color material as follows
     (it will be this way unless you change it):
     <PRE>
      float spec[] = {1.0f, 1.0f, 1.0f, 1.0f};
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec);
      glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10);
      glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
      glEnable(GL_COLOR_MATERIAL);
     </PRE>
     */
    void enableLighting();
    void disableLighting();

    /**
     Multiplies v by the current object to world and world to camera matrices,
     then by the projection matrix to obtain a 2D point and z-value.  
     
     The result is the 2D position to which the 3D point v corresponds.  You
     can use this to make results rendered with push2D() line up with those
     rendered with a 3D transformation.
     */
    Vector4 project(const Vector4& v) const;
    Vector4 project(const Vector3& v) const;


    /**
     Returns a new Milestone that can be passed to setMilestone and waitForMilestone.
     Milestones are garbage collected.
     */
    MilestoneRef createMilestone(const std::string& name);

    /**
     Inserts a milestone into the GPU processing list.  You can later call
     waitForMilestone to force the CPU to stall until the GPU has reached
     this milestone.
     <P>
     A milestone may be set multiple times, even without waiting for it in between.
     There is no requirement that a milestone be waited for once set.  Milestone
     setting transcends and is not affected by pushState()/popState() or beginFrame()/endFrame().
     */
    void setMilestone(const MilestoneRef& m);

    /**
     Stalls the CPU until the GPU has finished the milestone.  It is an error
     to wait for a milestone that was not set since it was last waited for.
     */
    void waitForMilestone(const MilestoneRef& m);

    /**
     Call within RenderDevice::pushState()...popState() so that you can
     restore the texture coordinate generation

     @param lightMVP The modelview projection matrix that
            was used to render the shadow map originally
            (you can get this from RenderDevice::getModelViewProjectionMatrix() 
            while rendering the shadow map).
     @param textureUnit The texture unit to use for shadowing. 0...RenderDevice::numTextureUnits()
            That unit cannot be used for texturing at the same time.
     */
    void configureShadowMap(
        uint                textureUnit,
        const Matrix4&      lightMVP,
        const TextureRef&   shadowMap);

    /**
     Call within RenderDevice::pushState()...popState() so that you can
     restore the texture coordinate generation.  Note that you can 
     obtain the reflection texture (aka environment map) from G3D::Sky
     or by loading it with G3D:Texture::fromFile.

     @param textureUnit The texture unit to use for shadowing. 0...RenderDevice::numTextureUnits()
            That unit cannot be used for texturing at the same time.
     */

    void configureReflectionMap(
        uint                textureUnit,
        TextureRef          reflectionTexture);

    #ifdef G3D_WIN32
        HDC getWindowHDC() const;
    #endif
};

} // namespace

#endif
