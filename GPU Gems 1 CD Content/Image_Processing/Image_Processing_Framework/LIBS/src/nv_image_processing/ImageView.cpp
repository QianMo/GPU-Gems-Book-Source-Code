// -----------------------------------------------------------------------------
//
// Contents:
//      class ImageView
//
// Description:
//      ImageWiew draws output from ImageOperators into OpenGL viewports. 
//          ImageView provides methods to register and unrester ImageOperators,
//      position and scale the output image, and to adapt to varying viewport 
//      sizes.
//          ImageView is part of a model-view-controller pattern. The model
//      being ImageOperator. Since ImageOperators are composits the actual
//      model is rather a complex hierarchy of interconnected ImageOperators
//      and ImageFilters than a single object.
//
// Author:
//      Frank Jargstorff (01/21/2003)
//
// -----------------------------------------------------------------------------


//
// Include 
//

#include "ImageView.h"
#include "SourceOperator.h"
#include "ImageFilter.h"
#include "ShaderManager.h"

#include "AssertGL.h"
#include "AssertCG.h"


#ifdef _WIN32
    #include <windows.h>
#endif

#include <GL/glu.h>

#include <assert.h>

#include <iostream>


//
// Defines
//

#define INITIAL_WINDOW_WIDTH   600
#define INITIAL_WINDOW_HEIGHT  400



// -----------------------------------------------------------------------------
// ImageView class
//


// 
// Public static constant initialization
//

        // aDefaultBackgroundColor
        //
        // Description:
        //      This constant contains the default background color.
        //      On contruction every ImageView instance will clear
        //      the frame buffer using this default color.
        //      The color is a dark gray (0.3 Luminance).
        //
        const 
        float 
ImageView::aDefaultBackgroundColor[3] = {0.0f, 0.3f, 0.3f};


//
// Construction and destruction
//

        // Default constructor
        //
        // Description:
        //
ImageView::ImageView(): _nViewportWidth(INITIAL_WINDOW_WIDTH)
                      , _nViewportHeight(INITIAL_WINDOW_HEIGHT)
                      , _nZoomFactor(1.0f)
                      , _nBrightness(1.0f)
                      , _nImagePositionX(0)
                      , _nImagePositionY(0)
{
    for (int i=0; i<3; i++)
        _aBackgroundColor[i] = aDefaultBackgroundColor[i];

                                // Set up the vertex program
    cgGLEnableProfile(ShaderManager::gVertexIdentityProfile);
    CG_ASSERT_NO_ERROR;
    cgGLBindProgram(ShaderManager::gVertexIdentityShader);
    CG_ASSERT_NO_ERROR;

                                // Set up the fragment program
    _oCgFragmentProgram = cgCreateProgramFromFile(ShaderManager::gCgContext, CG_SOURCE, 
                                                 CG_SOURCE_PATH "Texture.cg",
                                                 CG_PROFILE_FP30, 0, 0);
    CG_ASSERT_NO_ERROR;
    cgGLLoadProgram(_oCgFragmentProgram);
    cgGLEnableProfile(CG_PROFILE_FP30);
    cgGLBindProgram(_oCgFragmentProgram);
    CG_ASSERT_NO_ERROR;
                                // Grab the necessary parameters
    _hImage          = cgGetNamedParameter(_oCgFragmentProgram,  "oImage"    );
    CG_ASSERT_NO_ERROR;
}


//
// Public methods
//

        // recenter
        //
        // Description:
        //      Recenters the image in the viewport for the next
        //      display cycle.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None
        //
        void
ImageView::center()
{
    _nImagePositionX = ( _nViewportWidth  - _oImage.width()  ) / 2;
    _nImagePositionY = ( _nViewportHeight - _oImage.height() ) / 2;
}

        // imagePositionX
        //
        // Description:
        //      Image posion on the drawing plane.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      The x-coordinate of the image's lower left corner
        //      within the view's window (with window's origin in the
        //      lower left corner, also).
        //
        int
ImageView::imagePositionX()
        const
{
    return _nImagePositionX;
}

        // imagePositionY
        //
        // Description:
        //      Image posion on the drawing plane.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      The y-coordinate of the image's lower left corner
        //      within the view's window (with window's origin in the
        //      lower left corner, also).
        //
        int
ImageView::imagePositionY()
        const
{
    return _nImagePositionY;
}

        // setImagePosition
        //
        // Description:
        //      Places the image at a new position.
        //
        // Parameters:
        //      nX - The new x-coordinate.
        //      nY - The new y-coordinate.
        //
        // Returns:
        //      None
        //
        void
ImageView::setImagePosition(int nX, int nY)
{
    _nImagePositionX = nX;
    _nImagePositionY = nY;
}



        // zoomFactor
        //
        // Description:
        //      Get the zoom factor.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      The zoom factor.
        //
        float
ImageView::zoomFactor()
        const
{
    return _nZoomFactor;
}

        // setZoomFactor
        //
        // Description:
        //      Set the zoom factor.
        //
        // Parameters:
        //      nZoomFactor - the new zoom factor.
        //
        // Returns:
        //      None
        //
        void
ImageView::setZoomFactor(float nZoomFactor)
{
    _nZoomFactor = nZoomFactor;
}

        // changeZoomFactor
        //
        // Description:
        //      Change zoom factor by a given percentage.
        //
        // Parametes:
        //      nPercent - the increase or decrease in percent
        //          of the original image size.
        //
        // Returns:
        //      None
        //
        void
ImageView::changeZoomFactor(float nPercent)
{
    _nZoomFactor += nPercent / 100.0f;
}

        // reshape
        //
        // Description:
        //      Reshape the view to reflect new vieport dimensions.
        //
        // Parameters:
        //      nViewportWidth  - the new width of the viewport.
        //      nViewportHeight - the new height of the viewport.
        //
        // Returns:
        //      None
        //
        void
ImageView::reshape(int nViewportWidth, int nViewportHeight)
{
                                // "Recenter" the image. In case
                                // the image was moved offcenter
                                // this will move the image in an 
                                // intuitive manner.
    _nImagePositionX += (nViewportWidth  - _nViewportWidth)  / 2;
    _nImagePositionY += (nViewportHeight - _nViewportHeight) / 2;
                                
                                // Now update the viewport info
                                // and the projection matrix.
    _nViewportWidth  = nViewportWidth;
    _nViewportHeight = nViewportHeight;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, _nViewportWidth, 0, _nViewportHeight);
    assert(glGetError() == GL_NO_ERROR);
}



        // display
        //
        // Description:
        //      Renders the registered image to the screen.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None
        void 
ImageView::display()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);       

    glClearColor(_aBackgroundColor[0], 
                 _aBackgroundColor[1], 
                 _aBackgroundColor[2], 
                 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    update();

    float nWidth  = static_cast<float>(_oImage.width());
    float nHeight = static_cast<float>(_oImage.height());

    glMatrixMode(GL_MODELVIEW);
    GL_ASSERT_NO_ERROR;

                            // Place image at its position
    glPushMatrix();
    glLoadIdentity();
    glTranslatef((GLfloat) _nImagePositionX + nWidth  /2, 
                 (GLfloat) _nImagePositionY + nHeight /2, 
                 (GLfloat) 0.0f);
    glScalef((GLfloat)  _nZoomFactor, 
             (GLfloat) -_nZoomFactor, 
             (GLfloat) 1.0f);
    glTranslatef((GLfloat)- nWidth  /2, 
                 (GLfloat)- nHeight /2, 
                 (GLfloat) 0.0f);
    GL_ASSERT_NO_ERROR;

    cgGLEnableProfile(ShaderManager::gVertexIdentityProfile);
    cgGLEnableProfile(CG_PROFILE_FP30);
    CG_ASSERT_NO_ERROR;

    cgGLBindProgram(ShaderManager::gVertexIdentityShader);
    cgGLBindProgram(_oCgFragmentProgram);
    CG_ASSERT_NO_ERROR;

    cgGLSetTextureParameter(_hImage, _oImage.textureID());
    cgGLEnableTextureParameter(_hImage);
    CG_ASSERT_NO_ERROR;

    cgGLSetStateMatrixParameter(ShaderManager::gVertexIdentityModelView, 
                                CG_GL_MODELVIEW_PROJECTION_MATRIX,
                                CG_GL_MATRIX_IDENTITY);
    CG_ASSERT_NO_ERROR;

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.0f,   0.0f);     glVertex3f(  0.0f, 0.0f,    0.0f);
        glTexCoord2f(nWidth, 0.0f);     glVertex3f(nWidth, 0.0f,    0.0f);
        glTexCoord2f(nWidth, nHeight);  glVertex3f(nWidth, nHeight, 0.0f);
        glTexCoord2f(0.0f,   nHeight);  glVertex3f(  0.0f, nHeight, 0.0f);
    }
    glEnd();

    cgGLDisableTextureParameter(_hImage);
    cgGLDisableProfile(ShaderManager::gVertexIdentityProfile);
    cgGLDisableProfile(CG_PROFILE_FP30);
    CG_ASSERT_NO_ERROR;

                            // clean up the stack for more drawing
    glPopMatrix();
    glFinish();
}

        // update
        //
        // Description:
        //      Makes the view get a new image from the pipeline.
        //
        // Parameters:
        //      None
        //
        // Returns:
        //      None        
        //
        void    
ImageView::update()
{
    if (_pSourceOperator)
    {   
        if (_bDirty || _pSourceOperator->dirty())
        {
            _oImage = _pSourceOperator->image();
            _bDirty = false;
        }

        GL_ASSERT_NO_ERROR;
    }
}
