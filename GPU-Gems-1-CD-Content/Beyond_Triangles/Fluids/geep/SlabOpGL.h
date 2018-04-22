  //--------------------------------------------------------------------------
// File : SlabOpGL.h
//----------------------------------------------------------------------------
// Copyright 2003 Mark J. Harris and
//     The University of North Carolina at Chapel Hill
//----------------------------------------------------------------------------
// Permission to use, copy, modify, distribute and sell this software and its 
// documentation for any purpose is hereby granted without fee, provided that 
// the above copyright notice appear in all copies and that both that 
// copyright notice and this permission notice appear in supporting 
// documentation.  Binaries may be compiled with this software without any 
// royalties or restrictions. 
//
// The author(s) and The University of North Carolina at Chapel Hill make no 
// representations about the suitability of this software for any purpose. 
// It is provided "as is" without express or implied warranty.
/**
 * @file SlabOpGL.h
 * 
 * SlabOp Policies for OpenGL.
 */
#ifndef __SLABOPGL_H__
#define __SLABOPGL_H__

#include <memory.h>
#include <math.h>

namespace geep
{

//----------------------------------------------------------------------------
/**
 * @class SimpleGLComputePolicy
 * @brief Handles SlabOp rendering (a quad) with a single texture coordinate.
 */
class SimpleGLComputePolicy
{
public:

  //--------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetSlabRect(float minX, float minY, float maxX, float maxY, float z)
   * @brief Sets the dimensions of the quad to render.
   */ 
  void SetSlabRect(float minX, float minY, float maxX, float maxY, float z = 0)
  {
    _rMinX = minX; _rMaxX = maxX;
    _rMinY = minY; _rMaxY = maxY;
    _rZ = z;
  }

protected:  // methods

  SimpleGLComputePolicy() : _rMinX(0), _rMinY(0), _rMaxX(1), _rMaxY(1), _rZ(0)
  {
  }

  ~SimpleGLComputePolicy() {}

  //--------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn Compute()
   * @brief Performs the slab computation.  Called by SlabOp::Compute()
   */ 
  void Compute()
  {
    glBegin(GL_QUADS);
    {
      glVertex3f(_rMinX, _rMinY, _rZ);
      glVertex3f(_rMaxX, _rMinY, _rZ);
      glVertex3f(_rMaxX,  _rMaxY, _rZ);
      glVertex3f(_rMinX,  _rMaxY, _rZ);
    }
    glEnd();
  }

protected: // data
  float _rMinX, _rMinY;
  float _rMaxX, _rMaxY;
  float _rZ;
};

//----------------------------------------------------------------------------
/**
 * @class SingleTextureGLComputePolicy
 * @brief Handles SlabOp rendering (a quad) with a single texture coordinate.
 */
class SingleTextureGLComputePolicy
{
public:

  //--------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetSlabRect(float minX, float minY, float maxX, float maxY)
   * @brief Sets the dimensions of the quad to render.
   */ 
  void SetSlabRect(float minX, float minY, float maxX, float maxY)
  {
    _rMinX = minX; _rMaxX = maxX;
    _rMinY = minY; _rMaxY = maxY;
  }

  //--------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetTexCoordRect(int texUnit, float minS, float minT, float maxS, float maxT)
   * @brief Sets the texture coordinate extents.
   */ 
  void SetTexCoordRect(float minS, float minT, float maxS, float maxT)
  { 
    _rMinS = minS; _rMaxS = maxS;
    _rMinT = minT; _rMaxT = maxT;
  }

protected:  // methods

  SingleTextureGLComputePolicy() : _rMinX(0), _rMinY(0), _rMaxX(1), _rMaxY(1) 
  {
    SetTexCoordRect(0, 0, 1, 1);
  }

  virtual ~SingleTextureGLComputePolicy() {}

  //--------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn Compute()
   * @brief Performs the slab computation.  Called by SlabOp::Compute()
   */ 
  void Compute()
  {
    glBegin(GL_QUADS);
    {
      glTexCoord2f(_rMinS, _rMinT); glVertex2f(_rMinX, _rMinY);
      glTexCoord2f(_rMaxS, _rMinT); glVertex2f(_rMaxX, _rMinY);
      glTexCoord2f(_rMaxS, _rMaxT); glVertex2f(_rMaxX,  _rMaxY);
      glTexCoord2f(_rMinS, _rMaxT); glVertex2f(_rMinX,  _rMaxY);
    }
    glEnd();
  }

protected: // data
  float _rMinX, _rMinY;
  float _rMaxX, _rMaxY;
  float _rMinS, _rMaxS;
  float _rMinT, _rMaxT;
};

//----------------------------------------------------------------------------
/**
 * @class MultiTextureGLComputePolicy
 * @brief Handles SlabOp rendering (a quad) with multiple texture coordinates.
 */
class MultiTextureGLComputePolicy
{
public:

  //--------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetSlabRect(float minX, float minY, float maxX, float maxY)
   * @brief Sets the dimensions of the quad to render.
   */ 
  void SetSlabRect(float minX, float minY, float maxX, float maxY)
  {
    _rMinX = minX; _rMaxX = maxX;
    _rMinY = minY; _rMaxY = maxY;
  }

  //--------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetTexCoordRect(int texUnit, float minS, float minT, float maxS, float maxT)
   * @brief Sets the texture coordinate extents.
   */ 
  void SetTexCoordRect(int texUnit, 
                     float minS, float minT, float maxS, float maxT)
  { 
    _rMinS[texUnit] = minS; _rMaxS[texUnit] = maxS;
    _rMinT[texUnit] = minT; _rMaxT[texUnit] = maxT;
  }

protected:  // methods

  MultiTextureGLComputePolicy() : _rMinX(0), _rMinY(0), _rMaxX(1), _rMaxY(1) 
  {
    glGetIntegerv(GL_MAX_TEXTURE_UNITS_ARB, &_iNumTextureUnits);
    _rMinS = new float[_iNumTextureUnits];
    _rMaxS = new float[_iNumTextureUnits];
    _rMinT = new float[_iNumTextureUnits];
    _rMaxT = new float[_iNumTextureUnits];

    for (int i = 0; i < _iNumTextureUnits; ++i)
    {
      SetTexCoordRect(i, 0, 0, 1, 1);
    }
  }

  virtual ~MultiTextureGLComputePolicy() 
  {
    delete [] _rMinS; delete [] _rMaxS;
    delete [] _rMinT; delete [] _rMaxT;
  }

  //--------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn Compute()
   * @brief Performs the slab computation.  Called by SlabOp::Compute()
   */ 
  void Compute()
  {
    glBegin(GL_QUADS);
    {
      int i;
      // LL
      for (i = 0; i < _iNumTextureUnits; ++i)
        glMultiTexCoord2f(GL_TEXTURE0_ARB + i, _rMinS[i], _rMinT[i]);
      glVertex2f(_rMinX, _rMinY);
      // LR
      for (i = 0; i < _iNumTextureUnits; ++i)
        glMultiTexCoord2f(GL_TEXTURE0_ARB + i, _rMaxS[i], _rMinT[i]);
      glVertex2f(_rMaxX, _rMinY);
      // UR
      for (i = 0; i < _iNumTextureUnits; ++i)
        glMultiTexCoord2f(GL_TEXTURE0_ARB + i, _rMaxS[i], _rMaxT[i]);
      glVertex2f(_rMaxX,  _rMaxY);
      // UL
      for (i = 0; i < _iNumTextureUnits; ++i)
        glMultiTexCoord2f(GL_TEXTURE0_ARB + i, _rMinS[i], _rMaxT[i]);
      glVertex2f(_rMinX,  _rMaxY);
    }
    glEnd();
  }

protected: // data
  int   _iNumTextureUnits;
  float _rMinX,  _rMinY;
  float _rMaxX,  _rMaxY;
  float *_rMinS, *_rMaxS;
  float *_rMinT, *_rMaxT;
};


//----------------------------------------------------------------------------
/**
 * @class BoundaryGLComputePolicy
 * @brief Renders only the boundaries (one texel width) of a slab.
 */
class BoundaryGLComputePolicy
{
public:

  //--------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetSlabRect(float minX, float minY, float maxX, float maxY)
   * @brief Sets the dimensions of the quad to render.
   */ 
  void SetSlabRect(float minX, float minY, float maxX, float maxY)
  {
    _rMinX = minX; _rMaxX = maxX;
    _rMinY = minY; _rMaxY = maxY;
    UpdateCoordinates();
  }

  //--------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetTexCoordRect(int texUnit, float minS, float minT, float maxS, float maxT)
   * @brief Sets the texture coordinate extents.
   */ 
  void SetTexCoordRect(float minS, float minT, float maxS, float maxT)
  { 
    _rMinS = minS; _rMaxS = maxS;
    _rMinT = minT; _rMaxT = maxT;
    UpdateCoordinates();
  }

  //--------------------------------------------------------------------------
  // Function     	  : SetTexResolution
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetTexResolution(int width, int height)
   * @brief Sets the resolution of the texture
   */ 
  void SetTexResolution(int width, int height)
  {
    _iTexResS = width; _iTexResT = height;
    UpdateCoordinates();
  }


  //--------------------------------------------------------------------------
  // Function     	  : EnableSides
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn EnableSides(bool left, bool right, bool bottom, bool top)
   * @brief Individually enables / dsables the four side boundaries.
   */ 
  void EnableSides(bool left, bool right, bool bottom, bool top)
  {
    _bLeft = left; _bRight = right; _bBottom = bottom; _bTop = top;
  }


  //--------------------------------------------------------------------------
  // Function     	  : SetPeriodic
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetPeriodic(bool left, bool right, bool bottom, bool top)
   * @brief Individually sets each boundary as periodic or not.
   */ 
  void SetPeriodic(bool left, bool right, bool bottom, bool top)
  {
    _bLeftPeriodic = left; _bRightPeriodic = right; 
    _bBottomPeriodic = bottom; _bTopPeriodic = top;
  }


protected:  // methods

  BoundaryGLComputePolicy() 
  : _iTexResS(1), _iTexResT(1), _rMinX(0), _rMinY(0), _rMaxX(1), _rMaxY(1),
    _bLeft(true), _bRight(true), _bBottom(true), _bTop(true),
    _bLeftPeriodic(false), _bRightPeriodic(false), 
    _bBottomPeriodic(false), _bTopPeriodic(false)
  {
    SetTexCoordRect(0, 0, 1, 1);
  }

  virtual ~BoundaryGLComputePolicy() {}


  void UpdateCoordinates()
  {
    _rPixelWidth  = (_rMaxX - _rMinX) / (_rMaxS - _rMinS);
    _rPixelHeight = (_rMaxY - _rMinY) / (_rMaxT - _rMinT);
    _rTexelWidth  = (_rMaxS - _rMinS) / _iTexResS;
    _rTexelHeight = (_rMaxT - _rMinT) / _iTexResT;
  }

  //--------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn Compute()
   * @brief Performs the slab computation on the slab boundaries.  Called by SlabOp::Compute()
   * 
   * This renders the four 1-pixel wide boundaries of the slab.
   */ 
  void Compute()
  {
    // texcoord 0 is the base texture coordinate.
    // texcoord 1 is the offset (done here to avoid proliferation of fragment 
    //   programs, or of conditionals inside the fragment program.)

    glBegin(GL_LINES);
    {      
      // left boundary
      if (_bLeft)
      {
        glMultiTexCoord2f(GL_TEXTURE1_ARB, 
                          _bLeftPeriodic ? _iTexResS - 2 * _rTexelWidth : 
                                           _rTexelWidth, 0); // offset amount
        glTexCoord2f(_rMinS, _rMinT); glVertex2f(_rMinX, _rMinY);
        glTexCoord2f(_rMinS, _rMaxT); glVertex2f(_rMinX, _rMaxY);
      }
      // right boundary
      if (_bRight)
      {
        glMultiTexCoord2f(GL_TEXTURE1_ARB, 
                          _bRightPeriodic ? -_iTexResS + 2 * _rTexelWidth : 
                                            -_rTexelWidth, 0); // offset amount
        glTexCoord2f(_rMaxS - _rTexelWidth, _rMinT); 
        glVertex2f(_rMaxX - _rPixelWidth, _rMinY); 
        glTexCoord2f(_rMaxS - _rTexelWidth, _rMaxT); 
        glVertex2f(_rMaxX - _rPixelWidth, _rMaxY);
      }
      // bottom boundary
      if (_bBottom)
      {
        glMultiTexCoord2f(GL_TEXTURE1_ARB, 0, 
                          _bBottomPeriodic ? _iTexResT - 2 * _rTexelHeight : 
                                             _rTexelHeight); // offset amount
        glTexCoord2f(_rMinS, _rMinT); 
        glVertex2f(_rMinX, _rMinY + 0.5f * _rPixelHeight);
        glTexCoord2f(_rMaxS, _rMinT); 
        glVertex2f(_rMaxX, _rMinY + 0.5f * _rPixelHeight);
      }
      // top boundary
      if (_bTop)
      {
        glMultiTexCoord2f(GL_TEXTURE1_ARB, 0, 
                          _bTopPeriodic ? -_iTexResT + 2 * _rTexelHeight : 
                                          -_rTexelHeight); // offset amount
        glTexCoord2f(_rMinS, _rMaxT - _rTexelHeight); 
        glVertex2f(_rMinX, _rMaxY - 0.5f * _rPixelHeight); 
        glTexCoord2f(_rMaxS, _rMaxT - _rTexelHeight); 
        glVertex2f(_rMaxX, _rMaxY - 0.5f * _rPixelHeight);
      }
    }
    glEnd();
    /*glBegin(GL_QUADS);
    {
      // left boundary
      glMultiTexCoord2f(GL_TEXTURE1_ARB, _rTexelWidth, 0); // offset amount
      glTexCoord2f(_rMinS, _rMinT); glVertex2f(_rMinX, _rMinY); 
      glTexCoord2f(_rMinS + _rTexelWidth, _rMinT); glVertex2f(_rMinX + _rPixelWidth, _rMinY);
      glTexCoord2f(_rMinS + _rTexelWidth, _rMaxT); glVertex2f(_rMinX + _rPixelWidth, _rMaxY);
      glTexCoord2f(_rMinS, _rMaxT); glVertex2f(_rMinX, _rMaxY);
      // right boundary
      glMultiTexCoord2f(GL_TEXTURE1_ARB, -_rTexelWidth, 0); // offset amount
      glTexCoord2f(_rMaxS - _rTexelWidth, _rMinT); glVertex2f(_rMaxX - _rPixelWidth, _rMinY); 
      glTexCoord2f(_rMaxS, _rMinT); glVertex2f(_rMaxX, _rMinY);
      glTexCoord2f(_rMaxS, _rMaxT); glVertex2f(_rMaxX, _rMaxY);
      glTexCoord2f(_rMaxS - _rTexelWidth, _rMaxT); glVertex2f(_rMaxX - _rPixelWidth, _rMaxY);
      // bottom boundary
      glMultiTexCoord2f(GL_TEXTURE1_ARB, 0, _rTexelHeight); // offset amount
      glTexCoord2f(_rMinS, _rMinT); glVertex2f(_rMinX, _rMinY); 
      glTexCoord2f(_rMaxS, _rMinT); glVertex2f(_rMaxX, _rMinY);
      glTexCoord2f(_rMaxS, _rMinT + _rTexelHeight); glVertex2f(_rMaxX, _rMinY + _rPixelHeight);
      glTexCoord2f(_rMinS, _rMinT + _rTexelHeight); glVertex2f(_rMinX, _rMinY + _rPixelHeight);
      // top boundary
      glMultiTexCoord2f(GL_TEXTURE1_ARB, 0, -_rTexelHeight); // offset amount
      glTexCoord2f(_rMinS, _rMaxT - _rTexelHeight); glVertex2f(_rMinX, _rMaxY - _rPixelHeight); 
      glTexCoord2f(_rMaxS, _rMaxT - _rTexelHeight); glVertex2f(_rMaxX, _rMaxY - _rPixelHeight);
      glTexCoord2f(_rMaxS, _rMaxT); glVertex2f(_rMaxX, _rMaxY);
      glTexCoord2f(_rMinS, _rMaxT); glVertex2f(_rMinX, _rMaxY);
    }
    glEnd();*/
  }

protected: // data
  float _rPixelWidth, _rPixelHeight;
  float _rTexelWidth, _rTexelHeight;
  int   _iTexResS,    _iTexResT;
  float _rMinX,       _rMinY;
  float _rMaxX,       _rMaxY;
  float _rMinS,       _rMaxS;
  float _rMinT,       _rMaxT;
  float _bLeft, _bRight, _bBottom, _bTop;
  float _bLeftPeriodic, _bRightPeriodic, _bBottomPeriodic, _bTopPeriodic;
};


//----------------------------------------------------------------------------
/**
 * @class CopyTexGLUpdatePolicy
 * @brief Handles the copy to texture needed by some Slab Operations.
 */
class CopyTexGLUpdatePolicy
{
public:

  //--------------------------------------------------------------------------
  // Function     	  : SetOutputTexture
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetOutputTexture(GLuint id, int width, int height)
   * @brief Sets the data needed to perform the texture update.
   */ 
  void SetOutputTexture(GLuint id, int width, int height)
  { 
    _iOutputTexture = id; 
    _iWidth         = width;
    _iHeight        = height;
  }


  //--------------------------------------------------------------------------
  // Function     	  : SetOutputRectangle
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetOutputRectangle(int xOffset, int yOffset, int x, int y, width, height)
   * @brief Specifies the rectangle to copy to texture.
   */ 
  void SetOutputRectangle(int xOffset, int yOffset, 
                          int x, int y, 
                          int width, int height)
  {
    _iXOffset = xOffset; _iYOffset = yOffset;
    _iX = x; _iY = y; 
    _iWidth = width; _iHeight = height;
  }

protected: // methods
  CopyTexGLUpdatePolicy() 
    : _iOutputTexture(0), _iXOffset(0), _iYOffset(0),
      _iX(0), _iY(0), _iWidth(0), _iHeight(0) {}
  virtual ~CopyTexGLUpdatePolicy() {}

  void SetViewport()
  {
    glGetIntegerv(GL_VIEWPORT, _iOldVP);
    glViewport(_iX, _iY, _iWidth, _iHeight);
  }

  //--------------------------------------------------------------------------
  // Function     	  : UpdateOutputSlab
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn UpdateOutputSlab()
   * @brief Updates the output texture.  Called by SlabOp::Compute()
   */ 
  void UpdateOutputSlab()
  {
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, _iOutputTexture);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 
                        0, _iXOffset, _iYOffset, _iX, _iY, _iWidth, _iHeight);
    glViewport(_iOldVP[0], _iOldVP[1], _iOldVP[2], _iOldVP[3]);
  }


protected: // data
  GLuint _iOutputTexture;
  int    _iXOffset;
  int    _iYOffset;
  int    _iX;
  int    _iY;
  int    _iWidth;
  int    _iHeight;

  int    _iOldVP[4];
};

//----------------------------------------------------------------------------
/**
 * @class Copy3DTexGLUpdatePolicy
 * @brief Handles the copy to a 3D texture slice needed by some Slab Operations.
 */
class Copy3DTexGLUpdatePolicy
{
public:

  //--------------------------------------------------------------------------
  // Function     	  : SetOutputTexture
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetOutputTexture(GLuint id, int width, int height)
   * @brief Sets the data needed to perform the texture update.
   */ 
  void SetOutputTexture(GLuint id, int width, int height, int depth)
  { 
    _iOutputTexture = id; 
    _iWidth         = width;
    _iHeight        = height;
    _iDepth         = depth;
  }


  //--------------------------------------------------------------------------
  // Function     	  : SetOutputRectangle
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetOutputRectangle(int xOffset, int yOffset, int zOffset, int x, int y, width, height)
   * @brief Specifies the rectangle to copy to texture slice.
   */ 
  void SetOutputRectangle(int xOffset, int yOffset, int zOffset,
                          int x, int y, 
                          int width, int height)
  {
    _iXOffset = xOffset; _iYOffset = yOffset; _iZOffset = zOffset;
    _iX = x; _iY = y; 
    _iWidth = width; _iHeight = height;
  }

  //--------------------------------------------------------------------------
  // Function     	  : SetOutputSlice
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn SetOutputSlice(int zOffset)
   * @brief Specifies the slice to copy to. 
   * Assumes one prior call to SetOutputRectangle to initialize the region 
   * to copy.
   */ 
  void SetOutputSlice(int zOffset)
  {
    _iZOffset = zOffset;
  }

protected: // methods
  Copy3DTexGLUpdatePolicy() 
    : _iOutputTexture(0), _iXOffset(0), _iYOffset(0), _iZOffset(0),
      _iX(0), _iY(0), _iWidth(0), _iHeight(0), _iDepth(0) {}
  ~Copy3DTexGLUpdatePolicy() {}

  void SetViewport()
  {
    glGetIntegerv(GL_VIEWPORT, _iOldVP);
    glViewport(_iX, _iY, _iWidth, _iHeight);
  }

  //--------------------------------------------------------------------------
  // Function     	  : UpdateOutputSlab
  // Description	    : 
  //--------------------------------------------------------------------------
  /**
   * @fn UpdateOutputSlab()
   * @brief Updates a slice in the output texture. Called by SlabOp::Compute()
   */ 
  void UpdateOutputSlab()
  {
    //glFlush();
    glBindTexture(GL_TEXTURE_3D, _iOutputTexture);
    glCopyTexSubImage3D(GL_TEXTURE_3D, 
                        0, _iXOffset, _iYOffset, _iZOffset, 
                        _iX, _iY, _iWidth, _iHeight);
    glViewport(_iOldVP[0], _iOldVP[1], _iOldVP[2], _iOldVP[3]);
  }


protected: // data
  GLuint _iOutputTexture;
  int    _iXOffset;
  int    _iYOffset;
  int    _iZOffset;
  int    _iX;
  int    _iY;
  int    _iWidth;
  int    _iHeight;
  int    _iDepth;

  int    _iOldVP[4];
};

}; // namespace geep

#endif //__SLABOPGL_H__