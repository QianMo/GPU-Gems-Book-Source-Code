 //-----------------------------------------------------------------------------
// File : Flat3DTextureSlabOpGL.h
//-----------------------------------------------------------------------------
// Copyright 2003 Mark J. Harris and
//     The University of North Carolina at Chapel Hill
//-----------------------------------------------------------------------------
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
 * @file Flat3DTextureSlabOpGL.h
 * 
 * @todo <WRITE FILE DOCUMENTATION>
 */
#ifndef __FLAT3DTEXTURESLABOPGL_H__
#define __FLAT3DTEXTURESLABOPGL_H__

#include <camera.hpp>

//-----------------------------------------------------------------------------
/**
 * @class SingleTextureGLComputePolicy
 * @brief Handles SlabOp rendering (a quad) with a single texture coordinate.
 */
class Flat3DTexComputePolicy
{
public:

  //---------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //---------------------------------------------------------------------------
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

  //---------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //---------------------------------------------------------------------------
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

  //---------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn SetTexCoordRect(int texUnit, float minS, float minT, float maxS, float maxT)
   * @brief Sets the texture coordinate extents.
   */ 
  void SetSliceTexDimensions(int width, int height, int numS, int numT)
  { 
    _iSliceResS = width;
    _iSliceResT = height;
    _iNumS      = numS; 
    _iNumT      = numT;
    _iTexResS   = width * numS; 
    _iTexResT   = height * numT;
    UpdateCoordinates();
  }

protected:  // methods

  Flat3DTexComputePolicy() : _rMinX(0), _rMinY(0), _rMaxX(1), _rMaxY(1) 
  {
    SetTexCoordRect(0, 0, 1, 1);
  }

  ~Flat3DTexComputePolicy() {}

  void UpdateCoordinates()
  {
    _rPixelWidth  = (_rMaxX - _rMinX) / (_rMaxS - _rMinS);
    _rPixelHeight = (_rMaxY - _rMinY) / (_rMaxT - _rMinT);
    _rSliceWidth  = (_rMaxX - _rMinX) / _iNumS;
    _rSliceHeight = (_rMaxY - _rMinY) / _iNumT;

    // input to output texel ratio expected to be integral
    float Swidth = _rMaxS - _rMinS;
    float Twidth = _rMaxT - _rMinT;
    _rTexelWidth  = Swidth / _iTexResS;
    _rTexelHeight = Twidth / _iTexResT;
  }

  //---------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn Compute()
   * @brief Performs the slab computation.  Called by SlabOp::Compute()
   */ 
  void Compute()
  {
    glBegin(GL_QUADS);
    {
      float rLeftX, rRightX;
      float rLeftS, rRightS;
      float rBottomY, rTopY;
      float rBottomT, rTopT;    
    
      rBottomY = _rMinY + _rPixelHeight;
      rTopY    = _rMinY + _rSliceHeight - _rPixelHeight;
      rBottomT = _rMinT + _rTexelHeight; 
      rTopT    = _rMinT + _iSliceResT - _rTexelHeight;

      float slice = 0.5;

      for (int j = 0; j < _iNumT; ++j)
      {
        rLeftX  = _rMinX + _rPixelWidth; 
        rRightX = _rMinX + _rSliceWidth - _rPixelWidth;
        rLeftS  = _rMinS + _rTexelWidth;
        rRightS = _rMinS + _iSliceResS  - _rTexelWidth;

        for (int i = 0; i < _iNumS; ++i)
        {
          if (((0 != (i + j)) && (_iNumS * _iNumT - 1 != i + j * _iNumS)))
          {
            glMultiTexCoord3f(GL_TEXTURE2_ARB, 
                              _rMinS + _rTexelWidth, _rMinT + _rTexelHeight, 
                              slice);
            glTexCoord2f(rLeftS, rBottomT);  glVertex2f(rLeftX, rBottomY);
            glMultiTexCoord3f(GL_TEXTURE2_ARB, 
                              _iSliceResS - _rTexelWidth, _rMinT + _rTexelHeight, 
                              slice);
            glTexCoord2f(rRightS, rBottomT); glVertex2f(rRightX, rBottomY);
            glMultiTexCoord3f(GL_TEXTURE2_ARB, 
                              _iSliceResS - _rTexelWidth, _iSliceResT - _rTexelHeight, 
                              slice);
            glTexCoord2f(rRightS, rTopT);    glVertex2f(rRightX,  rTopY);
            glMultiTexCoord3f(GL_TEXTURE2_ARB, 
                              _rMinS + _rTexelWidth, _iSliceResT - _rTexelHeight,
                              slice);
            glTexCoord2f(rLeftS, rTopT);     glVertex2f(rLeftX,  rTopY);
          }   
          rLeftX  += _rSliceWidth; rRightX += _rSliceWidth;
          rLeftS  += _iSliceResS;  rRightS += _iSliceResS;
          slice += 1;
        }
        rBottomY += _rSliceHeight; rTopY += _rSliceHeight;
        rBottomT += _iSliceResT;   rTopT += _iSliceResT;
      }
    }
    glEnd();
  }

protected: // data
  float _rPixelWidth, _rPixelHeight;
  float _rTexelWidth, _rTexelHeight;
  int   _iTexResS,    _iTexResT;
  int   _iSliceResS,  _iSliceResT;
  float _rSliceWidth, _rSliceHeight;
  int   _iNumS,       _iNumT; // number of slices per row / column
  
  float _rMinX,       _rMinY;
  float _rMaxX,       _rMaxY;
  float _rMinS,       _rMaxS;
  float _rMinT,       _rMaxT;
  
};

//-----------------------------------------------------------------------------
/**
 * @class Flat3DVectorizedTexComputePolicy
 * @brief Handles SlabOp rendering (a quad) with a single texture coordinate.
 *        Use when going between full res and vectorized scalar textures.
 *        Works both for vectorized->full_res and full_res->vectorized.
 *        For vectorized->vectorized, just use the regular
 *        Flat3DTexComputePolicy
 */
class Flat3DVectorizedTexComputePolicy
{
public:

  //---------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //---------------------------------------------------------------------------
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

  //---------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //---------------------------------------------------------------------------
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

  //---------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn SetTexCoordRect(int texUnit, float minS, float minT, float maxS, float maxT)
   * @brief Sets the texture coordinate extents.
   */ 
  void SetSliceTexDimensions(int width, int height, int numS, int numT)
  { 
    _iSliceResS = width;
    _iSliceResT = height;
    _iNumS      = numS; 
    _iNumT      = numT;
    _iTexResS   = width * numS; 
    _iTexResT   = height * numT;
    UpdateCoordinates();
  }

protected:  // methods

  Flat3DVectorizedTexComputePolicy() : _rMinX(0), _rMinY(0), _rMaxX(1), _rMaxY(1) 
  {
    SetTexCoordRect(0, 0, 1, 1);
  }

  ~Flat3DVectorizedTexComputePolicy() {}

  void UpdateCoordinates()
  {
    _rPixelWidth  = (_rMaxX - _rMinX) / _iTexResS;
    _rPixelHeight = (_rMaxY - _rMinY) / _iTexResT;
    _rSliceWidth  = (_rMaxX - _rMinX) / _iNumS;
    _rSliceHeight = (_rMaxY - _rMinY) / _iNumT;

    // input to output texel ratio expected to be integral
    float Swidth = _rMaxS - _rMinS;
    float Twidth = _rMaxT - _rMinT;
    if (Swidth >= _iTexResS) 
      _rTexelWidth = ceilf(Swidth / _iTexResS);
    else
      _rTexelWidth = 1.0f/ceilf(_iTexResS/Swidth);
    if (Twidth >= _iTexResT) 
      _rTexelHeight = ceilf(Twidth / _iTexResT);
    else
      _rTexelHeight = 1.0f/ceilf(_iTexResT/Twidth);
  }

  //---------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn Compute()
   * @brief Performs the slab computation.  Called by SlabOp::Compute()
   */ 
  void Compute()
  {
    glBegin(GL_QUADS);
    {
      float rLeftX, rRightX;
      float rLeftS, rRightS;
      float rBottomY, rTopY;
      float rBottomT, rTopT;    
      float rSliceLeftS, rSliceRightS;
      float rSliceBottomT, rSliceTopT;    
   
      // Rendering texture domain of size
      //   _rTexelWidth  * _iSliceResS x
      //   _rTexelHeight * _iSliceResT
      // 
      // onto rectangle of size
      //   _iSliceResS x _iSliceResT
      //
      // But really we only have real input data of size
      //   (_MaxS-_MinS)/_iNumS x (_MaxT-_MinT)/_iNumT
      //   == SrcTexWidth x SrcTexHeight
      //

      rBottomY = _rMinY + _rPixelHeight;
      rTopY    = _rMinY + _rSliceHeight - _rPixelHeight;
      rBottomT = _rMinT + 1;
      rTopT    = _rMinT + _iSliceResT - 1;

      rSliceLeftS   = _rMinS + _rTexelWidth;
      rSliceRightS  = _rMinS + (_iSliceResS-1)*_rTexelWidth;
      rSliceBottomT = _rMinT + _rTexelHeight;
      rSliceTopT    = _rMinT + (_iSliceResT-1)*_rTexelHeight;


      float slice = 0.5;

      for (int j = 0; j < _iNumT; ++j)
      {
        rLeftX  = _rMinX + _rPixelWidth; 
        rRightX = _rMinX + _rSliceWidth - _rPixelWidth;
        rLeftS  = _rMinS + 1;
        rRightS = _rMinS + _iSliceResS - 1;

        for (int i = 0; i < _iNumS; ++i)
        {
          if (((0 != (i + j)) && (_iNumS * _iNumT - 1 != i + j * _iNumS)))
          {
            // TEXTURE2 is used for local slice coordinates at source resolution
            // TEXTURE0 is used for global coords at destination resolution
            glMultiTexCoord3f(GL_TEXTURE2_ARB, rSliceLeftS, rSliceBottomT, slice);
            glTexCoord2f(rLeftS, rBottomT);  glVertex2f(rLeftX, rBottomY);

            glMultiTexCoord3f(GL_TEXTURE2_ARB, rSliceRightS, rSliceBottomT, slice);
            glTexCoord2f(rRightS, rBottomT); glVertex2f(rRightX, rBottomY);
 
            glMultiTexCoord3f(GL_TEXTURE2_ARB, rSliceRightS, rSliceTopT, slice);
            glTexCoord2f(rRightS, rTopT);    glVertex2f(rRightX,  rTopY);
 
            glMultiTexCoord3f(GL_TEXTURE2_ARB, rSliceLeftS, rSliceTopT, slice);
            glTexCoord2f(rLeftS, rTopT);     glVertex2f(rLeftX,  rTopY);
          }   
          rLeftX  += _rSliceWidth; rRightX += _rSliceWidth;
          rLeftS  += _iSliceResS;  rRightS += _iSliceResS;
          slice += 1;
        }
        rBottomY += _rSliceHeight; rTopY += _rSliceHeight;
        rBottomT += _iSliceResT;   rTopT += _iSliceResT;
      }
    }
    glEnd();
  }

protected: // data
  float _rPixelWidth, _rPixelHeight;
  float _rTexelWidth, _rTexelHeight;
  int   _iTexResS,    _iTexResT;
  int   _iSliceResS,  _iSliceResT;
  float _rSliceWidth, _rSliceHeight;
  int   _iNumS,       _iNumT; // number of slices per row / column
  
  float _rMinX,       _rMinY;
  float _rMaxX,       _rMaxY;
  float _rMinS,       _rMaxS;
  float _rMinT,       _rMaxT;
  
};

//-----------------------------------------------------------------------------
/**
 * @class Flat3DBoundaryComputePolicy
 * @brief Draws boundary edges for a "flattened" 3D texture.
 */
class Flat3DBoundaryComputePolicy
{
public:

  //---------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //---------------------------------------------------------------------------
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

  //---------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //---------------------------------------------------------------------------
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

  //---------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn SetTexCoordRect(int texUnit, float minS, float minT, float maxS, float maxT)
   * @brief Sets the texture coordinate extents.
   */ 
  void SetSliceTexDimensions(int width, int height, int numS, int numT)
  { 
    _iSliceResS = width;
    _iSliceResT = height;
    _iNumS      = numS; 
    _iNumT      = numT;
    _iTexResS   = width * numS; 
    _iTexResT   = height * numT;
    UpdateCoordinates();
  }

  //-----------------------------------------------------------------------------
  // Function     	  : EnableSides
  // Description	    : 
  //-----------------------------------------------------------------------------
  /**
   * @fn EnableSides(bool left, bool right, bool bottom, bool top, bool near, bool far)
   * @brief Individually enables / dsables the four side boundaries.
   */ 
  void EnableSides(bool left, bool right, bool bottom, bool top, bool hither, bool yon)
  {
    _bLeft = left; _bRight = right; 
    _bBottom = bottom; _bTop = top; 
    _bNear = hither; _bFar = yon;
  }


  //-----------------------------------------------------------------------------
  // Function     	  : SetPeriodic
  // Description	    : 
  //-----------------------------------------------------------------------------
  /**
   * @fn SetPeriodic(bool left, bool right, bool bottom, bool top)
   * @brief Individually sets each boundary as periodic or not.
   */ 
  void SetPeriodic(bool left, bool right, bool bottom, bool top, bool hither, bool yon)
  {
    _bLeftPeriodic = left; _bRightPeriodic = right; 
    _bBottomPeriodic = bottom; _bTopPeriodic = top;
    _bNearPeriodic = hither; _bFarPeriodic = yon;
  }

protected:  // methods

  Flat3DBoundaryComputePolicy() 
  : _iTexResS(1), _iTexResT(1), 
    _rMinX(0), _rMinY(0), _rMaxX(1), _rMaxY(1),
    _bLeft(true), _bRight(true), 
    _bBottom(true), _bTop(true), 
    _bNear(true), _bFar(true),
    _bLeftPeriodic(false), _bRightPeriodic(false), 
    _bBottomPeriodic(false), _bTopPeriodic(false),
    _bNearPeriodic(false), _bFarPeriodic(false)
  {
    SetTexCoordRect(0, 0, 1, 1);
    SetSliceTexDimensions(1, 1, 1, 1);
  }

  ~Flat3DBoundaryComputePolicy() {}


  void UpdateCoordinates()
  {
    _rPixelWidth  = (_rMaxX - _rMinX) / (_rMaxS - _rMinS);
    _rPixelHeight = (_rMaxY - _rMinY) / (_rMaxT - _rMinT);
    _rTexelWidth  = (_rMaxS - _rMinS) / _iTexResS;
    _rTexelHeight = (_rMaxT - _rMinT) / _iTexResT;
    _rSliceWidth  = (_rMaxX - _rMinX) / _iNumS;
    _rSliceHeight = (_rMaxY - _rMinY) / _iNumT;
  }

  //---------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //---------------------------------------------------------------------------
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

    // NOTE: this should work, but there seems to be a bug in the NVIDIA drivers 
    // currently that causes a weird shift in float pbuffers.
    
    float rLeftX, rRightX;
    float rLeftS, rRightS;
    float rBottomY, rTopY;
    float rBottomT, rTopT;
    
    rLeftX  = _rMinX + 0.5f * _rPixelWidth; 
    rRightX = _rMinX + _rSliceWidth - 0.5f * _rPixelWidth;
    rLeftS  = _rMinS; 
    rRightS = _rMinS + _iSliceResS  - _rTexelWidth;
    
    rBottomY = _rMinY + 0.5f * _rPixelHeight; 
    rTopY    = _rMinY + _rSliceHeight - 0.5f * _rPixelHeight;
    rBottomT = _rMinT; 
    rTopT    = _rMinT + _iSliceResT - _rTexelHeight;

    // "near" boundary face (face 0)
    if (_bNear)
    {
      glBegin(GL_QUADS);
      {
        if (_bNearPeriodic)
        {
          glMultiTexCoord2f(GL_TEXTURE1_ARB, 
            (_iNumS > 1) ? (float)((_iNumS - 2) * _iSliceResS) : 0,
            (_iNumT > 1) ? (float)((_iNumT - 1) * _iSliceResT) : 0);
        }
        else if (_iNumS > 1)
          glMultiTexCoord3f(GL_TEXTURE1_ARB, (float)_iSliceResS, 0, 1);
        /*else if (_iNumT > 1)
          glMultiTexCoord2f(GL_TEXTURE1_ARB, 0, (float)_iSliceResT);*/
        else
          glMultiTexCoord3f(GL_TEXTURE1_ARB, 0, 0, 0);
        
        glTexCoord2f(rLeftS, rBottomT);  glVertex2f(rLeftX, rBottomY);
        glTexCoord2f(rRightS, rBottomT); glVertex2f(rRightX, rBottomY);
        glTexCoord2f(rRightS, rTopT);    glVertex2f(rRightX, rTopY);
        glTexCoord2f(rLeftS, rTopT);     glVertex2f(rLeftX, rTopY);
      }
      glEnd();  
    }

    glBegin(GL_LINES);
    {      
      
      // vertical boundaries
      for (int col = 0; col < _iNumS; ++col)
      {
        // left boundary
        if (_bLeft)
        {
          glMultiTexCoord3f(GL_TEXTURE1_ARB, 
                            _bLeftPeriodic ? _iSliceResS - 2 * _rTexelWidth : 
                                             _rTexelWidth, 0, 0); // offset amount
          glTexCoord2f(rLeftS, _rMinT); glVertex2f(rLeftX, _rMinY);
          glTexCoord2f(rLeftS, _rMaxT); glVertex2f(rLeftX, _rMaxY);
        }
        
        // right boundary
        if (_bRight)
        {
          glMultiTexCoord3f(GL_TEXTURE1_ARB, 
                            _bRightPeriodic ? -_iSliceResS + 2 * _rTexelWidth : 
                                              -_rTexelWidth, 0, 0); // offset amount
          glTexCoord2f(rRightS, _rMinT); glVertex2f(rRightX, _rMinY);
          glTexCoord2f(rRightS, _rMaxT); glVertex2f(rRightX, _rMaxY);
        }
        rLeftX  += _rSliceWidth; rRightX += _rSliceWidth;
        rLeftS  += _iSliceResS;  rRightS += _iSliceResS;
      }

      // horizontal boundaries
      for (int row = 0; row < _iNumT; ++row)
      {
        // bottom boundary
        if (_bBottom)
        {
          glMultiTexCoord3f(GL_TEXTURE1_ARB, 0, 
                            _bBottomPeriodic ? _iSliceResT - 2 * _rTexelHeight : 
                                               _rTexelHeight, 0); // offset amount
          glTexCoord2f(_rMinS, rBottomT); glVertex2f(_rMinX, rBottomY);
          glTexCoord2f(_rMaxS, rBottomT); glVertex2f(_rMaxX, rBottomY);
        }
        // top boundary
        if (_bTop)
        {
          glMultiTexCoord3f(GL_TEXTURE1_ARB, 0, 
                            _bTopPeriodic ? -_iTexResT + 2 * _rTexelHeight : 
                                            -_rTexelHeight, 0); // offset amount
          glTexCoord2f(_rMinS, rTopT); glVertex2f(_rMinX, rTopY); 
          glTexCoord2f(_rMaxS, rTopT); glVertex2f(_rMaxX, rTopY);
        }
        rBottomY += _rSliceHeight; rTopY += _rSliceHeight;
        rBottomT += _iSliceResT;   rTopT += _iSliceResT;
      }      
    }
    glEnd();

    rLeftX   -= _rSliceWidth;  rRightX -= _rSliceWidth;
    rLeftS   -= _iSliceResS;   rRightS -= _iSliceResS;
    rBottomY -= _rSliceHeight; rTopY   -= _rSliceHeight;
    rBottomT -= _iSliceResT;   rTopT   -= _iSliceResT;

    // "far" boundary face (face D)
    if (_bFar)
    {
      glBegin(GL_QUADS);
      {
        if (_bFarPeriodic)
        {
          glMultiTexCoord2f(GL_TEXTURE1_ARB, 
            (_iNumS > 1) ? (float)((_iNumS - 2) * -_iSliceResS) : 0,
            (_iNumT > 1) ? (float)((_iNumT - 1) * -_iSliceResT) : 0);
        }
        else if (_iNumS > 1)
          glMultiTexCoord3f(GL_TEXTURE1_ARB, (float)-_iSliceResS, 0, 1);
        /*else if (_iNumT > 1)
          glMultiTexCoord2f(GL_TEXTURE1_ARB, 0, (float)-_iSliceResT);*/
        else
          glMultiTexCoord3f(GL_TEXTURE1_ARB, 0, 0, 0);
      
        glTexCoord2f(rLeftS, rBottomT);  glVertex2f(rLeftX, rBottomY);
        glTexCoord2f(rRightS, rBottomT); glVertex2f(rRightX, rBottomY);
        glTexCoord2f(rRightS, rTopT);    glVertex2f(rRightX, rTopY);
        glTexCoord2f(rLeftS, rTopT);     glVertex2f(rLeftX, rTopY);
      }
      glEnd();
    }
  }

protected: // data
  float _rPixelWidth, _rPixelHeight;
  float _rTexelWidth, _rTexelHeight;
  int   _iTexResS,    _iTexResT;
  int   _iSliceResS,  _iSliceResT;
  float _rSliceWidth, _rSliceHeight;
  float _rMinX,       _rMinY;
  float _rMaxX,       _rMaxY;
  float _rMinS,       _rMaxS;
  float _rMinT,       _rMaxT;
  int   _iNumS,       _iNumT; // number of slices per row / column
  float _bLeft, _bRight, _bBottom, _bTop, _bNear, _bFar;
  float _bLeftPeriodic,   _bRightPeriodic;
  float _bBottomPeriodic, _bTopPeriodic;
  float _bNearPeriodic,   _bFarPeriodic;

};


//-----------------------------------------------------------------------------
/**
 * @class MultiTextureGLComputePolicy
 * @brief Handles SlabOp rendering (a quad) with multiple texture coordinates.
 */
class VolumeComputePolicy
{
public:

  //---------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn SetSlabRect(float minX, float minY, float maxX, float maxY)
   * @brief Sets the dimensions of the quad to render.
   */ 
  void SetVolume(float minX, float minY, float minZ, 
                 float maxX, float maxY, float maxZ)
  {
    _rMinX = minX; _rMaxX = maxX;
    _rMinY = minY; _rMaxY = maxY;
    _rMinZ = minZ; _rMaxZ = maxZ;
  }

  void SetCamera(const Camera& cam)
  {
    _cam = cam;
  }
    
  void SetNumSlices(int num)
  {
    _iNumSlices = num;
  }

  //---------------------------------------------------------------------------
  // Function     	  : SetTexCoordRect
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn SetTexCoordRect(int texUnit, float minS, float minT, float maxS, float maxT)
   * @brief Sets the texture coordinate extents.
   */ 
  /*void SetTexCoordRect(int texUnit, 
                       float minS, float minT, float minR,
                       float maxS, float maxT, float maxR)
  { 
    _rMinS[texUnit] = minS; _rMaxS[texUnit] = maxS;
    _rMinT[texUnit] = minT; _rMaxT[texUnit] = maxT;
    _rMinR[texUnit] = minR; _rMaxR[texUnit] = maxR;
  }*/

protected:  // methods

  VolumeComputePolicy() 
    : _rMinX(0), _rMinY(0), _rMinZ(0), _rMaxX(1), _rMaxY(1), _rMaxZ(1),
      _iNumSlices(16)
  {
    memset(_M, 0, 9 * sizeof(float));
    _M[0] = _M[5] = _M[10] = _M[15] = 1;
    /*glGetIntegerv(GL_MAX_TEXTURE_UNITS_ARB, &_iNumTextureUnits);
    _rMinS = new float[_iNumTextureUnits];
    _rMaxS = new float[_iNumTextureUnits];
    _rMinT = new float[_iNumTextureUnits];
    _rMaxT = new float[_iNumTextureUnits];
    _rMinR = new float[_iNumTextureUnits];
    _rMaxR = new float[_iNumTextureUnits];

    for (int i = 0; i < _iNumTextureUnits; ++i)
    {
      SetTexCoordRect(i, 0, 0, 0, 1, 1, 1);
    }*/
  }

  ~VolumeComputePolicy() 
  {
/*    delete [] _rMinS; delete [] _rMaxS;
    delete [] _rMinT; delete [] _rMaxT;
*/}

  //---------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn Compute()
   * @brief Performs the slab computation.  Called by SlabOp::Compute()
   */ 
  void Compute()
  {
    int i;
    glColor4f(1, 1, 1, 1);

    // enable clip planes
    double cp[4] = {1, 0, 0, 1};
    glClipPlane(GL_CLIP_PLANE0, cp);
    cp[0] = -1;
    glClipPlane(GL_CLIP_PLANE1, cp);
    cp[0] = 0; cp[1] = 1;
    glClipPlane(GL_CLIP_PLANE2, cp);
    cp[1] = -1;
    glClipPlane(GL_CLIP_PLANE3, cp);
    cp[1] = 0; cp[2] = 1;
    glClipPlane(GL_CLIP_PLANE4, cp);
    cp[2] = -1;
    glClipPlane(GL_CLIP_PLANE5, cp);
    for (i = 0; i < 6; ++i)
    {
      glEnable(GL_CLIP_PLANE0 +i);
    }
    
    glDisable(GL_DEPTH_TEST);
    glAlphaFunc(GL_GREATER, 0);
    glEnable(GL_ALPHA_TEST);
    
    glActiveTextureARB(GL_TEXTURE0_ARB);

    // enable clip planes
    static GLfloat planeS[] = { 0.5, 0.0, 0.0, 0.5 };
    static GLfloat planeT[] = { 0.0, 0.5, 0.0, 0.5 };
    static GLfloat planeR[] = { 0.0, 0.0, 0.5, 0.5 };
  
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
    glTexGenfv(GL_S, GL_OBJECT_PLANE, planeS);
    glTexGenfv(GL_T, GL_OBJECT_PLANE, planeT);
    glTexGenfv(GL_R, GL_OBJECT_PLANE, planeR);
    glEnable(GL_TEXTURE_GEN_S);
    glEnable(GL_TEXTURE_GEN_T);
    glEnable(GL_TEXTURE_GEN_R);
     
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    glBegin(GL_QUADS);
    {      
      float radius = (float)sqrt(3);
    
      Vec3f ll(0, 0, 0), lr(0, 0, 0), ur(0, 0, 0), ul(0, 0, 0);
      ll -= _cam.X; ll -= _cam.Y; ll -= _cam.Z;
      ll *= radius;
      lr += _cam.X; lr -= _cam.Y; lr -= _cam.Z;
      lr *= radius;
      ur += _cam.X; ur += _cam.Y; ur -= _cam.Z;
      ur *= radius;
      ul -= _cam.X; ul += _cam.Y; ul -= _cam.Z;
      ul *= radius;
    
      float rSpacing = 2 * radius / (float) _iNumSlices;
    
      Vec3f step = _cam.Z; step *= rSpacing;
    
      for (int i = 0; i < _iNumSlices; ++i)
      {
        glVertex3fv(ll);
        glVertex3fv(lr);
        glVertex3fv(ur);
        glVertex3fv(ul);
      
        ll += step; lr += step; ur += step; ul += step;
      }      
    }
    glEnd();

    // disable clip planes
    for (i = 0; i < 6; ++i)
    {
      glDisable(GL_CLIP_PLANE0 + i);
    }

    glDisable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    
    // disable texgen
    glActiveTextureARB(GL_TEXTURE0_ARB);
    glDisable(GL_TEXTURE_GEN_S);
    glDisable(GL_TEXTURE_GEN_T);
    glDisable(GL_TEXTURE_GEN_R);
  }

protected: // data
  int   _iNumTextureUnits;
  float _rMinX,  _rMinY, _rMinZ;
  float _rMaxX,  _rMaxY, _rMaxZ;
  float _M[16];
  int   _iNumSlices;
  Camera _cam;
  /*float *_rMinS, *_rMaxS;
  float *_rMinT, *_rMaxT;*/
};

class VolumeGLComputePolicy
{
public:

  //---------------------------------------------------------------------------
  // Function     	  : SetSlabRect
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn SetSlabRect(float minX, float minY, float maxX, float maxY, float z)
   * @brief Sets the dimensions of the quad to render.
   */ 
  void SetSlabRect(Vec3f ll, Vec3f lr, Vec3f ur, Vec3f ul)
  {
    _ll = ll;
    _lr = lr;
    _ul = ul;
    _ur = ur;
  }

protected:  // methods

  VolumeGLComputePolicy() : _ll(0,0,0), _lr(0,0,0), _ul(0,0,0), _ur(0,0,0)
  {
  }

  ~VolumeGLComputePolicy() {}

  //---------------------------------------------------------------------------
  // Function     	  : Compute
  // Description	    : 
  //---------------------------------------------------------------------------
  /**
   * @fn Compute()
   * @brief Performs the slab computation.  Called by SlabOp::Compute()
   */ 
  void Compute()
  {
    glBegin(GL_QUADS);
    {
      glMultiTexCoord3fvARB(GL_TEXTURE2_ARB, _ll); glVertex3fv(_ll);
      glMultiTexCoord3fvARB(GL_TEXTURE2_ARB, _lr); glVertex3fv(_lr);
      glMultiTexCoord3fvARB(GL_TEXTURE2_ARB, _ur); glVertex3fv(_ur);
      glMultiTexCoord3fvARB(GL_TEXTURE2_ARB, _ul); glVertex3fv(_ul);
    }
    glEnd();
  }

protected: // data
  Vec3f _ll, _lr, _ul, _ur;
};

#endif //__FLAT3DTEXTURESLABOPGL_H__
