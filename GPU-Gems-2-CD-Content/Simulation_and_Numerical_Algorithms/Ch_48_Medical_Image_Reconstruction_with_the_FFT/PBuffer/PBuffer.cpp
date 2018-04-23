/***************************************************************************
*		FILE NAME:  PBuffer.cpp
*
* ONE LINE SUMMARY:
*		This file contains some of the member function for the PBuffer class.
*        
*		Thilaka Sumanaweera
*		Siemens Medical Solutions USA, Inc.
*		1230 Shorebird Way
*		Mountain View, CA 94039
*		USA
*		Thilaka.Sumanaweera@siemens.com
*
* DESCRIPTION:
*
*****************************************************************************
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
****************************************************************************/
#include <stdlib.h>
#include <iostream>
#include "PBuffer.h"
using namespace std;

#define MAX_ATTRIBS 32
#define MAX_FORMATS 32

PBuffer::PBuffer(Vendor vendor,
				 int width, 
				 int height, 
				 InternalFormat IntForm,
				 int RedBits, 
				 int GreenBits, 
				 int BlueBits, 
				 int AlphaBits, 
				 bool isFloat, 
				 bool dBuffer, 
				 bool hasDepth,
				 int NoAuxBuffers,
				 bool hasStencil, 
				 bool hasStereo,
				 bool texture, 
				 bool share) :
   m_buffer(0), m_RC(0), m_DC(0), m_prevRC(0), m_prevDC(0)
{
   int iAttribs[2*MAX_ATTRIBS];
   float fAttribs[2*MAX_ATTRIBS];
   int niAttribs = 0;
   int nfAttribs = 0;
   int pformats[MAX_FORMATS];
   unsigned int nformat;

   HGLRC rc = wglGetCurrentContext();
   HDC hdc = wglGetCurrentDC();

   //full float buffer
   niAttribs = 0;
   nfAttribs = 0;

   memset( iAttribs, 0, sizeof(int)*2*MAX_ATTRIBS);
   memset( fAttribs, 0, sizeof(int)*2*MAX_ATTRIBS);

   iAttribs[ niAttribs*2] = WGL_DRAW_TO_PBUFFER_ARB;
   iAttribs[ niAttribs*2 + 1] = 1;
   niAttribs++;

   if (isFloat) {
	   switch(vendor) {
	   case GPU_NVIDIA:
		   iAttribs[ niAttribs*2] = WGL_PIXEL_TYPE_ARB;
		   iAttribs[ niAttribs*2 + 1] = WGL_TYPE_RGBA_ARB;
		   niAttribs++;
		   iAttribs[ niAttribs*2] = WGL_FLOAT_COMPONENTS_NV;
		   iAttribs[ niAttribs*2 + 1] = GL_TRUE;
		   niAttribs++;
		   break;
	   case GPU_ATI:
		   iAttribs[ niAttribs*2] = WGL_PIXEL_TYPE_ARB;
		   iAttribs[ niAttribs*2 + 1] = WGL_TYPE_RGBA_FLOAT_ATI;
		   niAttribs++;
		   break;
	   default:
		   break;
	   }
   }
   else {
	   iAttribs[ niAttribs*2] = WGL_PIXEL_TYPE_ARB;
	   iAttribs[ niAttribs*2 + 1] = WGL_TYPE_RGBA_ARB;
	   niAttribs++;
   }


   iAttribs[ niAttribs*2] = WGL_DOUBLE_BUFFER_ARB;
   iAttribs[ niAttribs*2 + 1] = (dBuffer) ? 1 : 0;
   niAttribs++;
   iAttribs[ niAttribs*2] = WGL_SUPPORT_OPENGL_ARB;
   iAttribs[ niAttribs*2 + 1] = 1;
   niAttribs++;
   iAttribs[ niAttribs*2] = WGL_RED_BITS_ARB;
   iAttribs[ niAttribs*2 + 1] = RedBits;
   niAttribs++;
   iAttribs[ niAttribs*2] = WGL_GREEN_BITS_ARB;
   iAttribs[ niAttribs*2 + 1] = GreenBits;
   niAttribs++;
   iAttribs[ niAttribs*2] = WGL_BLUE_BITS_ARB;
   iAttribs[ niAttribs*2 + 1] = BlueBits;
   niAttribs++;
   iAttribs[ niAttribs*2] = WGL_ALPHA_BITS_ARB;
   iAttribs[ niAttribs*2 + 1] = AlphaBits;
   niAttribs++;
   iAttribs[ niAttribs*2] = WGL_COLOR_BITS_ARB;
   iAttribs[ niAttribs*2 + 1] = RedBits+GreenBits+BlueBits;
   niAttribs++;

   iAttribs[ niAttribs*2] = WGL_DEPTH_BITS_ARB;
   iAttribs[ niAttribs*2 + 1] = hasDepth ? 24 : 0;
   niAttribs++;
   iAttribs[ niAttribs*2] = WGL_STENCIL_BITS_ARB;
   iAttribs[ niAttribs*2 + 1] = hasStencil ? 8 : 0;
   niAttribs++;
   iAttribs[ niAttribs*2] = WGL_STEREO_ARB;
   iAttribs[ niAttribs*2 + 1] = hasStereo ? GL_TRUE : GL_FALSE;
   niAttribs++;
   /*
   if (hasDepth)
   {
      iAttribs[ niAttribs*2] = WGL_DEPTH_BITS_ARB;
      iAttribs[ niAttribs*2 + 1] = 24;
      niAttribs++;
   }
   if (hasStencil)
   {
      iAttribs[ niAttribs*2] = WGL_STENCIL_BITS_ARB;
      iAttribs[ niAttribs*2 + 1] = 8;
      niAttribs++;
   }
   */

   if (NoAuxBuffers)
   {
      iAttribs[ niAttribs*2] = WGL_AUX_BUFFERS_ARB;
      iAttribs[ niAttribs*2 + 1] = NoAuxBuffers;
      niAttribs++;
   }

   if (texture)
   {
	   switch (vendor) {
	   case GPU_NVIDIA:

		   //IntForm
		   if (isFloat) {
			   switch (IntForm) {
			   case PBUF_RGBA:
				   iAttribs[niAttribs*2] = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV;
				   break;
			   case PBUF_RG:
				   iAttribs[niAttribs*2] = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV;
				   break;
			   case PBUF_R:
				   iAttribs[niAttribs*2] = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV;
				   break;
			   default:
				   break;
			   }
		   }
		   else {
			   iAttribs[ niAttribs*2] = WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV;
		   }
		   iAttribs[ niAttribs*2 + 1] = GL_TRUE;
		   //iAttribs[ niAttribs*2 + 1] = GL_FALSE;
		   niAttribs++;
		   break;
	   case GPU_ATI:
		   iAttribs[ niAttribs*2]     = WGL_BIND_TO_TEXTURE_RGBA_ARB;
		   iAttribs[ niAttribs*2 + 1] = GL_TRUE;
		   niAttribs++;
		   break;
	   default:
		   break;
	   }
   }



   if (wglChoosePixelFormatARB( hdc, iAttribs, fAttribs, MAX_FORMATS, pformats, &nformat))
   {
      niAttribs = 0;
      if (texture)
      {
		  switch (vendor) {
		  case GPU_NVIDIA:
			  iAttribs[niAttribs++] = WGL_TEXTURE_FORMAT_ARB;
			  if (isFloat) {
				  switch (IntForm) {
				  case PBUF_RGBA:
					  iAttribs[niAttribs++] = WGL_TEXTURE_FLOAT_RGBA_NV;
					  break;
				  case PBUF_RG:
					  iAttribs[niAttribs++] = WGL_TEXTURE_FLOAT_RG_NV;
					  break;
				  case PBUF_R:
					  iAttribs[niAttribs++] = WGL_TEXTURE_FLOAT_R_NV;
					  break;
				  default:
					  break;
				  }
				  //iAttribs[niAttribs++] = WGL_TEXTURE_FLOAT_RGBA_NV;
			  }
			  else {
				  iAttribs[niAttribs++] = WGL_TEXTURE_RGBA_ARB;
			  }
			  //iAttribs[niAttribs++] = (isFloat) ? WGL_TEXTURE_FLOAT_RGBA_NV : WGL_TEXTURE_RGBA_ARB;
			  iAttribs[niAttribs++] = WGL_TEXTURE_TARGET_ARB;
			  iAttribs[niAttribs++] = WGL_TEXTURE_RECTANGLE_NV;
			  break;
		  case GPU_ATI:
			  iAttribs[niAttribs++] = WGL_TEXTURE_FORMAT_ARB;
			  iAttribs[niAttribs++] = WGL_TEXTURE_RGBA_ARB;
			  iAttribs[niAttribs++] = WGL_TEXTURE_TARGET_ARB;
			  iAttribs[niAttribs++] = WGL_TEXTURE_2D_ARB;
			  break;
		  default:
			  break;
		  }
      }
      iAttribs[niAttribs] = 0;


      m_buffer = wglCreatePbufferARB( hdc, pformats[0], width, height, iAttribs);
	  int w, h;
	  wglQueryPbufferARB(m_buffer, WGL_PBUFFER_HEIGHT_ARB, &h);
	  wglQueryPbufferARB(m_buffer, WGL_PBUFFER_WIDTH_ARB, &w);
	  //cerr << "Returned pBuffer: [" << w << ", " << h << "]" << endl;

	  niAttribs = 0;
	  iAttribs[niAttribs++] = WGL_AUX_BUFFERS_ARB;
	  int Vals;
	  wglGetPixelFormatAttribivARB(hdc, pformats[0], 0, niAttribs, iAttribs, &Vals);
	  //cerr << "No of aux buffers: " << Vals << endl;


      if (m_buffer)
      {
         m_DC = wglGetPbufferDCARB( m_buffer);

         if (m_DC)
         {
            m_RC = wglCreateContext( m_DC);

            if (m_RC)
            {
               if (share)
               {
                  wglShareLists( rc, m_RC);
               }
            }
            else //floatRC
            {
               wglReleasePbufferDCARB( m_buffer, m_DC);
               m_DC = NULL;
               wglDestroyPbufferARB( m_buffer);
               m_buffer = NULL;
            }
         }
         else //floatDC
         {
            wglDestroyPbufferARB( m_buffer);
            m_buffer = NULL;
         }
      }
      else //floatBuffer
      {
         //nothing presently
      }
   }
   if (m_buffer == NULL) {
	   cerr << "Could not create a pBuffer" << endl;
	   exit(0);
   }
}



//
//
//
////////////////////////////////////////////////////////////////////////////////
PBuffer::~PBuffer()
{
   if (m_RC)
   {
      wglDeleteContext( m_RC);
      m_RC = NULL;
   }

   if (m_DC)
   {
		wglReleasePbufferDCARB( m_buffer,m_DC );
      m_DC = NULL;
   }

   if (m_buffer)
   {
		wglDestroyPbufferARB( m_buffer);
      m_buffer = NULL;
   }
}

//
//
//
////////////////////////////////////////////////////////////////////////////////
bool PBuffer::IsValid()
{
   return (m_RC != NULL);
}

//
//
//
////////////////////////////////////////////////////////////////////////////////
void PBuffer::Bind()
{
   m_prevDC = wglGetCurrentDC();
   m_prevRC = wglGetCurrentContext();

   wglMakeCurrent( m_DC, m_RC);
}

//
//
//
////////////////////////////////////////////////////////////////////////////////
void PBuffer::Unbind()
{
   wglMakeCurrent( m_prevDC, m_prevRC);
}

//
//
//
////////////////////////////////////////////////////////////////////////////////
char PBuffer::BindAsTexture( int buffer)
{
   return(wglBindTexImageARB( m_buffer, buffer));
}

//
//
//
////////////////////////////////////////////////////////////////////////////////
char PBuffer::ReleaseTexture( int buffer)
{
   return(wglReleaseTexImageARB( m_buffer, buffer));
}

//
//
//
////////////////////////////////////////////////////////////////////////////////
void PBuffer::Swap()
{
   wglSwapLayerBuffers( m_DC, WGL_SWAP_MAIN_PLANE);
}