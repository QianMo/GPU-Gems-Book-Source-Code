/***************************************************************************
*        FILE NAME:  PBuffer.h
*
* ONE LINE SUMMARY:
*        PBuffer class is declared in this file.
*        
*        Thilaka Sumanaweera
*        Siemens Medical Solutions USA, Inc.
*        1230 Shorebird Way
*        Mountain View, CA 94039
*        USA
*        Thilaka.Sumanaweera@siemens.com
*
* DESCRIPTION:
*
*****************************************************************************
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
****************************************************************************/
#ifndef PBUFFER_H
#define PBUFFER_H

#include <gl/glew.h>
#include <gl/wglew.h>
#include <gl/glut.h>

enum Vendor {GPU_NVIDIA, GPU_ATI, GPU_NONE};
enum InternalFormat {PBUF_R, PBUF_RG, PBUF_RGBA};

class PBuffer
{
   public:
      //create a Pbuffer with the correct attributes, share allows it to share lists with the current context
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
				 bool share);

      ~PBuffer();

      //is this buffer usable
      bool IsValid();
      
      //make it available for rendering
      void Bind();
      void Unbind();

      //make it available as a texture if applicable 
      char BindAsTexture( int buffer);
      char ReleaseTexture( int buffer);

      //swap if applicable
      void Swap();

   private:
      HPBUFFERARB m_buffer;

      HGLRC m_RC;

      HDC m_DC;

      //used to restore the previous RC when unbinding
      HGLRC m_prevRC;
      HDC m_prevDC;
};


#endif //PBUFFER_H

