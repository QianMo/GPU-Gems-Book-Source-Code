/***************************************************************************
*        FILE NAME:  PingPong.h
*
* ONE LINE SUMMARY:
*        PingPong class is declared in this file.
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
#ifndef __PINGPONG__
#define __PINGPONG__

enum PingMethod {SINGLE_PBUFFER, DOUBLE_PBUFFER, AUTO_PBUFFER};

#include <PBuffer.h>
#include <GL/glut.h>
#include <GL/glext.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>

/*
C L A S S
*/
class PingPong {
public:
    // member functions
	PingPong::PingPong() {};
	PingPong::PingPong(Vendor GPU, PingMethod Method_, int Width, int Height);
	~PingPong(void);
	void PingPong::CreatePbuffer(Vendor GPU, int Width, int Height, PBuffer *&pbuf);
	bool PingPong::Quadro_p(void);
	void PingPong::BindTextures(PBuffer *Pbuf, GLuint *Texs);
	void PingPong::UnBindTextures(PBuffer *Pbuf);
	void PingPong::PingForDisplay(GLuint *Texs);
	void PingPong::Ping(bool FirstTime_p, GLuint *Texs);
	void PingPong::Pong(bool FirstTime_p);
	void PingPong::Swap(void);
	void PingPong::Bind(void);
	void PingPong::Unbind(void);

	PingMethod Method;							// Method of managing the buffers: SINGLE_PBUFFER or DOUBLE_PBUFFER
	PBuffer *Pbuffer1;							// a pbuffer containing 8 or 4 draw buffers, depending
												// on SINGLE_PBUFFER or DOUBLE_PBUFFER method.
	PBuffer *Pbuffer2;							// a pbuffer containing 4 draw buffers, depending
												// on SINGLE_PBUFFER or DOUBLE_PBUFFER method. Not used
												// when using SINGLE_PBUFFER
	int NBuf;
	GLenum *TexUnits;							// An array containg four texture unit tokens
	GLenum *BufSet1;							// A set of 4 draw buffers
	GLenum *BufSet2;							// A second set of 4 draw buffers
	GLenum *wglBufSet1;							// A set of 4 wgl draw buffers
	GLenum *wglBufSet2;							// A second set of 4 wgl draw buffers
	
	PBuffer *Src;								// source pbuffer
	PBuffer *Dst;								// destination pbuffer
	GLenum *SrcBufSet;							// source draw buffer set
	GLenum *DstBufSet;							// destination draw buffer set
	GLenum *wglSrcBufSet;						// source wgl draw buffer set
	GLenum *wglDstBufSet;						// destination wgl draw buffer set

  private:
};


#endif  /* __PINGPONG__ */
