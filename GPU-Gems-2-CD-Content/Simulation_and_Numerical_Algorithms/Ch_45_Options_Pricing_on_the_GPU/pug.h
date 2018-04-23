/*********************************************************************NVMH3****

  Copyright NVIDIA Corporation 2004
  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
  *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
  OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
  NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
  CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
  LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
  INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGES.

  ****************************************************************************/

#ifndef PUG_H
#define PUG_H

#include <windows.h>
#include <GL/gl.h>
#include <GL/wglext.h>
#include <Cg/cg.h>

class PUGProgram;

enum PUGTarget
{
    PUG_FRONT,
    PUG_BACK,
    PUG_NUM_TARGETS
};

struct PUGBuffer 
{
    PUGBuffer() { }
    PUGBuffer(int w, int h, int nc, HDC dc, HPBUFFERARB pbuf, HGLRC rc,
              GLuint th, GLuint ib, bool db=false);

    GLuint texHandle, imageBuffer;

    int width, height, nComponents;
    bool bound[2];

    // PBUFFER data
    HPBUFFERARB hpbuffer;   // pbuffer handle
    HDC hpbufferDC;         // pbuffer device context 
    HGLRC hpbufferRC;       // pbuffer render context

    // double-buffered PBUFFER data 
    bool doublebuffered;
    PUGTarget currentDrawBuffer;
};


struct PUGRect {
    PUGRect() { }
    PUGRect(int xx0, int xx1, int yy0, int yy1) {
        x00 = x01 = xx0;
        x10 = x11 = xx1;
        y00 = y10 = yy0;
        y01 = y11 = yy1;
    }
    PUGRect(int xx00, int xx10, int xx01, int xx11,
            int yy00, int yy10, int yy01, int yy11) {
        x00 = xx00; x10 = xx10; x01 = xx01; x11 = xx11;
        y00 = yy00; y10 = yy10; y01 = yy01; y11 = yy11;
    }
	    
    int x00, x01, x10, x11;
    int y00, y01, y10, y11;
};

extern bool pugInit(const char* includePath = "./", CGcontext cgcontext = 0, bool createOGLContext = true);
extern bool pugCleanup();

extern PUGProgram *pugLoadProgram(const char *filename,
                                  const char *entrypoint,
                                  const char **args = NULL);
extern PUGProgram *pugLoadReductionProgram(const char *filename,
                                           const char *reduceop,
                                           int samples = 2,
                                           int components = 1,
                                           const char **args = NULL);

extern bool pugBindFloat(PUGProgram *prog, const char *param,
			             float v0, float v1 = 0, float v2 = 0, float v3 = 0);
extern bool pugBindStream(PUGProgram *prog, const char *param,
			              PUGBuffer *buf, PUGTarget target = PUG_FRONT);
extern bool pugBindDomain(PUGProgram *prog, const char *param,
                          const PUGRect &domain);
extern bool pugRunProgram(PUGProgram *prog, const PUGRect &writeRect, 
			              const PUGRect &readRect);
extern bool pugReadMemory(float *out, PUGBuffer *buf);
extern bool pugReadMemory(float *out, PUGBuffer *buf, PUGRect rect);
extern PUGRect pugTransposeRect(PUGRect &rect);

// over all of current output buffer
extern bool pugRunProgram(PUGProgram *prog, PUGBuffer *output);
extern bool pugRunProgram(PUGProgram *prog, PUGBuffer *output, const PUGRect domain);

extern bool pugRunProgram(PUGProgram *prog, PUGBuffer *output, const PUGRect domain,
                          PUGTarget target);

enum PUGBufferMode {
    PUG_READ,
    PUG_WRITE,
    PUG_READWRITE
};

extern PUGBuffer* pugAllocateBuffer(int width, int height, PUGBufferMode mode,
				                    int components = 1, 
                                    bool doublebuffer = false);

extern void   pugDeleteBuffer(PUGBuffer* buf);

extern bool   pugInitBuffer(PUGBuffer *buf, const float *data, 
                            PUGTarget target = PUG_FRONT);

extern bool   pugStartReadBack(PUGBuffer *);
extern float* pugGetBufferData(PUGBuffer *);
extern bool   pugReleaseBufferData(PUGBuffer *);

extern bool   pugReleaseBuffer(PUGBuffer *buf, PUGTarget target = PUG_FRONT);

extern bool   pugBindTexture(PUGBuffer *buf, PUGTarget target = PUG_FRONT);
extern bool   pugMakeWindowCurrent();

extern void   pugWaitForThreads();

// Parallel Reductions

enum PUGDimension
{
    PUG_DIMENSION_X,
    PUG_DIMENSION_Y,
    PUG_DIMENSION_COUNT
};

extern PUGBuffer* pugReduce1D(PUGProgram *prog, PUGBuffer *inbuf, 
                              PUGBuffer *dblbuf, PUGDimension dim,
                              int nRows, int nCols);
extern PUGBuffer* pugReduce2D(PUGProgram *prog, PUGBuffer *inbuf, 
                              PUGBuffer *dblbuf, int nRows, int nCols);




#endif // PUG_H
