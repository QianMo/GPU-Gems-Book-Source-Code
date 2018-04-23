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

#ifndef _BLACKSCHOLESPRICE_H_
#define _BLACKSCHOLESPRICE_H_

#include "options.h"

class BlackScholesPricing : public OptionPricing {
public:
    BlackScholesPricing(OptionStyle ostyle, OptionType otype, int nopts);
    ~BlackScholesPricing();
    const char *GetName(void) { return "Black-Scholes"; }
    bool CPUPrice(OptionParams &params, float *buf);
    bool GPUPrice(OptionParams &params, float *buf);
private:
    float EuropeanPut(float S, float X, float T, float r, float v);
    float EuropeanCall(float S, float X, float T, float r, float v);
    bool bindParams(OptionParams &params);
    PUGBuffer *resultBuf;
    PUGBuffer *priceBuf;
    PUGBuffer *strikeBuf;
    PUGBuffer *yearsBuf;
    PUGBuffer *volBuf;
    PUGProgram *prog;
    int width, height;
};

#endif // _BLACKSCHOLESPRICE_H_
