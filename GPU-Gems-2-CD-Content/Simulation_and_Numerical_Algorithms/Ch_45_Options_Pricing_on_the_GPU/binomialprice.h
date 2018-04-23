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

#ifndef _BINOMIALPRICE_H_
#define _BINOMIALPRICE_H_

#include "options.h"
class BinomialPricing : public OptionPricing {
public:
    BinomialPricing(OptionStyle ostyle, OptionType otype, int nopts, int nsteps);
    ~BinomialPricing();
    const char *GetName(void) { return "Binomial"; }
    bool CPUPrice(OptionParams &params, float *buf);
    bool GPUPrice(OptionParams &params, float *buf);
private:
    float EuropeanPut(float S, float X, float T, float r, float v,
                      float Pu, float Pd, float u);
    float EuropeanCall(float S, float X, float T, float r, float v,
                       float Pu, float Pd, float u);
    float AmericanPut(float S, float X, float T, float r, float v,
                      float Pu, float Pd, float u);
    float AmericanCall(float S, float X, float T, float r, float v,
                       float Pu, float Pd, float u);
    void setNumSteps(int n);
    int numSteps;
    int numOpts;
    void initProbData(OptionParams &params);
    int width, height;
    PUGBuffer *resultBuf;
    PUGBuffer *priceBuf;
    PUGBuffer *strikeBuf;
    PUGBuffer *yearsBuf;
    PUGBuffer *volBuf;
    PUGProgram *prog;
    PUGProgram *iprog;
    PUGBuffer *puBuf, *pdBuf, *uBuf;
    float *pudata, *pddata, *udata;
    float *value;
};


#endif // _BINOMIALPRICE_H_
