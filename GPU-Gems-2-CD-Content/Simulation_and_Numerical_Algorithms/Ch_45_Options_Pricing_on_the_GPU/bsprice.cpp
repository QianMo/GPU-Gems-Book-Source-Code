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
#include "bsprice.h"
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.1415926535f
#endif

// The cumulative normal distribution function 
inline float CND(float X)
{
    float const a1 = 0.31938153f, a2 = -0.356563782f, a3 = 1.781477937f;
    float const a4 = -1.821255978f, a5 = 1.330274429f;

    float L = fabsf(X);
    float K = 1.0f / (1.0f + 0.2316419f * L);
    float K2 = K*K;
    float K3 = K2*K;
    float K4 = K3*K;
    float K5 = K4*K;
    float w = 1.0f - 1.0f / sqrtf(2.f * M_PI) * expf(-L * L * .5f) * 
    (a1 * K + a2 * K2 + a3 * K3 + a4 * K4 + a5 * K5);

    if (X < 0.)
        w = 1.0f - w;
    return w;
}

BlackScholesPricing::BlackScholesPricing(OptionStyle ostyle,
                                         OptionType otype,
                                         int num)
{

    optionStyle = ostyle;
    optionType = otype;
    prog = 0;

    if (ostyle == OPTION_STYLE_AMERICAN) {
        // No can do with Black-Scholes.
        return;
    }

    if (!pugInit()) {
        fprintf(stderr, "Unable to initialize Pug.\n");
        return;
    }

    if (num <= 4096 && num > 1) {
        width = num/2;
        height = 2;
    } else if ((num%2048) == 0) {
        width = 2048;
        height = num/2048;
    } else {
        width = height = (int)sqrt((float)num);
    }
    if (width*height != num)
        return;

    resultBuf = pugAllocateBuffer(width, height, PUG_WRITE, 1, true);
    priceBuf = pugAllocateBuffer(width, height, PUG_READ);
    strikeBuf = pugAllocateBuffer(width, height, PUG_READ);
    yearsBuf = pugAllocateBuffer(width, height, PUG_READ);
    volBuf = pugAllocateBuffer(width, height, PUG_READ);

    if (!resultBuf || !priceBuf || !strikeBuf || !yearsBuf || !volBuf)
        fprintf(stderr, "Unable to allocate all buffers\n");

    char entry[1024];
    if (optionStyle == OPTION_STYLE_EUROPEAN)
        strcpy(entry, "European");
    else if (optionStyle == OPTION_STYLE_AMERICAN)
        strcpy(entry, "American");
    else
        return;

    if (optionType == OPTION_TYPE_CALL)
        strcat(entry, "Call");
    else if (optionType == OPTION_TYPE_PUT)
        strcat(entry, "Put");
    else
        return;

    // Load the program onto the PUG
    prog = pugLoadProgram("bs.cg", entry);
    if (!prog)
        fprintf(stderr, "Unable to load Cg program\n");

    // Associate buffers with parameters to the program.
    if (!pugBindStream(prog, "stockPrice", priceBuf) ||
        !pugBindStream(prog, "strikePrice", strikeBuf) || 
        !pugBindStream(prog, "yearsToMaturity", yearsBuf) ||
        !pugBindStream(prog, "volatility", volBuf)) {
        fprintf(stderr, "Unable to set default PUG parameters.\n");
        prog = 0;
    }
}

BlackScholesPricing::~BlackScholesPricing()
{
    delete resultBuf;
    delete priceBuf;
    delete strikeBuf;
    delete yearsBuf;
    delete volBuf;
    if (!pugCleanup())
        fprintf(stderr, "PUG cleanup failed\n");
}

float BlackScholesPricing::EuropeanCall(float S, float X, float T, float r, float v)
{
    float d1=(log(S/X)+(r+v*v/2.f)*T)/(v*sqrt(T));
    float d2=d1-v*sqrt(T);

    return S * CND(d1) - X * exp(-r*T) * CND(d2);
}

float BlackScholesPricing::EuropeanPut(float S, float X, float T, float r, float v)
{
    float d1=(logf(S/X)+(r+v*v/2.f)*T)/(v*sqrtf(T));
    float d2=d1-v*sqrtf(T);

    return X * expf(-r*T) * CND(-d2) - S * CND(-d1);
}

bool BlackScholesPricing::CPUPrice(OptionParams &p, float *buf)
{
    if (optionStyle != OPTION_STYLE_EUROPEAN)
        return false;

    if (optionType == OPTION_TYPE_CALL) {
        for (int i = 0; i < p.n; i++) {
            buf[i] = EuropeanCall(p.S[i], p.X[i], p.T[i], p.r, p.v[i]);
        }
    } else if (optionType == OPTION_TYPE_PUT) {
        for (int i = 0; i < p.n; i++) {
            buf[i] = EuropeanPut(p.S[i], p.X[i], p.T[i], p.r, p.v[i]);
        }
    } else {
        return false;
    }
    return true;
}

bool BlackScholesPricing::bindParams(OptionParams &params)
{
    if (!pugInitBuffer(strikeBuf, params.X) ||
        !pugInitBuffer(priceBuf, params.S) ||
        !pugInitBuffer(yearsBuf, params.T) ||
        !pugInitBuffer(volBuf, params.v)) {
        fprintf(stderr, "Unable to initialize PUG buffers.\n");
        return false;
    }

    if (!pugBindFloat(prog, "riskFreeRate", params.r)) {
        fprintf(stderr, "Unable to set PUG parameters.\n");
        return false;
    }
    return true;
}

bool BlackScholesPricing::GPUPrice(OptionParams &params, float *buf)
{
    if (prog == 0) return false;

    if (!bindParams(params))
        return false;

    // Run program for all entries of resultBuf
    if (!pugRunProgram(prog, resultBuf)) {
        fprintf(stderr, "Couldn't run Black-Scholes PUG program.\n");
        return false;
    }

    if (readback) {
        // Read results back to CPU
        pugReadMemory(buf, resultBuf);
    }
    return true;
}
