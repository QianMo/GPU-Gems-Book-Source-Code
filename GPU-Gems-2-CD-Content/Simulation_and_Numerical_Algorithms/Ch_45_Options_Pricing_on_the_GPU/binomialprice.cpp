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

#include "binomialprice.h"
#include <stdio.h>
#include <math.h>

BinomialPricing::BinomialPricing(OptionStyle ostyle, OptionType otype,
                                 int nopts, int nsteps)
{
    prog = 0;

    optionStyle = ostyle;
    optionType = otype;

    numOpts = nopts;
    numSteps = nsteps;

    width = numOpts;
    height = numSteps+1;

    // Allocate scratch buffers
    value = new float[numSteps+1];
    pudata = new float[width];
    pddata = new float[width];
    udata  = new float[width];

    if (!pugInit()) {
        fprintf(stderr, "Unable to initialize Pug.\n");
        return;
    }

    // Allocate PUG buffers
    resultBuf = pugAllocateBuffer(width, height, PUG_WRITE, 1, true);
    uBuf      = pugAllocateBuffer(width, 1, PUG_READ);
    priceBuf  = pugAllocateBuffer(width, 1, PUG_READ);
    strikeBuf = pugAllocateBuffer(width, 1, PUG_READ);
    yearsBuf = pugAllocateBuffer(width, 1, PUG_READ);
    volBuf = pugAllocateBuffer(width, 1, PUG_READ);
    puBuf = pugAllocateBuffer(width, 1, PUG_READ);
    pdBuf = pugAllocateBuffer(width, 1, PUG_READ);
    if (!(volBuf && yearsBuf && strikeBuf && priceBuf && resultBuf &&
          puBuf && pdBuf && uBuf)) {
        fprintf(stderr, "Could not allocate buffers.\n");
        prog = 0;
        return;
    }

    // Construct initialization and iteration entrypoint names
    char initEntry[1024];
    char iterateEntry[1024];

    if (optionStyle == OPTION_STYLE_EUROPEAN) {
        strcpy(iterateEntry, "EuropeanIterate");
    } else if (optionStyle != OPTION_STYLE_AMERICAN) { 
        fprintf(stderr, "Unknown option style.\n");
        return;
    }

    if (optionType == OPTION_TYPE_PUT) {
        strcpy(initEntry, "PutInit");
        if (optionStyle == OPTION_STYLE_AMERICAN)
            strcpy(iterateEntry, "AmericanPutIterate");
    } else if (optionType == OPTION_TYPE_CALL) {
        strcpy(initEntry, "CallInit");
        if (optionStyle == OPTION_STYLE_AMERICAN)
            strcpy(iterateEntry, "AmericanCallIterate");
    } else {
        fprintf(stderr, "Unknown option type.\n");
        return;
    }

    prog = pugLoadProgram("binomial.cg", initEntry);
    if (prog == 0) {
        fprintf(stderr, "Could not compile binomial.cg:%s\n", initEntry);
        return;
    }

    iprog = pugLoadProgram("binomial.cg", iterateEntry);
    if (iprog == 0) {
        fprintf(stderr, "Could not compile binomial.cg:%s\n", iterateEntry);
        prog = 0;
        return;
    }

    // Associate uniforms with program parameters.
    if (!pugBindStream(prog, "stockPrice", priceBuf) ||
        !pugBindStream(prog, "strikePrice", strikeBuf) ||
        !pugBindStream(prog, "u", uBuf) ||
        !pugBindFloat( prog, "numSteps", (float)nsteps)) {
        fprintf(stderr, "Unable to set binomial init parameters.\n");
        prog = 0;
    }

    if (!pugBindStream(iprog, "Pu", puBuf) ||
        !pugBindStream(iprog, "Pd", pdBuf) ||
        (optionStyle == OPTION_STYLE_AMERICAN &&
         (!pugBindStream(iprog, "stockPrice", priceBuf) ||
          !pugBindStream(iprog, "strikePrice", strikeBuf) ||
          !pugBindStream(iprog, "u", uBuf)))) {
        fprintf(stderr, "Unable to set binomial iteration parameters.\n");
        prog = 0;
    }
}

BinomialPricing::~BinomialPricing()
{
    delete[] value;
    delete[] pudata;
    delete[] pddata;
    delete[] udata;
    delete resultBuf;
    delete priceBuf;
    delete strikeBuf;
    delete yearsBuf;
    delete volBuf;
    delete uBuf;
    delete puBuf;
    delete pdBuf;
    if (!pugCleanup())
        fprintf(stderr, "PUG cleanup failed\n");
}

//
// There is a good deal of shared code in the various pricing methods
// that could be efficiently shared via template magic, or less efficiently
// using extra function call(s).  We opt for simplicity over elegance.
//

float BinomialPricing::EuropeanPut(float S, float X, float T, float r, float v,
                                   float Pu, float Pd, float u)
{
    int i, j;

    float price = S * pow(u, -numSteps);
    for (i = 0; i <= numSteps; i++) {
        value[i] = max(X-price, 0);
        price *= u*u;
    }

    for (j = numSteps-1; j >= 0; j--) {
        for (i = 0; i <= j; i++) {
            // compute discounted expected value
            value[i] = (Pu*value[i+1] + Pd*value[i]);
        }
    }
    return value[0];
}

float BinomialPricing::EuropeanCall(float S, float X, float T, float r, float v,
                                   float Pu, float Pd, float u)
{
    int i, j;

    float price = S * pow(u, -numSteps);
    for (i = 0; i <= numSteps; i++) {
        value[i] = max(price-X, 0);
        price *= u*u;
    }

    for (j = numSteps-1; j >= 0; j--) {
        for (i = 0; i <= j; i++) {
            // compute discounted expected value
            value[i] = (Pu*value[i+1] + Pd*value[i]);
        }
    }
    return value[0];
}

float BinomialPricing::AmericanCall(float S, float X, float T, float r, float v,
                                   float Pu, float Pd, float u)
{
    int i, j;

    float price = S * pow(u, -numSteps);
    for (i = 0; i <= numSteps; i++) {
        value[i] = max(price-X, 0);
        price *= u*u;
    }

    for (j = numSteps-1; j >= 0; j--) {
        price = S * pow(u, -j);
        for (i = 0; i <= j; i++) {
            // compute immediate payoff and discounted expected value
            float payoff = price - X;
            float expected = (Pu*value[i+1] + Pd*value[i]);
            value[i] = max(payoff, expected);
            price *= u*u;
        }
    }
    return value[0];
}

float BinomialPricing::AmericanPut(float S, float X, float T, float r, float v,
                                   float Pu, float Pd, float u)
{
    int i, j;
    float price = S * pow(u, -numSteps);
    for (i = 0; i <= numSteps; i++) {
        value[i] = max(X-price, 0);
        price *= u*u;
    }

    for (j = numSteps-1; j >= 0; j--) {
        price = S * pow(u, -j);
        for (i = 0; i <= j; i++) {
            // compute immediate payoff and discounted expected value
            float payoff = X - price;
            float expected = (Pu*value[i+1] + Pd*value[i]);
            value[i] = max(expected, payoff);
            price *= u*u;
        }
    }
    return value[0];
}

bool BinomialPricing::CPUPrice(OptionParams &p, float *buf)
{
    if (value == NULL)
        return false;

    initProbData(p);

    if (optionStyle == OPTION_STYLE_EUROPEAN &&
        optionType == OPTION_TYPE_CALL) {
        for (int i = 0; i < p.n; i++)
            buf[i] = EuropeanCall(p.S[i], p.X[i], p.T[i], p.r, p.v[i],
                                  pudata[i], pddata[i], udata[i]);
    } else if (optionStyle == OPTION_STYLE_EUROPEAN &&
               optionType == OPTION_TYPE_PUT) {
        for (int i = 0; i < p.n; i++)
            buf[i] = EuropeanPut(p.S[i], p.X[i], p.T[i], p.r, p.v[i],
                                 pudata[i], pddata[i], udata[i]);
    } else if (optionStyle == OPTION_STYLE_AMERICAN &&
               optionType == OPTION_TYPE_CALL) {
        for (int i = 0; i < p.n; i++)
            buf[i] = AmericanCall(p.S[i], p.X[i], p.T[i], p.r, p.v[i],
                                  pudata[i], pddata[i], udata[i]);
    } else if (optionStyle == OPTION_STYLE_AMERICAN &&
               optionType == OPTION_TYPE_PUT) {
        for (int i = 0; i < p.n; i++)
            buf[i] = AmericanPut(p.S[i], p.X[i], p.T[i], p.r, p.v[i],
                                 pudata[i], pddata[i], udata[i]);
    } else {
        return false;
    }
    return true;
}

//
// Compute Pu, Pd, and u for each pricing problem.
//
void BinomialPricing::initProbData(OptionParams &params)
{
    int i;
    for (i = 0; i < params.n; i++) {
        float deltaT = params.T[i]/numSteps;    // change in time per step
        float R = exp(params.r*deltaT);         // riskless return per step
        float Rinv = (float)1./R;               // discount factor
        float u = udata[i] = exp(params.v[i]*sqrt(deltaT)); // up move
        float d = (float)1./u;           // corresponding down move
        float Pu = (R - d) / (u - d);    // pseudo-probability of upward move
        float Pd = (float)1. - Pu;       // pseudo-probability of downard move
        // We multiply the pseudo-probabilities by the discount factor
        // here in order to save a multiply in the various inner loops.
        pudata[i] = Rinv*Pu;             // 'discounted' pseudo-prob
        pddata[i] = Rinv*Pd;             // 'discounted' psuedo-prob
    }
}

bool BinomialPricing::GPUPrice(OptionParams &params, float *buf)
{
    PUGTarget target = PUG_FRONT, source = PUG_BACK;

    if (prog == 0 || numOpts != params.n || buf == 0)
        return false;

    // Initialize derived problem input
    initProbData(params);

    if (!pugInitBuffer(puBuf, pudata) ||
        !pugInitBuffer(pdBuf, pddata) ||
        !pugInitBuffer(uBuf, udata) ||
        !pugInitBuffer(strikeBuf, params.X) ||
        !pugInitBuffer(priceBuf, params.S)) {
        fprintf(stderr, "Unable to initialize binomial PUG buffers.\n");
        return false;
    }

    PUGRect range(0, width, 0, height);
    // Initialize expiration option values
    if (!pugRunProgram(prog, resultBuf, range, target)) {
        fprintf(stderr, "Couldn't run binomial PUG program.\n");
        return false;
    }


    // Iteratively compute option values backwards towards root of tree
    for (int j = numSteps-1; j >= 0; j--) {
        // ping-pong source and target
        target = (target == PUG_FRONT ? PUG_BACK : PUG_FRONT);
        source = (source == PUG_FRONT ? PUG_BACK : PUG_FRONT);

        range = PUGRect(0, width, 0, j+1);
        PUGRect oip1(0, width, 1, j+2);

        if (optionStyle == OPTION_STYLE_AMERICAN &&
            !pugBindFloat(iprog, "step", (float)j)) {
            fprintf(stderr, "Could not bind step.\n");
            return false;
        }

        if (!pugBindStream(iprog, "optval", resultBuf, source)) {
            fprintf(stderr, "Could not bind optval stream.\n");
            return false;
        }

        if (!pugBindDomain(iprog, "offsetplus1", oip1)) {
            fprintf(stderr, "Could not bind binomial PUG domain.\n");
            return false;
        }

        if (!pugRunProgram(iprog, resultBuf, range, target)) {
            fprintf(stderr, "Couldn't run binomial step.\n");
            return false;
        }
    }

    if (readback) {
        // Read results back to CPU
        pugWaitForThreads();
        range = PUGRect(0, width, 0, 1);
        pugReadMemory(buf, resultBuf, range);
    }
    return true;
}
