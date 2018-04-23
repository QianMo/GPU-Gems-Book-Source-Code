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
//
// GPU-based options pricing demo
//

#include <math.h>
#include <stdio.h>
#include <windows.h>
#include <time.h>

#include "options.h"
#include "binomialprice.h"
#include "bsprice.h"

int NumOptions = 1024;
int NumSteps = 1024;
bool EuropeanOption = true;
bool PutOption = true;
bool DoGPU = true;
bool DoCPU = true;
bool DoReadback = true;
int NumTests = 1;
float RiskFree = .02f;

OptionType optionType = OPTION_TYPE_PUT;
OptionStyle optionStyle = OPTION_STYLE_EUROPEAN;

typedef enum {
    METHOD_UNKNOWN,
    METHOD_BINOMIAL,
    METHOD_BLACKSCHOLES,
} PricingMethod;

static char *OptionStyleName[] = {
    "unknown",
    "European",
    "American"
};

static char *OptionTypeName[] = {
    "unknown",
    "put",
    "call"
};


PricingMethod Method = METHOD_BINOMIAL;

inline float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.f - t) * low + t * high;
}

static void initParams(OptionParams &p)
{
    for (int i = 0; i < p.n; ++i) {
        p.S[i] = RandFloat(1.f, 30.f);
        p.X[i] = p.S[i] * RandFloat(0.5f, 2.f);
        p.T[i] = RandFloat(.25f, 4.f);
        p.v[i] = RandFloat(0.2f, .8f);
        p.r = RiskFree;
    }
}

double timeDelta(LARGE_INTEGER s, LARGE_INTEGER e, LARGE_INTEGER f)
{
    __int64 start = s.QuadPart, end = e.QuadPart, freq = f.QuadPart;

    return double(end-start)/double(freq);
}

bool Price(OptionPricing *pricing, OptionParams &params, float *buf, bool gpu)
{
    bool ret;
    // Perform one pricing iteration outside of the timing
    // loop in order to amortize away buffer creation and the like.
    if (gpu)
        ret = pricing->GPUPrice(params, buf);
    else
        ret = pricing->CPUPrice(params, buf);

    if (ret == false)
        return false;

    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);
    pricing->readback = DoReadback;
    QueryPerformanceCounter(&start);
    if (gpu) {
        for (int i = 0; i < NumTests; i++) {
            ret = pricing->GPUPrice(params, buf);
        }
    } else {
        for (int i = 0; i < NumTests; i++) {
            ret = pricing->CPUPrice(params, buf);
        }
    }
    if (gpu)
        pugWaitForThreads();
    QueryPerformanceCounter(&end);
    double time = timeDelta(start, end, freq);

    if (ret == false)
        return false;

    double persec = (double)(NumTests*params.n)/(double)time;
    printf("%s Elapsed time for %d %s: %.3f (%.3f K/sec)\n",
            gpu ? "GPU" : "CPU",
            NumTests, NumTests == 1 ? "run" : "runs", time, persec/1.e3);
    fflush(stdout);
    return true;
}

void usage(void)
{
    fprintf(stderr, "usage: options [-gpu|-cpu] [-a|-e] [-put|-call] [-bs|-bin n] [-rb] [-n noptions] [-sn sqrtnoptions] [-t ntests] [-seed seed] [-h]\n");
}

void parseArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-')
            break;
        if (!strcmp(argv[i], "-a")) {
            optionStyle = OPTION_STYLE_AMERICAN;
        } else if (!strcmp(argv[i], "-e")) {
            optionStyle = OPTION_STYLE_EUROPEAN;
        } else if (!strcmp(argv[i], "-gpu")) {
            DoGPU = !DoGPU;
        } else if (!strcmp(argv[i], "-cpu")) {
            DoCPU = !DoCPU;
        } else if (!strcmp(argv[i], "-rb")) {
            DoReadback = !DoReadback;
        } else if (!strcmp(argv[i], "-n")) {
            NumOptions = atoi(argv[i+1]);
            if (i+1 == argc) {
                usage();
                exit(1);
            }
            i++;
        } else if (!strcmp(argv[i], "-sn")) {
            NumOptions = atoi(argv[i+1]);
            NumOptions *= NumOptions;
            if (i+1 == argc) {
                usage();
                exit(1);
            }
            i++;
        } else if (!strcmp(argv[i], "-bs")) {
            Method = METHOD_BLACKSCHOLES;
        } else if (!strcmp(argv[i], "-bin")) {
            Method = METHOD_BINOMIAL;
            if (i+1 == argc) {
                usage();
                exit(1);
            }
            NumSteps = atoi(argv[i+1]);
            i++;
        } else if (!strcmp(argv[i], "-put")) {
            optionType = OPTION_TYPE_PUT;
        } else if (!strcmp(argv[i], "-call")) {
            optionType = OPTION_TYPE_CALL;
        } else if (!strcmp(argv[i], "-t")) {
            NumTests = atoi(argv[i+1]);
            i++;
        } else if (!strcmp(argv[i], "-h")) {
            usage();
            exit(0);
        } else if (!strcmp(argv[i], "-seed")) {
            srand(atoi(argv[i+1]));
            i++;
        } else {
            usage();
            exit(1);
        }
    }

    if (i < argc) {
        usage();
        exit(1);
    }
}

OptionPricing *CreateMethod(PricingMethod meth, int nsteps, int nopts)
{
    OptionPricing *pricing = NULL;

    if (meth == METHOD_BLACKSCHOLES) {
        if (optionStyle == OPTION_STYLE_EUROPEAN) {
            pricing = new BlackScholesPricing(optionStyle, optionType, nopts);
        } else {
            fprintf(stderr, "Black-Scholes only supports European options.\n");
        }
    } else if (meth == METHOD_BINOMIAL) {
        pricing = new BinomialPricing(optionStyle, optionType, nopts, nsteps);
    } else {
        fprintf(stderr, "Unknown pricing method: %d\n", Method);
    }
    return pricing;
}

int main(int argc, char **argv)
{
    OptionPricing *pricing;

    parseArgs(argc, argv);

    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    OptionParams params(NumOptions);

    initParams(params);

    float *cpubuf = new float[NumOptions];
    float *gpubuf = new float[NumOptions];

    pricing = CreateMethod(Method, NumSteps, NumOptions);

    if (pricing == NULL)
        return 1;

    QueryPerformanceCounter(&end);
    double time = timeDelta(start, end, freq);
    printf("Initialization: %.3fs\n", time);

    printf("%d %s %s %s, %s", NumOptions,
                          OptionStyleName[(int)optionStyle],
                          OptionTypeName[(int)optionType],
                          NumOptions == 1 ? "option" : "options",
                          pricing->GetName());
    if (Method == METHOD_BINOMIAL)
        printf(" (%d steps)", NumSteps);
    printf("\n");
    fflush(stdout);

    bool success;

    if (DoCPU) {
        success = Price(pricing, params, cpubuf, false);
        if (!success) {
            fprintf(stderr, "CPU pricing failed.\n");
            return 1;
        }
    }

    if (DoGPU) {
        success = Price(pricing, params, gpubuf, true);
        if (!success) {
            fprintf(stderr, "GPU pricing failed.\n");
            return 1;
        }
    }

    if (!DoGPU || !DoCPU)
        return 0;

    double diff = 0.;
    double gpusum = 0.;
    double cpusum = 0.;
    double maxdiff = 0.;
    double rms = 0.;
    for (int i = 0; i < params.n; i++) {
        cpusum += cpubuf[i];
        gpusum += gpubuf[i];
        diff = fabs(cpubuf[i] - gpubuf[i]);
        double reldiff = diff/max(fabs(cpubuf[i]), fabs(gpubuf[i]));
        if (reldiff > maxdiff) maxdiff = reldiff;
        rms += diff*diff;
    }
    rms = sqrt(rms/params.n);
    printf("RMS diff: %g, Max diff: %.3f%%, GPU Avg: %g, CPU Avg: %g\n",
            rms, maxdiff*100., gpusum/params.n, cpusum/params.n);
    return 0;
}
