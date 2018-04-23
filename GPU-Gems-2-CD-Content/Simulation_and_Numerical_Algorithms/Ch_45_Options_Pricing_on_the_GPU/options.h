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
#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include "pug.h"

typedef enum {
    OPTION_STYLE_UNKNOWN = 0,
    OPTION_STYLE_EUROPEAN,
    OPTION_STYLE_AMERICAN
} OptionStyle;

typedef enum {
    OPTION_TYPE_UNKNOWN = 0,
    OPTION_TYPE_PUT,
    OPTION_TYPE_CALL
} OptionType;

class OptionParams {
public:
    OptionParams();
    OptionParams(int num);
    ~OptionParams();
    int n;
    float *S, *X, *T, *v, r;
};

class OptionPricing {
public:
    OptionPricing(void) {
        optionType = OPTION_TYPE_UNKNOWN;
        optionStyle = OPTION_STYLE_UNKNOWN;
        readback = true;
    }
    virtual ~OptionPricing(void) {}
    virtual bool CPUPrice(OptionParams &params, float *buf) = 0;
    virtual bool GPUPrice(OptionParams &params, float *buf) = 0;
    virtual const char *GetName(void) = 0;
    OptionType optionType;
    OptionStyle optionStyle;
    bool readback;
};

#endif // _OPTIONS_H_
