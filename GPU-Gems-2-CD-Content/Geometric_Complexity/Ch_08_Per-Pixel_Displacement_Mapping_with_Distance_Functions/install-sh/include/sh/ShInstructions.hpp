// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Michael D. McCool
// 
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
// 
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 
// 1. The origin of this software must not be misrepresented; you must
// not claim that you wrote the original software. If you use this
// software in a product, an acknowledgment in the product documentation
// would be appreciated but is not required.
// 
// 2. Altered source versions must be plainly marked as such, and must
// not be misrepresented as being the original software.
// 
// 3. This notice may not be removed or altered from any source
// distribution.
//////////////////////////////////////////////////////////////////////////////
#ifndef SHINSTRUCTIONS_HPP
#define SHINSTRUCTIONS_HPP

#include "ShDllExport.hpp"
#include "ShVariable.hpp"

namespace SH {

/** @defgroup instructions Intermediate-representation instructions
 * You probably don't ever need to call any of these.
 * They are mostly used internally.
 * Instead, use the library functions provided by ShLib.hpp.
 * @{
 */

SH_DLLEXPORT
void shASN(ShVariable& dest, const ShVariable& src);
//void shNEG(ShVariable& dest, const ShVariable& src);
SH_DLLEXPORT
void shADD(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shMUL(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shDIV(ShVariable& dest, const ShVariable& a, const ShVariable& b);

SH_DLLEXPORT
void shSLT(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shSLE(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shSGT(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shSGE(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shSEQ(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shSNE(ShVariable& dest, const ShVariable& a, const ShVariable& b);

SH_DLLEXPORT
void shABS(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shACOS(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shASIN(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shATAN(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shATAN2(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shCBRT(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shCEIL(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shCOS(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shCMUL(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shCSUM(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shDOT(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shDX(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shDY(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shEXP(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shEXP2(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shEXP10(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shFLR(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shFRAC(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shLOG(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shLOG2(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shLOG10(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shLRP(ShVariable& dest, const ShVariable& alpha,
           const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shMAD(ShVariable& dest, const ShVariable& a,
           const ShVariable& b, const ShVariable& c);
SH_DLLEXPORT
void shMAX(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shMIN(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shMOD(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shPOW(ShVariable& dest, const ShVariable& a, const ShVariable& b);
SH_DLLEXPORT
void shRCP(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shRND(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shRSQ(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shSGN(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shSIN(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shSQRT(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shTAN(ShVariable& dest, const ShVariable& a);

SH_DLLEXPORT
void shNORM(ShVariable& dest, const ShVariable& a);
SH_DLLEXPORT
void shXPD(ShVariable& dest, const ShVariable& a, const ShVariable& b);

SH_DLLEXPORT
void shCOND(ShVariable& dest, const ShVariable& cond,
            const ShVariable& a, const ShVariable& b);

// interval arithmetic ops 
SH_DLLEXPORT
void shLO(ShVariable &dest, const ShVariable &a);

SH_DLLEXPORT
void shHI(ShVariable &dest, const ShVariable &a);

SH_DLLEXPORT
void shSETLO(ShVariable &dest, const ShVariable &a);

SH_DLLEXPORT
void shSETHI(ShVariable &dest, const ShVariable &a);

SH_DLLEXPORT
void shKIL(const ShVariable& cond);

/*@}*/

}

#endif
