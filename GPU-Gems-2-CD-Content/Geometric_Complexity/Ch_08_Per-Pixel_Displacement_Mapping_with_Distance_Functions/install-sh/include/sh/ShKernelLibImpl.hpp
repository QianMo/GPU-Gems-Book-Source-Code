// Sh: A GPU metaprogramming language.
//
// Copyright (c) 2003 University of Waterloo Computer Graphics Laboratory
// Project administrator: Michael D. McCool
// Authors: Zheng Qin, Stefanus Du Toit, Kevin Moule, Tiberiu S. Popa,
//          Bryan Chan, Michael D. McCool
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
#ifndef SHUTIL_KERNELLIBIMPL_HPP 
#define SHUTIL_KERNELLIBIMPL_HPP 

#include <sstream>
#include "ShSyntax.hpp"
#include "ShPosition.hpp"
#include "ShManipulator.hpp"
#include "ShAlgebra.hpp"
#include "ShProgram.hpp"
#include "ShNibbles.hpp"
#include "ShKernelLib.hpp"
#include "ShFunc.hpp"
#include "ShTexCoord.hpp"
#include "ShVector.hpp"
#include "ShPoint.hpp"
#include "ShPosition.hpp"
#include "ShNormal.hpp"

/** \file ShKernelLibImpl.hpp
 * This is an implementation of useful kernels and nibbles (simple kernels).
 */

namespace ShUtil {

using namespace SH;

template<int N, ShBindingType Binding, typename T>
ShProgram ShKernelLib::shVsh(const ShMatrix<N, N, Binding, T> &mv,
                             const ShMatrix<N, N, Binding, T> &mvp,
                             int numTangents, int numLights)
{
  int i;
  ShProgram generalVsh = SH_BEGIN_VERTEX_PROGRAM {
    // INPUTS
    ShInputTexCoord2f SH_NAMEDECL(u, "texcoord");  
    ShInputNormal3f SH_NAMEDECL(nm, "normal");     
    ShVector3f tgt; 
    ShVector3f tgt2; 
    if(numTangents > 0) {
      ShInputVector3f SH_NAMEDECL(inTangent, "tangent");
      tgt = inTangent;
      if( numTangents > 1) {
        ShInputVector3f SH_NAMEDECL(inTangent2, "tangent2");
        tgt2 = inTangent2;
      }  else {
        tgt2 = cross(nm, tgt);
      }
    }
    ShInputPoint3f* lpv = new ShInputPoint3f[numLights];                 
    for(i = 0; i < numLights; ++i) lpv[i].name(makeName("lightPos", i));
    ShInputPosition4f SH_NAMEDECL(pm, "posm");     

    // OUTPUTS
    ShOutputTexCoord2f SH_NAMEDECL(uo, "texcoord");  
    ShOutputPoint3f SH_NAMEDECL(pv, "posv");         
    ShOutputPoint4f SH_NAMEDECL(pmo, "posm");
    
    // VCS outputs
    ShOutputNormal3f SH_NAMEDECL(nv, "normal");      
    ShOutputVector3f SH_NAMEDECL(tv, "tangent");
    ShOutputVector3f SH_NAMEDECL(tv2, "tangent2");
    ShOutputVector3f SH_NAMEDECL(vv, "viewVec");     
    ShOutputVector3f* hv = new ShOutputVector3f[numLights];    
    for(i = 0; i < numLights; ++i) hv[i].name(makeName("halfVec", i).c_str());
    ShOutputVector3f* lv = new ShOutputVector3f[numLights];    
    for(i = 0; i < numLights; ++i) lv[i].name(makeName("lightVec", i).c_str()); 

    ShOutputPoint3f* lpo = new ShOutputPoint3f[numLights];    
    for(i = 0; i < numLights; ++i) lpo[i].name(makeName("lightPos", i).c_str()); 

    // TCS outputs
    ShOutputNormal3f SH_NAMEDECL(nvt, "normalt");      
    ShOutputVector3f SH_NAMEDECL(vvt, "viewVect");     
    ShOutputVector3f* hvt = new ShOutputVector3f[numLights];    
    for(i = 0; i < numLights; ++i) hvt[i].name(makeName("halfVect", i).c_str());
    ShOutputVector3f* lvt = new ShOutputVector3f[numLights];    
    for(i = 0; i < numLights; ++i) lvt[i].name(makeName("lightVect", i).c_str()); 

    ShOutputPosition4f SH_NAMEDECL(pd, "posh");      

    uo = u;
    pv = (mv | pm)(0,1,2); 
    pmo = pm;

    // VCS outputs
    nv = normalize(mv | nm); 
    vv = normalize(-pv);
    for(i = 0; i < numLights; ++i) {
      lv[i] = normalize(lpv[i] - pv); 
      hv[i] = normalize(vv + lv[i]); 
      lpo[i] = lpv[i];
    }

    // TCS outputs
    tv = mv | tgt;
    tv2 = mv | tgt2;
    nvt = normalize(changeBasis(nv, tv, tv2, nv));
    vvt = normalize(changeBasis(nv, tv, tv2, vv));
    for(i = 0; i < numLights; ++i) {
      hvt[i] = normalize(changeBasis(nv, tv, tv2, hv[i]));
      lvt[i] = normalize(changeBasis(nv, tv, tv2, lv[i])); 
    }

    pd = mvp | pm;

    delete [] lvt;
    delete [] hvt;
    delete [] lpo;
    delete [] lv;
    delete [] hv;
    delete [] lpv;
  } SH_END;
  return generalVsh;
}

}

#endif
