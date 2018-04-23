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

#ifndef SHUTIL_WORLEYIMPL_HPP 
#define SHUTIL_WORLEYIMPL_HPP

#include <cstdlib>
#include "ShSyntax.hpp"
#include "ShWorley.hpp"
#include "ShNoise.hpp"
#include "ShFunc.hpp"
#include "ShImage.hpp"
#include "ShTexCoord.hpp"

namespace ShUtil {

using namespace SH;

template<int D, typename T>
void GridGenFactory<D, T>::operator()(const ShGeneric<D, T> &p, Generator<D, T> result[]) const {
  ShAttrib<D, SH_TEMP, T> pCell = floor(p);

  // each set of two bits represents offset along one dimension
  // if the bit value = 0, offset by -1, 1 = no offset, 2 = offset by 1 
  int i, j, offsetBits;
  ShConstAttrib3f offsets(-1, 0, 1);
  int offsetSwiz[D];
  j = 0;
  for(offsetBits = 0; offsetBits < (1 << (D * 2)); ++offsetBits) {
    for(i = 0; i < D; ++i) {
      offsetSwiz[i] = ((offsetBits >> (i * 2)) & 3); 
      if(offsetSwiz[i] == 3) break;
    }
    if(i < D) continue;

    Generator<D, T> gen;
    result[j].offset = offsets.template swiz<D>(offsetSwiz); 
    SH_DEBUG_PRINT("Offset:" << result[j].offset); 
    result[j].cell = pCell + result[j].offset;
    makePos(result[j]);
    ++j;
  }
}


template<int D, typename T>
void DefaultGenFactory<D, T>::makePos(Generator<D, T> &g) const {
  g.pos = g.cell + cellnoise<D>(g.cell, m_useTexture);
}

template<int D, typename T>
void NullGenFactory<D, T>::makePos(Generator<D, T> &g) const {
  g.pos = g.cell + fillcast<D>(ShAttrib<1, SH_CONST, T>(0.5f)); 
}

template<int D, typename T>
LerpGenFactory<D, T>::LerpGenFactory(const ShGeneric<1, T> &time, bool useTexture)
  : m_time(time), m_useTexture(useTexture) {}

template<int D, typename T>
void LerpGenFactory<D, T>::makePos(Generator<D, T> &g) const {
  ShAttrib<1, SH_TEMP, T> lastTime = floor(m_time);
  ShAttrib<1, SH_TEMP, T> timeOffset = frac(m_time);
  ShAttrib<D + 1, SH_TEMP, T> offsetCell;

  offsetCell = fillcast<D+1>(g.cell);
  offsetCell(D) = lastTime;
  ShAttrib<D, SH_TEMP, T> p1 = cellnoise<D>(offsetCell, m_useTexture); 

  offsetCell(D) += 1;
  ShAttrib<D, SH_TEMP, T> p2 = cellnoise<D>(offsetCell, m_useTexture); 

  g.pos = g.cell + lerp(timeOffset, p2, p1);
}

template<int N, int D, typename T, typename P1, typename P2>
CombinedPropFactory<N, D, T, P1, P2>::CombinedPropFactory(const P1 *propFactory1, const P2 *propFactory2)
  : m_propFactory1(propFactory1), m_propFactory2(propFactory2) {}

template<int N, int D, typename T, typename P1, typename P2>
ShGeneric<N, T> CombinedPropFactory<N, D, T, P1, P2>::operator()(
    const ShGeneric<D, T> &p, const Generator<D, T> &g) const {
  return join((*m_propFactory1)(p, g), (*m_propFactory2)(p, g));
}

template<typename P1, typename P2>
PropertyFactory<P1::NUM_PROPS + P2::NUM_PROPS, P1::DIM, typename P1::PropType>*
combine(const P1 *propFactory1, const P2 *propFactory2) {
  const int N = P1::NUM_PROPS + P2::NUM_PROPS;
  return new CombinedPropFactory<N, P1::DIM, typename P1::PropType, P1, P2>(propFactory1, propFactory2);
}

template<int D, typename T>
ShGeneric<1, T> DistSqPropFactory<D, T>::operator()(
    const ShGeneric<D, T> &p, const Generator<D, T> &g) const {
  ShAttrib<D, SH_TEMP, T> delta = p - g.pos;
  return delta | delta; 
}

template<int D, typename T>
ShGeneric<1, T> Dist_1PropFactory<D, T>::operator() (
    const ShGeneric<D, T> &p, const Generator<D, T> &g) const {
  ShAttrib<D, SH_TEMP, T> delta = abs(p - g.pos);
  return delta | fillcast<D>(ShConstAttrib1f(1.0f)); 
}

template<int D, typename T>
ShGeneric<1, T> Dist_InfPropFactory<D, T>::operator() (
    const ShGeneric<D, T> &p, const Generator<D, T> &g) const {
  ShAttrib<D, SH_TEMP, T> delta = abs(p - g.pos);

  // TODO replace with tuple max function when implemented 
  ShAttrib<1, SH_TEMP, T> result = 0; 
  for(int i = 0; i < D; ++i) result = max(result, delta(i));
  return result;
}

template<int D, typename T>
ShGeneric<D + 1, T> DistSqGradientPropFactory<D, T>::operator() (
    const ShGeneric<D, T> &p, const Generator<D, T> &g) const {
  ShAttrib<D + 1, SH_TEMP, T> result;
  ShAttrib<D, SH_TEMP, T> delta = p - g.pos;

  result(0) = delta | delta; 
  for(int i = 1; i < D + 1; ++i) {
    result(i) = delta(i-1) * result(0);
  }
  return result;
};

template<int D, typename T>
ShGeneric<D + 1, T> Dist_1GradientPropFactory<D, T>::operator() (
    const ShGeneric<D, T> &p, const Generator<D, T> &g) const {

  ShAttrib<D + 1, SH_TEMP, T> result;
  ShAttrib<D, SH_TEMP, T> delta = p - g.pos;

  ShAttrib<1, SH_CONST, T> ONE(1);
  result(0) = abs(delta) | fillcast<D>(ONE);
  for(int i = 1; i < D + 1; ++i) { // TODO switch to a swizzled version
    result(i) = cond(delta(i-1), ONE, -ONE); 
  }
  return result;
};

template<int D, typename T>
ShGeneric<D + 1, T> Dist_InfGradientPropFactory<D, T>::operator() (
    const ShGeneric<D, T> &p, const Generator<D, T> &g) const {
  ShAttrib<D + 1, SH_TEMP, T> result;
  ShAttrib<D, SH_TEMP, T> delta = p - g.pos;
  ShAttrib<D, SH_TEMP, T> absDelta = abs(delta); 

  result(0) = 0;
  for(int i = 0; i < D; ++i) result(0) = max(result(0), absDelta(i)); 

  // TODO remove this when constant folding is available
  ShAttrib<1, SH_CONST, T> ONE(1);
  for(int i = 1; i < D + 1; ++i) { // TODO switch to a swizzled version
    result(i) = (absDelta(i-1) >= result(0)) * cond(delta(i-1), ONE, -ONE); 
  }
  return result;
};

template<int N, int D, typename T>
ShGeneric<N, T> CellnoisePropFactory<N, D, T>::operator() (
    const ShGeneric<D, T> &p, const Generator<D, T> &g) const {
  // TODO remove frac once Issue58 fixed
  return frac(cellnoise<N>(g.cell, m_useTexture));
};

template<typename TexType, typename T>
Tex2DPropFactory<TexType, T>::Tex2DPropFactory(
    const ShBaseTexture2D<TexType> &tex, const ShGeneric<1, T> &scale)
  : m_tex(tex), m_scale(scale), invScale(ShConstAttrib2f(1.0f, 1.0f) / tex.size()) {}

/* Moved to ShWorley.hpp
template<typename TexType, typename T>
ShGeneric<TexType::typesize, T> Tex2DPropFactory<TexType, T>::operator()(
    const ShGeneric<2, T> &p, const Generator<2, T> &g) const {
}
*/

template<int N, int K, int P, typename T>
void kSelect(const ShGeneric<P, T> vals[N], ShGeneric<K, T> result[N], float LARGE = 1e10) {
  result[0] = fillcast<K>(ShConstAttrib1f(LARGE));

  int i, j, k, l;
  int shiftswiz[K];
  ShAttrib1f c;
  for(i = 0; i < K; ++i) shiftswiz[i] = (i == 0 ? 0 : i - 1);

  // insert into result one by one
  for(i = 0; i < P; ++i) {
    // check if smaller than the first one
    c = vals[0](i) < result[0](0);
    for(j = 0; j < N; ++j) {
      result[0] = lerp(c, result[0].template swiz<K>(shiftswiz), result[0]); 
      result[0](0) = lerp(c, vals[0](i), result[0](0));
    }

    // check for the other ones
    for(j = 1; j < K; ++j) {
      c = (result[0](j-1) < vals[0](i)) * (vals[0](i) < result[0](j));
      for(k = 0; k < N; ++k) {
        for(l = 0; l < K; ++l) shiftswiz[l] = l + ( l >= j ? -1 : 0); 
        result[k] = lerp(c, result[k].template swiz<K>(shiftswiz), result[k]);
        result[k](j) = lerp(c, vals[k](i), result[k](j));
      }
    }
  }
}

template<int K, int L, int P, int D, typename T>
void worley(ShGeneric<K, T> result[], const ShGeneric<D, T> &p,
    const GeneratorFactory<P, D, T> *genFactory,
    const PropertyFactory<L, D, T> *propFactory) {

  int i, j;
  Generator<D, T> generators[P];
  ShAttrib<P, SH_TEMP, T> props[L]; 
  ShAttrib<L, SH_TEMP, T> propTemp; 

  (*genFactory)(p, generators); // make generators
  for(i = 0; i < P; ++i) {
    propTemp = (*propFactory)(p, generators[i]);
    for(j = 0; j < L; ++j) props[j](i) = propTemp(j);
  }

  // sort points & gradients by distance
  groupEvenOddSort<L>(props); 

  // weighted sum of basis function values to get final result
  for(j = 0; j < L; ++j) result[j] = cast<K>(props[j]);
}

template<int K, int D, typename T>
#ifdef WIn32
ShGeneric<K, T> worley(const ShGeneric<D, T> &p, bool useTexture=true) {
#else
ShGeneric<K, T> worley(const ShGeneric<D, T> &p, bool useTexture=true) {
#endif
  DefaultGenFactory<D, T> genFactory(useTexture);
  DistSqPropFactory<D, T> propFactory;
  ShAttrib<K, SH_TEMP, T> result;
  worley(&result, p, &genFactory, &propFactory);
  return result;
}

template<int K, int D, typename T>
#ifdef WIn32
ShProgram shWorley(bool useTexture=true) {
#else
ShProgram shWorley(bool useTexture) {
#endif
  DefaultGenFactory<D, T> genFactory(useTexture);
  DistSqPropFactory<D, T> propFactory;
  return shWorley<K>(&genFactory, &propFactory); 
}

template<int K, int L, int P, int D, typename T>
ShProgram shWorley(const GeneratorFactory<P, D, T> *genFactory,
        const PropertyFactory<L, D, T> *propFactory) {
  ShProgram program = SH_BEGIN_PROGRAM() {
    ShTexCoord<D, SH_INPUT, T> SH_DECL(texcoord);

    ShAttrib<K, SH_OUTPUT, T> result[L]; 
    worley(&result[0], texcoord, genFactory, propFactory); 
  } SH_END_PROGRAM;
  return program;
}

}

#endif
