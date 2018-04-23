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
#ifndef SHUTIL_WORLEY_HPP 
#define SHUTIL_WORLEY_HPP

#include <vector>
#include "ShAttrib.hpp"
#include "ShColor.hpp"
#include "ShTexture.hpp"

namespace ShUtil {

using namespace SH;
/** \file ShWorley.hpp
 * This is an implementation of 2D ShWorley texture.  A Worley
 * texture is made by distributing a set of points in a space and
 * taking a weighted sum of basis functions for the k-nearest 
 * neighbours of a lookup point p.  These basis functions can 
 * be the distance itself from p to the adjacent generators (the
 * typical Worley function), or any other information.  For example,
 * we can attach colour information or refraction indicies to the generators.
 *
 * This Worley implementation is split into two parts.
 * The first part is a Generator Factory that produces
 * a fixed number of candidate generators around a lookup point p.
 *
 * A Property Factory is attaches properties to each Generator.
 * Currently, there is no internal Sh representation for a struct of
 * Sh types, so properties are generated into a long N-tuple.
 * The generators are sorted by the first component in the N-tuple, 
 * so this should typically be a distance.
 *
 * In the future, when it's possible to "call" an ShProgram object
 * directly, Property Factories should actually be ShProgram objects
 * that take a integer cell position as an input and output 
 * the desired properties. 
 *
 * The worley function itself sorts the generators based on
 * the first component in its properties and returns a N-tuple
 * containing the weighted sum of all properties.
 */

// Integer Power using template metaprogramming...
// TODO find a better way than this.
// sort on N-tuples needs to know at C++ compile time exactly 
template<int N, int M> 
struct _IntPow {
  static const int value = N * _IntPow<N, M-1>::value;
};

template<int N> 
struct _IntPow<N, 1> {
  static const int value = N;
};


// A Generator point represents the position of a generator relative
// to the cell origin of a lookup position. 
//
template<int D, typename T>
struct Generator {
  Generator() {}
  ShAttrib<D, SH_TEMP, T> pos; // position 
  ShAttrib<D, SH_TEMP, T> offset; // offset of the cell relative to lookup point's cell
  ShAttrib<D, SH_TEMP, T> cell; // integer cell (this should actually be typed T = int, but not possible yet) 
};

// Generator Factory 
// A Point Generator must implement a function that 
// sets a Generator array with P candidate generator points. 
template<int P, int D, typename T>
struct GeneratorFactory {
  static const int NUM_POINTS = P;
  virtual ~GeneratorFactory() {}
  virtual void operator()(const ShGeneric<D, T> &p, Generator<D, T> result[]) const = 0;
};

// The foundation for the regular grid-based generators
template<int D, typename T>
struct GridGenFactory: public GeneratorFactory<_IntPow<3, D>::value, D, T> { 
  void operator()(const ShGeneric<D, T> &p, Generator<D, T> result[]) const;

  private:
    // given a Generator that has its cell and offset filled in, this function
    // generates the actual pos.
    virtual void makePos(Generator<D, T> &g) const = 0; 
};

// The default generator - uses a uniform integer grid with one
// generator point in each grid cell.
// If PointType is d-dimensional, it generates 3^d points from 
// p's cell and all adjacent grid cells.
template<int D, typename T>
struct DefaultGenFactory: public GridGenFactory<D, T> { 
  DefaultGenFactory(bool useTexture): m_useTexture(useTexture) {}

  private:
    void makePos(Generator<D, T> &g) const;
    bool m_useTexture;
};

// generates in a 2D tiling of the plane by squares (but each
// row is offset by 0.5 from the previous one, giving a hexagonal 
// structure with less adjacent cells than the grid lookup)
/*
template<typename T>
struct HexGenFactory {
  HexGenFactory(bool useTexture): m_useTexture(useTexture) {}
  void operator()(const ShGeneric<D, T> &p, Generator<D, T> result[]) const;

  private:
    bool m_useTexture;
};
*/


// a null point generator - just makes a non-jittered grid
// of generators (good for debugging, or certain kinds of patterns...)
template<int D, typename T>
struct NullGenFactory: public GridGenFactory<D, T> { 
  private:
    void makePos(Generator<D, T> &g) const;
};

// An animating point generator - uses same uniform integer grid
// method as the default, except that it linearly interpolates
// between pairs of generator point sets, moving through a sequence
// of point sets as time increases. 
template<int D, typename T>
struct LerpGenFactory: GridGenFactory<D, T> {
  LerpGenFactory(const ShGeneric<1, T> &time, bool useTexture);

  private:
    void makePos(Generator<D, T> &g) const;
    const ShGeneric<1, T> &m_time;
    bool m_useTexture;
};

// Property Factory
// This takes a Generator and attaches N properties to it.
// The first property in the result is used to sort the generator points 
// (and so should probably be a distance metric).
template<int N, int D, typename T>
struct PropertyFactory {
    static const int NUM_PROPS = N;
    static const int DIM = D;
    typedef T PropType; 

    virtual ~PropertyFactory() {} 
    virtual ShGeneric<N, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const = 0; 
};

// when ShProgram objects can be called like functions, this will 
// no longer be necessary as it is analogous to the Algebra combine operator.
template<int N, int D, typename T, typename P1, typename P2>
struct CombinedPropFactory: 
public PropertyFactory<N, D, T> {
    CombinedPropFactory(const P1 *propFactory1, 
                                const P2 *propFactory2);

    ShGeneric<N, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const; 

  private:
    const P1* m_propFactory1;
    const P2* m_propFactory2;
};


// standard distance based property factories 
// Re-write later to take function pointer (or ShProgram object)
// to a distance function.
template<int D, typename T>
struct DistSqPropFactory: public PropertyFactory<1, D, T> {
  ShGeneric<1, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const; 
};

template<int D, typename T>
struct Dist_1PropFactory: public PropertyFactory<1, D, T> {
  ShGeneric<1, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const; 
};

template<int D, typename T>
struct Dist_InfPropFactory: public PropertyFactory<1, D, T> {
  ShGeneric<1, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const; 
};

template<int D, typename T>
struct DistSqGradientPropFactory: public PropertyFactory<D + 1, D, T> {
  ShGeneric<D + 1, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const; 
};

template<int D, typename T>
struct Dist_1GradientPropFactory: public PropertyFactory<D + 1, D, T> {
  ShGeneric<D + 1, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const; 
};

template<int D, typename T>
struct Dist_InfGradientPropFactory: public PropertyFactory<D + 1, D, T> {
  ShGeneric<D + 1, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const; 
};

template<int N, int D, typename T>
struct CellnoisePropFactory: public PropertyFactory<N, D, T> {
  CellnoisePropFactory(bool useTexture): m_useTexture(useTexture) {}
  ShGeneric<N, T> operator()(const ShGeneric<D, T> &p, const Generator<D, T> &g) const; 

  private:
    bool m_useTexture;
};

template<typename TexType, typename T>
struct Tex2DPropFactory: public PropertyFactory<TexType::typesize, 2, T> {
  Tex2DPropFactory(const ShBaseTexture2D<TexType> &tex, const ShGeneric<1, T> &scale);
  ShGeneric<TexType::typesize, T> operator()(const ShGeneric<2, T> &p, const Generator<2, T> &g) const
  {
    // Moved here from WorleyImpl.hpp because MSVC gets confused otherwise
    return m_tex(frac(g.cell * invScale * m_scale)) * ShConstAttrib1f(1.0f);
  }

  private:
    const ShBaseTexture2D<TexType> &m_tex;
    const ShGeneric<1, T> &m_scale;
    ShConstAttrib2f invScale;
    // TODO remove invScale and restrict to RECT textures later 
};

#ifndef WIN32
// MSVC++ .NET does not recognize these as being the same as their implementation.
template<typename P1, typename P2>
PropertyFactory<P1::NUM_PROPS + P2::NUM_PROPS, P1::DIM, typename P1::PropType>*
combine(const P1 *propFactory1, const P2 *propFactory2);

/** \brief Worley texture generator.
 * This uses the DefaultGeneratorFactory and DistSqPropFactory 
 * TODO allow arbitrary distance function
 * @{
 */
template<int K, int D, typename T>
ShGeneric<K, T> worley(const ShGeneric<D, T> &p, bool useTexture = true); 
//@}

/** \brief Worley texture generator.
 * This uses a GeneratorFactory and PropertyFactory of your choice.
 */
template<int K, int L, int P, int D, typename T>
void worley(ShGeneric<K, T> result[], const ShGeneric<D, T> &p, 
    const GeneratorFactory<P, D, T> *genFactory,
    const PropertyFactory<L, D, T> *propFactory);

/** Makes a shader that takes 
 *  IN(1) ShTexCoord<D,T> texcoord; // texture lookup coordinates
 *
 * TODO make Output a struct of some kind when Sh supports structs
 *  OUT(0) ShAttrib<K, T> result[N]; // properties of k-nearest neighbours 
 * @{
 */
template<int K, int D, typename T>
ShProgram shWorley(bool useTexture);

template<int K, int N, int P, int D, typename T>
ShProgram shWorley(const GeneratorFactory<P, D, T> *genFactory,
    const PropertyFactory<N, D, T> *propFactory);
//@}

#endif // ifndef WIN32

} // namespace ShUtil

#include "ShWorleyImpl.hpp"

#endif
