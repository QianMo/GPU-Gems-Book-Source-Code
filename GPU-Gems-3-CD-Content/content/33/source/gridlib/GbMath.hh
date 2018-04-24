/*----------------------------------------------------------------------
|
| $Id: GbMath.hh,v 1.15 2005/07/22 08:22:30 DOMAIN-I15+prkipfer Exp $
|
+---------------------------------------------------------------------*/

#ifndef  GBMATH_HH
#define  GBMATH_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#ifdef IRIX
#include <math.h>
#else
#include <cmath>
#endif

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

/*!
  \brief Class for numerical operations - abstraction of libm.so
  \author Peter Kipfer
  $Revision: 1.15 $
  $Date: 2005/07/22 08:22:30 $

  \note This is a version modified for GPUGems from the original gridlib file

  This class encapsulates numerical stuff, that should be available
  in libm.so on most platforms. However, on some platforms there are
  subtle differences (signatures !) between the C++-library calls that 
  wrap the C-library functions. So, in case of doubt, implement it here.

  The fast math functions provided, use an approximating polynomial.
  Note that for performance reasons, there is no range checking on 
  the arguments.

  The approximation formulas are from:
  "Handbook of Mathematical Functions",
  Edited by M. Abramowitz and I.A. Stegun.
  Dover Publications, Inc.
  New York, NY
  Ninth printing, December 1972

  For the inverse tangent calls, all approximations are valid for \f$ |t| \le 1 \f$ .
  To compute \f$ \tan^{-1}(t) \f$ for \f$ t > 1 \f$, use \f$ \tan^{-1}(t) = \pi / 2 - \tan^{-1}(1/t) \f$ .
  For \f$ t < -1 \f$ , use \f$ \tan^{-1}(t) = - \pi / 2 - \tan^{-1}(1/t) \f$ .

  Speedups were measured on a Pentium II 400 Mhz with a release build of the
  code. Comparisons are to the calls \p sin, \p cos, \p tan, and \p atan in 
  the standard math library.
*/
template <class T>
class GRIDLIB_API GbMath
{
public:
  /** @name Predefined values */
  //@{

  //! Pseudo-zero tolerance - all numbers below this are regarded zero
  static const T ZERO_TOLERANCE;
  //! Predefined value of \f$ \pi \f$
  static const T PI;
  //! Predefined value of \f$ \pi / 2 \f$
  static const T PIHALF;
  //! Predefined value of \f$ 2 * \pi \f$
  static const T TWOPI;
  //! Predefined value of \f$ \frac{1}{\pi} \f$
  static const T INV_PI;
  //! Predefined value of \f$ \frac{1}{2 * \pi} \f$
  static const T INV_TWOPI;
  //! Convert degrees to radians
  static const T DEG2RAD;
  //! Convert radians to degrees
  static const T RAD2DEG;

  //@}

  /** @name Math calls */
  //@{

  //! Get the smaller of both values
  INLINE static T Min(T a, T b);
  
  //! Get the larger of both values
  INLINE static T Max(T a, T b);

  //! Get the absolut value of a number
  INLINE static T Abs(T v);

  //! Wrapper for \p exp
  INLINE static T Exp(T v);

  //! Wrapper for \p fmod
  INLINE static T Mod(T a, T b);

  //! Wrapper for \p floor
  INLINE static T Floor(T v);
  INLINE static T FastFloor(T v);
  //! Wrapper for \p ceil
  INLINE static T Ceil(T v);

  //! Wrapper for \p round
  INLINE static T Round(T v);

  //! Get the sign of the integer argument
  INLINE static int Sign (int i);
  //! Get the sign of the floating point argument
  INLINE static T Sign(T v);

  //! Get the square root of the argument
  INLINE static T Sqrt(T v);
  INLINE static T FastSqrt(T v);
  //! Get the inverse square root of the argument
  INLINE static T InvSqrt(T v);

  //! Get the square of the argument
  INLINE static T Sqr(T v);
  //! Get the logarithm
  INLINE static T Log(T v);
  //! Get a power b where b can be floating point
  INLINE static T Pow(T a, T b);

  //! Wrapper for \p sin
  INLINE static T Sin(T v);
  //! Wrapper for \p cos
  INLINE static T Cos(T v);
  //! Wrapper for \p tan
  INLINE static T Tan(T v);

  //! Wrapper for \p asin, clamped to [-1,1]
  INLINE static T ASin(T v);
  //! Wrapper for \p acos, clamped to [-1,1]
  INLINE static T ACos(T v);
  //! Wrapper for \p atan
  INLINE static T ATan(T v);
  //! Wrapper for \p atan2
  INLINE static T ATan2(T v1, T v2);

  //@}

  /** @name Random numbers */
  //@{

  //! Generate a random number in [0,1)
  INLINE static T UnitRandom(T seed = 0.0);

  //! Generate a random number in [-1,1)
  INLINE static T SymmetricRandom(T seed = 0.0);

  //! Generate a random number in [min,max)
  INLINE static T IntervalRandom (T fMin, T fMax, T seed = 0.0);

  //@}

  /** @name Accelerated routines */
  //@{

  //! Fast \p sin
  INLINE static T FastSin0 (T fT);
  //! Fast \p sin
  INLINE static T FastSin1 (T fT);
  //! Fast \p sin
  INLINE static T FastSin2 (T fT);
  //! Fast \p cos
  INLINE static T FastCos0 (T fT);
  //! Fast \p cos
  INLINE static T FastCos1 (T fT);
  //! Fast \p cos
  INLINE static T FastCos2 (T fT);
  //! Fast \p tan
  INLINE static T FastTan0 (T fT);
  //! Fast \p tan
  INLINE static T FastTan1 (T fT);
  //! Fast \p asin
  INLINE static T FastInvSin0 (T fT);
  //! Fast \p acos
  INLINE static T FastInvCos0 (T fT);
  //! Fast \p atan
  INLINE static T FastInvTan0 (T fT);
  //! Fast \p atan
  INLINE static T FastInvTan1 (T fT);
  //! Fast \p atan
  INLINE static T FastInvTan2 (T fT);

  //! Fast inverse square root
  INLINE static T FastInvSqrt(T v);

  //@}

  /** @name Gamma and related functions */
  //@{

    INLINE static T LogGamma (T fX);
    INLINE static T Gamma (T fX);
    INLINE static T IncompleteGamma (T fA, T fX);

  //@}

  /** @name Error functions */
  //@{

    //! Polynomial approximation
    INLINE static T Erf (T fX);
    //! erfc = 1-erf
    INLINE static T Erfc (T fX);

  //@}

  /** @name Modified Bessel functions of order 0 and 1 */
  //@{
    
    INLINE static T ModBessel0 (T fX);
    INLINE static T ModBessel1 (T fX);

  //@}

  /** @name Other numeric routines */
  //@{

    // Get the Gaussian density at point \a x
    INLINE static T GaussianDensity(T x, T variance, T mean);

    // Get the Laplacian density at point \a x
    INLINE static T LaplacianDensity(T x, T variance, T mean);

  //@}

protected:
    //! series form (used when fX < 1+fA)
    static T IncompleteGammaS (T fA, T fX);

    //! continued fraction form (used when fX >= 1+fA)
    static T IncompleteGammaCF (T fA, T fX);

    static const float sin_tab[];
};

#ifndef OUTLINE

#include "GbMath.in"
#include "GbMath.T"

#else

INSTANTIATE( GbMath<float> );
INSTANTIATE( GbMath<double> );

#endif 

#endif  // GBMATH_HH
