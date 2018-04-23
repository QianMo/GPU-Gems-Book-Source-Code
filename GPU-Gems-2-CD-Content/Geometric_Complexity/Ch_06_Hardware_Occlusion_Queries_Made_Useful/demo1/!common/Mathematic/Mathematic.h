//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#ifndef MathematicH
#define MathematicH

#include <math.h>
#include <limits>
#include <Base/ShortTypes.h>
#include <Base/BaseException.h>
//---------------------------------------------------------------------------
namespace Math {

struct MathException : public BaseException {
	MathException(const std::string& vMsg): BaseException(vMsg) { };
};

template<class T> inline void maximum(T& vMaximum, const T& a, const T& b) { vMaximum = (a < b ? b : a); }
template<class T> inline T maximum(const T& a, const T& b) { return (a < b ? b : a); }
template<class T> inline void minimum(T& vMinimum, const T& a, const T& b) { vMinimum = (a < b ? a : b); }
template<class T> inline T minimum(const T& a, const T& b) { return (a < b ? a : b); }
template<class T> inline T mean(const T& a, const T& b) { return (a+b)/2; }
//signum of double
template<class T> inline int sign(const T& a) { return (a > 0) ? 1 : ((a < 0) ? -1 : 0); }
//absolut value of double
template<class REAL> inline REAL abs(const REAL& a) { return REAL( (a < 0)? (-a):(a) ); }
template<> inline float abs<float>(const float& a) { return ::fabs(a); }

template<class REAL> inline REAL sqrt(const REAL& r) { return ::sqrt(r); }
template<class REAL> inline REAL floor(const REAL& r) { return ::floor(r); }
template<class REAL> inline REAL ceil(const REAL& r) {	return ::ceil(r); }
template<class REAL> inline REAL fract(const REAL& r) { return r-floor(r); }
template<class REAL> inline REAL mod(const REAL& a, const REAL& b) { return REAL(fmod(a,b)); }

template<class REAL>
struct Const {
    typedef REAL Real;
    typedef const REAL cReal;

	static inline const REAL pi() throw() { return REAL(3.141592653589793238462643383279502884197169399375105820974944592); }
	static inline const REAL pi_2() throw() { return pi()/REAL(2.0); }
	static inline const REAL pi_180() throw() { return pi()/REAL(180.0); }
	static inline const REAL c180_pi() throw() { return REAL(180.0)/pi(); }
	static inline const REAL infinity() throw() { return std::numeric_limits<REAL>::infinity(); }
	static inline const REAL epsilon() throw() { return std::numeric_limits<REAL>::epsilon(); }
	static inline const REAL near_epsilon() throw() { return REAL(10e-5); }
	static inline const REAL zero() throw() { return REAL(0.0); }
};

template<class REAL> inline REAL sin(const REAL& vIn) {	return REAL(::sin(vIn)); }
template<class REAL> inline REAL tan(const REAL& vIn) {	return REAL(::tan(vIn)); }
template<class REAL> inline REAL aCos(const REAL& vIn) { return REAL(::acos(vIn)); }
template<class REAL> inline REAL aSin(const REAL& vIn) { return REAL(::asin(vIn)); }
template<class REAL> inline REAL aTan(const REAL& vIn) { return REAL(::atan(vIn)); }
template<class REAL> inline REAL cos(const REAL& vIn) {	return REAL(::cos(vIn)); }
template<class REAL> inline REAL coTan(const REAL& r) { return -tan(r+Const<REAL>::pi_2()); }
template<class REAL> inline REAL rad2Deg(const REAL& r) { return REAL(r*Const<REAL>::c180_pi()); }
template<class REAL> inline REAL deg2Rad(const REAL& r) { return REAL(r*Const<REAL>::pi_180()); }
template<class REAL> inline REAL deg2_n180_180(const REAL&);


template<class REAL> 
inline REAL lerp(const REAL& a, const REAL& b, const REAL& fact) { return a+(b-a)*fact; }

template<class REAL> inline void interpolate(const REAL& dest, const REAL& step, REAL& value);

// return a random number form the interval [start .. stop]
template<class T> T randRange(const T& from, const T& to);

template<class REAL> REAL relativeEpsilon(const REAL&, const REAL& epsilon = Const<REAL>::near_epsilon());
template<class REAL> bool alike(const REAL&, const REAL&, const REAL& epsilon = Const<REAL>::near_epsilon());
/*	| a1 a2 |
	| b1 b2 | calculate the determinent of a 2x2 matrix in the from*/
template<class REAL> inline REAL det2x2(const REAL& a1, const REAL& a2, 
								    const REAL& b1, const REAL& b2);
/*	| a1 a2 a3 |
	| b1 b2 b3 |
	| c1 c2 c3 | calculate the determinent of a 3x3 matrix*/
template<class REAL> inline REAL det3x3(const REAL& a1, const REAL& a2, const REAL& a3,
									const REAL& b1, const REAL& b2, const REAL& b3,
									const REAL& c1, const REAL& c2, const REAL& c3);

void randomize();

namespace Geometry {
	struct GeometryException : public Math::MathException {
		GeometryException(const std::string& vMsg): MathException(vMsg) { };
	};
}

//namespace Math
}

#include "Mathematic.inl"

#endif
