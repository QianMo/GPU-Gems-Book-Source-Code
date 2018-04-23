#include <math.h>
#ifdef _WIN32
#include <float.h>
#endif
#ifndef BRTINTRINSIC_HPP
#define BRTINTRINSIC_HPP

#define UNINTRINSICMEMBER(FUNC,CALLFUNC,RET) \
template <typename T> vec<typename T::TYPE, 1> __##FUNC##_cpu_inner(const T &f) {return f.CALLFUNC();}
/*
inline RET __##FUNC##_cpu_inner (const __BrtFloat4 &f) { \
  return f.CALLFUNC(); \
} \
inline RET __##FUNC##_cpu_inner (const __BrtFloat3 &f) { \
  return f.CALLFUNC(); \
} \
inline RET __##FUNC##_cpu_inner (const __BrtFloat2 &f) { \
  return f.CALLFUNC(); \
} \
inline RET __##FUNC##_cpu_inner (const __BrtFloat1 &f) { \
  return f.CALLFUNC(); \
}
*/

#define UNINTRINSIC(FUNC,CALLFUNC) \
template <typename T> T __##FUNC##_cpu_inner (const T&f ) { \
        return T(CALLFUNC(f.getAt(0)), \
                 CALLFUNC(f.getAt(1)), \
                 CALLFUNC(f.getAt(2)), \
                 CALLFUNC(f.getAt(3)));         \
}
/*
inline __BrtFloat4 __##FUNC##_cpu_inner (const __BrtFloat4 &f) { \
  return __BrtFloat4 (CALLFUNC (f.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1)), \
                      CALLFUNC (f.unsafeGetAt(2)), \
                      CALLFUNC (f.unsafeGetAt(3))); \
} \
inline __BrtFloat3 __##FUNC##_cpu_inner (const __BrtFloat3 &f) { \
  return __BrtFloat3 (CALLFUNC (f.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1)), \
                      CALLFUNC (f.unsafeGetAt(2))); \
} \
inline __BrtFloat2 __##FUNC##_cpu_inner (const __BrtFloat2 &f) { \
  return __BrtFloat2 (CALLFUNC (f.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1))); \
} \
inline __BrtFloat1 __##FUNC##_cpu_inner (const __BrtFloat1 &f) { \
  return __BrtFloat1 (CALLFUNC (f.unsafeGetAt(0))); \
}
*/

#define UNINTRINSICINOUT(FUNC,CALLFUNC) \
inline __BrtDouble1 __##FUNC##_cpu_inner (const __BrtDouble1 &f, __BrtDouble1 &out) { \
  return __BrtDouble1 (CALLFUNC (f.unsafeGetAt(0),out.unsafeGetAt(0))); \
} \
inline __BrtDouble2 __##FUNC##_cpu_inner (const __BrtDouble2 &f, __BrtDouble2 &out) { \
    return __BrtDouble2 (CALLFUNC (f.unsafeGetAt(0),out.unsafeGetAt(0)), \
                         CALLFUNC(f.unsafeGetAt(1),out.unsafeGetAt(1))); \
} \
inline __BrtFloat4 __##FUNC##_cpu_inner (const __BrtFloat4 &f, __BrtFloat4 &out) { \
  return __BrtFloat4 (CALLFUNC (f.unsafeGetAt(0),out.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1),out.unsafeGetAt(1)), \
                      CALLFUNC (f.unsafeGetAt(2),out.unsafeGetAt(2)), \
                      CALLFUNC (f.unsafeGetAt(3),out.unsafeGetAt(3))); \
} \
inline __BrtFloat3 __##FUNC##_cpu_inner (const __BrtFloat3 &f, __BrtFloat3 &out) { \
  return __BrtFloat3 (CALLFUNC (f.unsafeGetAt(0),out.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1),out.unsafeGetAt(1)), \
                      CALLFUNC (f.unsafeGetAt(2),out.unsafeGetAt(2))); \
} \
inline __BrtFloat2 __##FUNC##_cpu_inner (const __BrtFloat2 &f, __BrtFloat2 &out) { \
  return __BrtFloat2 (CALLFUNC (f.unsafeGetAt(0),out.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1),out.unsafeGetAt(1))); \
} \
inline __BrtFloat1 __##FUNC##_cpu_inner (const __BrtFloat1 &f, __BrtFloat1 & out) { \
  return __BrtFloat1 (CALLFUNC (f.unsafeGetAt(0),out.unsafeGetAt(0))); \
}

#if defined (_MSC_VER) && (_MSC_VER <= 1200)
#define TEMPL_TYPESIZE sizeof(BRT_TYPE)/sizeof(BRT_TYPE::TYPE)
#else
#define TEMPL_TYPESIZE BRT_TYPE::size
#endif

#define BININTRINSIC(FUNC,CALLFUNC) \
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,float>::type, \
       LUB<TEMPL_TYPESIZE,4>::size> __##FUNC##_cpu_inner (const __BrtFloat4 &f, const BRT_TYPE &g) { \
  return f.CALLFUNC(g); \
} \
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,float>::type, \
       LUB<TEMPL_TYPESIZE,3>::size> __##FUNC##_cpu_inner (const __BrtFloat3 &f, const BRT_TYPE &g) { \
  return f.CALLFUNC(g); \
} \
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,float>::type, \
       LUB<TEMPL_TYPESIZE,2>::size> __##FUNC##_cpu_inner (const __BrtFloat2 &f, const BRT_TYPE &g) { \
  return f.CALLFUNC(g); \
} \
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,float>::type, \
       LUB<TEMPL_TYPESIZE,1>::size> __##FUNC##_cpu_inner (const __BrtFloat1 &f, const BRT_TYPE &g) { \
  return f.CALLFUNC(g); \
} \
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,double>::type, \
       LUB<TEMPL_TYPESIZE,2>::size> __##FUNC##_cpu_inner (const __BrtDouble2 &f, const BRT_TYPE &g) { \
  return f.CALLFUNC(g); \
} \
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,double>::type, \
       LUB<TEMPL_TYPESIZE,1>::size> __##FUNC##_cpu_inner (const __BrtDouble1 &f, const BRT_TYPE &g) { \
  return f.CALLFUNC(g); \
}

#define TRIINTRINSIC(FUNC,CALLFUNC) \
template <class ThirdEye> __BrtFloat4 __##FUNC##_cpu_inner (const __BrtFloat4 &f, \
             const __BrtFloat4 &g, \
             const ThirdEye &h) { \
  return __BrtFloat4 (CALLFUNC (f.unsafeGetAt(0), \
                                       g.unsafeGetAt(0), \
                                       h.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1), \
                                       g.unsafeGetAt(1), \
                                       h.getAt(1)), \
                      CALLFUNC (f.unsafeGetAt(2), \
                                       g.unsafeGetAt(2), \
                                       h.getAt(2)), \
                      CALLFUNC (f.unsafeGetAt(3), \
                                       g.unsafeGetAt(3), \
                                       h.getAt(3))); \
} \
template <class ThirdEye> __BrtFloat3 __##FUNC##_cpu_inner (const __BrtFloat3 &f, \
                         const __BrtFloat3 &g, \
                         const ThirdEye &h) { \
  return __BrtFloat3 (CALLFUNC (f.unsafeGetAt(0), \
                                       g.unsafeGetAt(0), \
                                       h.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1), \
                                       g.unsafeGetAt(1), \
                                       h.getAt(1)), \
                      CALLFUNC (f.unsafeGetAt(2), \
                                       g.unsafeGetAt(2), \
                                       h.getAt(2))); \
} \
template <class ThirdEye> __BrtFloat2 __##FUNC##_cpu_inner (const __BrtFloat2 &f, \
                                                            const __BrtFloat2 &g, \
                                                            const ThirdEye &h) { \
  return __BrtFloat2 (CALLFUNC (f.unsafeGetAt(0), \
                                       g.unsafeGetAt(0), \
                                       h.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1), \
                                       g.unsafeGetAt(1), \
                                       h.getAt(1))); \
} \
template <class ThirdEye> __BrtFloat1 __##FUNC##_cpu_inner (const __BrtFloat1 &f, \
                                                            const __BrtFloat1 &g, \
                                                            const ThirdEye &h) { \
  return __BrtFloat1 (CALLFUNC (f.unsafeGetAt(0), \
                                g.unsafeGetAt(0),    \
                                h.unsafeGetAt(0)));        \
} \
template <class ThirdEye> __BrtDouble2 __##FUNC##_cpu_inner (const __BrtDouble2 &f, \
                                                            const __BrtDouble2 &g, \
                                                            const ThirdEye &h) { \
  return __BrtDouble2 (CALLFUNC (f.unsafeGetAt(0), \
                                       g.unsafeGetAt(0), \
                                       h.unsafeGetAt(0)), \
                      CALLFUNC (f.unsafeGetAt(1), \
                                       g.unsafeGetAt(1), \
                                       h.getAt(1))); \
} \
template <class ThirdEye> __BrtDouble1 __##FUNC##_cpu_inner (const __BrtDouble1 &f, \
                                                            const __BrtDouble1 &g, \
                                                            const ThirdEye &h) { \
  return __BrtDouble1 (CALLFUNC (f.unsafeGetAt(0), \
                                g.unsafeGetAt(0),    \
                                h.unsafeGetAt(0)));        \
}

template <class T> T __normalize_cpu_inner (const T &x) {
   if (T::size==1) return x/x;
   if (T::size==2)
      return x/__sqrt_cpu_inner(x[0]*x[0]
                                + x[1]*x[1]);
   if (T::size==3)
      return x/__sqrt_cpu_inner(x[0]*x[0]
                                + x[1]*x[1]
                                + x[2]*x[2]);
   if (T::size==4)
      return x/__sqrt_cpu_inner(x[0]*x[0]
                                + x[1]*x[1]
                                + x[2]*x[2]
                                + x[3]*x[3]);
   typename T::TYPE size =x.unsafeGetAt(0)*x.unsafeGetAt(0);
   for (unsigned int i=1;i<T::size;++i) {
      size+=x.unsafeGetAt(i)*x.unsafeGetAt(i);
   }
   return x/__sqrt_cpu_inner(T(size));
}
template <typename T>  T degrees_float (T x) {
   return x*180.0f/3.1415926536f;
}
template <typename T>  float radians_float (T x) {
   return x*3.1415926536f/180.0f;
}
template <typename T>  T saturate_float (T x) {
   return x>1.0f?1.0f:x<0.0f?0.0f:x;
}

template <typename T>  T clamp_float(T x, T l, T u) {
   return x>u?u:x<l?l:x;
}
template <typename T>  T sign_float (T x) {
   return x==0.0f?0.0f:x<0.0f?-1.0f:1.0f;
}
template <typename T>  T exp2_float (T x) {
   return (T)pow(2.0f,x);
}
static const float _const_log2 = (float) log(2.0f);
template <typename T>  T log2_float (T x) {
   return (T)log (x)/_const_log2;
}
template <typename T>  T round_float (T x) {
   T f = x-(T)floor(x);
   T g = (T)ceil(x)-x;
   return f==g?(x<0.0f?(T)floor(x):(T)ceil(x)):f<g?(T)floor(x):(T)ceil(x);
}
template <typename T>  T lerp_float (T a, T b, T s) {
   return a + s*(b - a);
}
template <typename T>  T rsqrt_float (T x) {
    return (T)(1.0f/sqrt(x));
}
template <typename T>  T frac_float (T x) {
   T y = x-(T)floor(x);
   return x<0.0f?1.0f-y:y;
}
template <typename T>  T frc_float (T x) {
   return frac_float(x);
}
template <typename T>  T frexp_float (T x, T & oout) {
   int exp;
   x = (T)frexp(x,&exp);
   oout=(T)exp;
   return x;
}
template <typename T>  T modf_float (T x, T & oout) {
   double exp;
   x = (T)modf(x,&exp);
   oout=(T)exp;
   return x;
}
inline float finite_float (double x) {
#ifdef _WIN32
   return (float) _finite(x);
#else
#ifdef __APPLE__
   return (float) __isfinitef(x);
#else
   return (float) finite(x);
#endif
#endif
}
inline  float isnan_float (double x) {
#ifdef _WIN32
   return (float) _isnan(x);
#else
#ifdef __APPLE__
   return (float) __isnanf(x);
#else
   return (float) isnan(x);
#endif
#endif
}
inline float isinf_float (double x) {
#ifdef _WIN32
   return (!finite_float(x))&&(!isnan_float(x));
#else
#ifdef __APPLE__
   return __isinff(x);
#else
   return isinf(x);
#endif
#endif
}
  BININTRINSIC(atan2,atan2)
  BININTRINSIC(fmod,fmod)
  BININTRINSIC(pow,pow)
  UNINTRINSICMEMBER(any,any,__BrtFloat1);
  UNINTRINSICMEMBER(all,all,__BrtFloat1);
  UNINTRINSICMEMBER(length,length,__BrtFloat1);
  UNINTRINSICMEMBER(len,length,__BrtFloat1);
  UNINTRINSICINOUT(frexp,frexp_float);
  UNINTRINSIC(degrees,degrees_float)
  UNINTRINSIC(radians,radians_float)
  UNINTRINSIC(saturate,saturate_float)
  UNINTRINSIC(abs,fabsf)
  TRIINTRINSIC(clamp,clamp_float)
  UNINTRINSIC(isfinite,finite_float)
  UNINTRINSIC(isnan,isnan_float)
  BININTRINSIC(max,max_float)
  BININTRINSIC(min,min_float)
  UNINTRINSIC(sign,sign_float)
  UNINTRINSIC(acos,acos)
  UNINTRINSIC(asin,asin)
  UNINTRINSIC(atan,atan)
  UNINTRINSIC(ceil,ceil)
  UNINTRINSIC(cos,cos)
  UNINTRINSIC(cosh,cosh)
  UNINTRINSIC(exp,exp)
  UNINTRINSIC(exp2,exp2_float)
  UNINTRINSIC(floor,floor)
  UNINTRINSIC(frac,frac_float)
  UNINTRINSIC(frc,frc_float)
  UNINTRINSIC(isinf,isinf_float)
  BININTRINSIC(ldexp,ldexp_float)
  BININTRINSIC(distance,distance)
  UNINTRINSIC(log,log)
  UNINTRINSIC(log2,log2_float)
  UNINTRINSIC(log10,log10)
  UNINTRINSICINOUT(modf,modf_float)
  UNINTRINSIC(round,round_float)
  UNINTRINSIC(rsqrt,rsqrt_float)
  UNINTRINSIC(sin,sin)  
  //UNINTRINSIC(sincos)
  UNINTRINSIC(sinh,sinh)
  UNINTRINSIC(sqrt,sqrt)
  BININTRINSIC(step,step_float)
  UNINTRINSIC(tan,tan)
  UNINTRINSIC(tanh,tanh)
     //  TRIINTRINSIC(smoothstep)
  TRIINTRINSIC(lerp,lerp_float)

#undef UNINTRINSIC
#undef BININTRINSIC
#undef TRIINTRINSIC
inline __BrtFloat3 __cross_cpu_inner (const __BrtFloat3 &u, const __BrtFloat3 v) {
     return __BrtFloat3( u.unsafeGetAt(1)*v.unsafeGetAt(2)
                         -u.unsafeGetAt(2)*v.unsafeGetAt(1),
                         u.unsafeGetAt(2)*v.unsafeGetAt(0)
                         -u.unsafeGetAt(0)*v.unsafeGetAt(2),
                         
                         u.unsafeGetAt(0)*v.unsafeGetAt(1)
                         -u.unsafeGetAt(1)*v.unsafeGetAt(0));
}

template <class T, class BRT_TYPE> 
vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,GCCTYPENAME T::TYPE>::type, 1>
       __dot_cpu_inner (const T &f, const BRT_TYPE &g) { \
  return f.dot(g); \
}
/*
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,float>::type, 1>
       __dot_cpu_inner (const __BrtFloat4 &f, const BRT_TYPE &g) { \
  return f.dot(g); \
}
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,float>::type, 1>
       __dot_cpu_inner (const __BrtFloat3 &f, const BRT_TYPE &g) { \
  return f.dot(g); \
}
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,float>::type, 1>
       __dot_cpu_inner (const __BrtFloat2 &f, const BRT_TYPE &g) { \
  return f.dot(g); \
}
template <class BRT_TYPE> vec<GCCTYPENAME LCM<GCCTYPENAME BRT_TYPE::TYPE,float>::type, 1>
       __dot_cpu_inner (const __BrtFloat1 &f, const BRT_TYPE &g) { \
  return f.dot(g); \
}
*/
#endif
