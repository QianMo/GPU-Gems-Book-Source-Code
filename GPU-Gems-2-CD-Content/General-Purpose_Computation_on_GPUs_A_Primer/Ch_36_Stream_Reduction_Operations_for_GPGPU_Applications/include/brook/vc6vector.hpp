#line 1 "brtvector.hpp"
#ifndef VC6VECTOR_HPP
#define VC6VECTOR_HPP
#if defined (_MSC_VER) && _MSC_VER <=1200 && !defined(VC6VECTOR_HPP)
#include "vc6vector.hpp"
//the above headerfile has the template functions automatically expanded.
//not needed for recent compilers.
#else

#include <iostream>
#include <math.h>
#include "type_promotion.hpp"
template <class VALUE, unsigned int tsize> class vec;

template <class T, class B> inline T singlequestioncolon (const B& a,
                                                          const T&b,
                                                          const T&c){
    return a.questioncolon(b,c);
};
template <> inline float singlequestioncolon (const char & a,
                                              const float &b,
                                              const float&c) {
    return a?b:c;
}
template <> inline float singlequestioncolon (const float & a,
                                              const float &b,
                                              const float&c) {
    return a?b:c;
}

template <> inline vec<float,1> singlequestioncolon (const vec<float,1> &a,
                                                     const vec<float,1> &b,
                                                     const vec<float,1> &c);
template <> inline vec<float,2> singlequestioncolon (const vec<float,2> & a,
                                                     const vec<float,2> &b,
                                                     const vec<float,2> &c);
template <> inline vec<float,3> singlequestioncolon (const vec<float,3> &a,
                                                     const vec<float,3> &b,
                                                     const vec<float,3> &c);
template <> inline vec<float,4> singlequestioncolon (const vec<float,4> &a,
                                                     const vec<float,4> &b,
                                                     const vec<float,4> &c);

inline float step_float (float a, float x){
   return (float)((x>=a)?1.0f:0.0f);
}
inline float max_float(float x, float y) {
   return (float)(x>y?x:y);
}
inline float min_float(float x, float y) {
   return (float)(x<y?x:y);
}
inline float ldexp_float(float x, float y) {
   return (float)ldexp(x,(int)y);
}
template <class T> class GetValueOf {public:
    typedef typename T::TYPE type;
};
template <> class GetValueOf <float> {public:
    typedef float type;
};
template <> class GetValueOf <double> {public:
    typedef double type;
};
template <> class GetValueOf <int> {public:
    typedef int type;
};
template <> class GetValueOf <unsigned int> {public:
    typedef unsigned int type;
};
template <> class GetValueOf <char> {public:
    typedef char type;
};
template <> class GetValueOf <bool> {public:
    typedef bool type;
};
#if defined (_MSC_VER)
template <class T> class Holder {
public:
    static typename GetValueOf<T>::type getAt (const T&t, int i) {
        return t.getAt(i);
    }
};
#define HOLDER(TYP) template <> class Holder<TYP> { \
public: \
    static TYP getAt(TYP t, int i) { \
        return t; \
    } \
}
HOLDER(float);
HOLDER(double);
HOLDER(char);
HOLDER(unsigned int);
HOLDER(int);
HOLDER(bool);
template <class T> typename GetValueOf<T>::type GetAt (const T& in,int i) {
    return Holder<T>::getAt(in,i);
}
#else
template <class T> static typename GetValueOf<T>::type GetAt (const T& in,int i) {
    return in.getAt(i);
}
#define SPECIALGETAT(TYP) template <> static TYP GetAt (const TYP& in,int i) {return in;}

SPECIALGETAT(int)
SPECIALGETAT(unsigned int)
SPECIALGETAT(char)
SPECIALGETAT(float)
SPECIALGETAT(double)
SPECIALGETAT(bool)

#endif
template <class T> class BracketType {public:
  typedef T type;
};
template <> class BracketType <float> {public:
  typedef vec<float,1> type;
};
template <> class BracketType <int> {public:
  typedef vec<int,1> type;
};
template <> class BracketType <char> {public:
  typedef vec<char,1> type;
};
template <class T> class BracketOp {public:
  template <class U> T& operator ()(const U&u, unsigned int i) {
    return u.unsafeGetAt(i);
  }
  template <class U> T& operator () (U&u, unsigned int i) {
    return u.unsafeGetAt(i);
  }
};
template <> class BracketOp <float> {public:
  template <class U> U operator ()(const U&u, unsigned int i) {
     return vec<float,1>(u.unsafeGetAt(i));
  }
};
template <> class BracketOp <int> {public:
  template <class U> U operator ()(const U&u, unsigned int i) {
     return vec<int,1>(u.unsafeGetAt(i));  
  }
};
template <> class BracketOp <char> {public:
  template <class U> U operator ()(const U&u, unsigned int i) {
     return vec<char,1>(u.unsafeGetAt(i));
  }
};

enum MASKS {
  maskX=0,
  maskY=1,
  maskZ=2,
  maskW=3
};
template <class T> class InitializeClass {public:
    template <class V> T operator () (const V&a, const V&b, const V&c,
const V&d) {
        return T(a,b,c,d);
    }
};
#define INITBASECLASS(MYTYPE) template <> class InitializeClass<MYTYPE> { \
 public: \
    template <class V> MYTYPE operator () (const V&a, \
					  const V&b,  \
					  const V&c,  \
					  const V&d) { \
      return (MYTYPE)a; \
    } \
}
INITBASECLASS(float);
INITBASECLASS(double);
INITBASECLASS(int);
INITBASECLASS(unsigned int);
INITBASECLASS(char);
INITBASECLASS(unsigned char);


template <> class InitializeClass<bool> { 
 public: 
    template <class V> bool operator () (const V&a, 
					  const V&b,  
					  const V&c,  
					  const V&d) { 
      return (a||b||c||d)?true:false; 
    } 
};

#ifdef _MSC_VER
#if _MSC_VER <= 1200
#define GCCTYPENAME
#define INTERNALTYPENAME
#else
#define GCCTYPENAME typename
#define INTERNALTYPENAME typename
#endif
#else
#define GCCTYPENAME typename
#define INTERNALTYPENAME typename
#endif

template <class VALUE, unsigned int tsize> class vec{
public:
    typedef VALUE TYPE;
    enum SIZ{size=tsize};
    typedef VALUE array_type[size];
protected:
    VALUE f[size];
public:
    const VALUE &getAt (unsigned int i) const{
       return i<size?f[i]:f[size-1];
    }
    VALUE &getAt (unsigned int i) {
       return i<size?f[i]:f[size-1];
    }
    const VALUE &unsafeGetAt (unsigned int i) const{return f[i];}
    VALUE &unsafeGetAt (unsigned int i) {return f[i];}
    typename BracketType<VALUE>::type operator [] (int i)const {return BracketOp<VALUE>()(*this,i);}
    vec<VALUE,tsize>& gather() {
        return *this;
    }
    const vec<VALUE,tsize>& gather() const{
        return *this;
    }
    template<class BRT_TYPE> BRT_TYPE castTo() {
        return InitializeClass<BRT_TYPE>()(getAt(0),
					   getAt(1),
					   getAt(2),
					   getAt(3));
    }
   vec<VALUE,1> any() const{
      return vec<VALUE,1>(getAt(0)!=0.0f||getAt(1)!=0.0f||getAt(2)!=0.0f||getAt(3)!=0.0f);
   }
   vec<VALUE,1> all() const {
      return vec<VALUE,1>(getAt(0)!=0.0f&&getAt(1)!=0.0f&&getAt(2)!=0.0f&&getAt(3)!=0.0f);
   }
   vec<VALUE,1> length() const {
      unsigned int i;
      VALUE tot = unsafeGetAt(0);
      tot*=tot;
      for (i=1;i<tsize;++i) {
         tot+=unsafeGetAt(i)*unsafeGetAt(i);
      }
      return vec<VALUE,1>((VALUE)sqrt(tot));
   }
#define BROOK_UNARY_OP(op) vec<VALUE,tsize> operator op ()const { \
      return vec<VALUE, tsize > (op getAt(0),  \
                                 op getAt(1),  \
                                 op getAt(2),  \
                                 op getAt(3)); \
    }
    BROOK_UNARY_OP(+)
    BROOK_UNARY_OP(-)
    BROOK_UNARY_OP(!)
#undef BROOK_UNARY_OP
#define NONCONST_BROOK_UNARY_OP(op) vec<VALUE,tsize> operator op (){ \
      return vec<VALUE, tsize > (op getAt(0),  \
                                 tsize>1?op getAt(1):getAt(1),  \
                                 tsize>2?op getAt(2):getAt(2),  \
                                 tsize>3?op getAt(3):getAt(3)); \
    }
    NONCONST_BROOK_UNARY_OP(--);
    NONCONST_BROOK_UNARY_OP(++);
#undef BROOK_UNARY_OP
    vec<VALUE,4> swizzle4(int x,int y,int z,int w)const {
        return vec<VALUE,4>(unsafeGetAt(x),
                            unsafeGetAt(y),
                            unsafeGetAt(z),
                            unsafeGetAt(w));
    }
    vec<VALUE,3> swizzle3(int x,int y,int z)const {
        return vec<VALUE,3>(unsafeGetAt(x),unsafeGetAt(y),unsafeGetAt(z));
    }
    vec<VALUE,2> swizzle2(int x,int y)const {
        return vec<VALUE,2>(unsafeGetAt(x),unsafeGetAt(y));
    }
    vec<VALUE, 1> swizzle1(int x)const {
        return vec<VALUE,1>(unsafeGetAt(x));
    }
    vec() {}
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const int &inx, 
	   const int &iny, 
	   const int &inz, 
	   const int& inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const int& inx, 
				   const int& iny, 
				   const int& inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const int& inx, const int& iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const int& scalar) {
        (*this)=scalar;
    }
     operator int () const{
      return InitializeClass<int>()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const int & in) {  \
        f[0] op (VALUE)(GetAt<int>(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<int>(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<int>(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<int>(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const char &inx, 
	   const char &iny, 
	   const char &inz, 
	   const char& inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const char& inx, 
				   const char& iny, 
				   const char& inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const char& inx, const char& iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const char& scalar) {
        (*this)=scalar;
    }
     operator char () const{
      return InitializeClass<char>()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const char & in) {  \
        f[0] op (VALUE)(GetAt<char>(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<char>(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<char>(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<char>(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const float &inx, 
	   const float &iny, 
	   const float &inz, 
	   const float& inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const float& inx, 
				   const float& iny, 
				   const float& inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const float& inx, const float& iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const float& scalar) {
        (*this)=scalar;
    }
     operator float () const{
      return InitializeClass<float>()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const float & in) {  \
        f[0] op (VALUE)(GetAt<float>(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<float>(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<float>(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<float>(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const double &inx, 
	   const double &iny, 
	   const double &inz, 
	   const double& inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const double& inx, 
				   const double& iny, 
				   const double& inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const double& inx, const double& iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const double& scalar) {
        (*this)=scalar;
    }
     operator double () const{
      return InitializeClass<double>()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const double & in) {  \
        f[0] op (VALUE)(GetAt<double>(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<double>(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<double>(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<double>(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const unsigned int &inx, 
	   const unsigned int &iny, 
	   const unsigned int &inz, 
	   const unsigned int& inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const unsigned int& inx, 
				   const unsigned int& iny, 
				   const unsigned int& inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const unsigned int& inx, const unsigned int& iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const unsigned int& scalar) {
        (*this)=scalar;
    }
     operator unsigned int () const{
      return InitializeClass<unsigned int>()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const unsigned int & in) {  \
        f[0] op (VALUE)(GetAt<unsigned int>(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<unsigned int>(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<unsigned int>(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<unsigned int>(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const bool &inx, 
	   const bool &iny, 
	   const bool &inz, 
	   const bool& inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const bool& inx, 
				   const bool& iny, 
				   const bool& inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const bool& inx, const bool& iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const bool& scalar) {
        (*this)=scalar;
    }
     operator bool () const{
      return InitializeClass<bool>()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const bool & in) {  \
        f[0] op (VALUE)(GetAt<bool>(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<bool>(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<bool>(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<bool>(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<float,1>  &inx, 
	   const vec<float,1>  &iny, 
	   const vec<float,1>  &inz, 
	   const vec<float,1> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<float,1> & inx, 
				   const vec<float,1> & iny, 
				   const vec<float,1> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<float,1> & inx, const vec<float,1> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<float,1> & scalar) {
        (*this)=scalar;
    }
     operator vec<float,1>  () const{
      return InitializeClass<vec<float,1> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<float,1>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<float,1> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<float,1> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<float,1> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<float,1> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<int,1>  &inx, 
	   const vec<int,1>  &iny, 
	   const vec<int,1>  &inz, 
	   const vec<int,1> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<int,1> & inx, 
				   const vec<int,1> & iny, 
				   const vec<int,1> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<int,1> & inx, const vec<int,1> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<int,1> & scalar) {
        (*this)=scalar;
    }
     operator vec<int,1>  () const{
      return InitializeClass<vec<int,1> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<int,1>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<int,1> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<int,1> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<int,1> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<int,1> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<char,1>  &inx, 
	   const vec<char,1>  &iny, 
	   const vec<char,1>  &inz, 
	   const vec<char,1> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<char,1> & inx, 
				   const vec<char,1> & iny, 
				   const vec<char,1> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<char,1> & inx, const vec<char,1> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<char,1> & scalar) {
        (*this)=scalar;
    }
     operator vec<char,1>  () const{
      return InitializeClass<vec<char,1> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<char,1>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<char,1> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<char,1> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<char,1> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<char,1> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<float,2>  &inx, 
	   const vec<float,2>  &iny, 
	   const vec<float,2>  &inz, 
	   const vec<float,2> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<float,2> & inx, 
				   const vec<float,2> & iny, 
				   const vec<float,2> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<float,2> & inx, const vec<float,2> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<float,2> & scalar) {
        (*this)=scalar;
    }
     operator vec<float,2>  () const{
      return InitializeClass<vec<float,2> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<float,2>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<float,2> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<float,2> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<float,2> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<float,2> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<int,2>  &inx, 
	   const vec<int,2>  &iny, 
	   const vec<int,2>  &inz, 
	   const vec<int,2> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<int,2> & inx, 
				   const vec<int,2> & iny, 
				   const vec<int,2> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<int,2> & inx, const vec<int,2> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<int,2> & scalar) {
        (*this)=scalar;
    }
     operator vec<int,2>  () const{
      return InitializeClass<vec<int,2> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<int,2>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<int,2> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<int,2> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<int,2> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<int,2> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<char,2>  &inx, 
	   const vec<char,2>  &iny, 
	   const vec<char,2>  &inz, 
	   const vec<char,2> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<char,2> & inx, 
				   const vec<char,2> & iny, 
				   const vec<char,2> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<char,2> & inx, const vec<char,2> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<char,2> & scalar) {
        (*this)=scalar;
    }
     operator vec<char,2>  () const{
      return InitializeClass<vec<char,2> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<char,2>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<char,2> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<char,2> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<char,2> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<char,2> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<float,3>  &inx, 
	   const vec<float,3>  &iny, 
	   const vec<float,3>  &inz, 
	   const vec<float,3> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<float,3> & inx, 
				   const vec<float,3> & iny, 
				   const vec<float,3> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<float,3> & inx, const vec<float,3> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<float,3> & scalar) {
        (*this)=scalar;
    }
     operator vec<float,3>  () const{
      return InitializeClass<vec<float,3> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<float,3>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<float,3> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<float,3> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<float,3> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<float,3> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<int,3>  &inx, 
	   const vec<int,3>  &iny, 
	   const vec<int,3>  &inz, 
	   const vec<int,3> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<int,3> & inx, 
				   const vec<int,3> & iny, 
				   const vec<int,3> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<int,3> & inx, const vec<int,3> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<int,3> & scalar) {
        (*this)=scalar;
    }
     operator vec<int,3>  () const{
      return InitializeClass<vec<int,3> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<int,3>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<int,3> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<int,3> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<int,3> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<int,3> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<char,3>  &inx, 
	   const vec<char,3>  &iny, 
	   const vec<char,3>  &inz, 
	   const vec<char,3> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<char,3> & inx, 
				   const vec<char,3> & iny, 
				   const vec<char,3> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<char,3> & inx, const vec<char,3> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<char,3> & scalar) {
        (*this)=scalar;
    }
     operator vec<char,3>  () const{
      return InitializeClass<vec<char,3> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<char,3>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<char,3> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<char,3> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<char,3> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<char,3> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<float,4>  &inx, 
	   const vec<float,4>  &iny, 
	   const vec<float,4>  &inz, 
	   const vec<float,4> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<float,4> & inx, 
				   const vec<float,4> & iny, 
				   const vec<float,4> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<float,4> & inx, const vec<float,4> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<float,4> & scalar) {
        (*this)=scalar;
    }
     operator vec<float,4>  () const{
      return InitializeClass<vec<float,4> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<float,4>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<float,4> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<float,4> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<float,4> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<float,4> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<int,4>  &inx, 
	   const vec<int,4>  &iny, 
	   const vec<int,4>  &inz, 
	   const vec<int,4> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<int,4> & inx, 
				   const vec<int,4> & iny, 
				   const vec<int,4> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<int,4> & inx, const vec<int,4> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<int,4> & scalar) {
        (*this)=scalar;
    }
     operator vec<int,4>  () const{
      return InitializeClass<vec<int,4> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<int,4>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<int,4> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<int,4> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<int,4> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<int,4> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 279 "brtvector.hpp"
#define GENERAL_TEMPLATIZED_FUNCTIONS
     
      vec (const vec<char,4>  &inx, 
	   const vec<char,4>  &iny, 
	   const vec<char,4>  &inz, 
	   const vec<char,4> & inw) {
        f[0]=inx;
        if (size>1) f[1]=iny;
        if (size>2) f[2]=inz;
        if (size>3) f[3]=inw;
    }
     vec (const vec<char,4> & inx, 
				   const vec<char,4> & iny, 
				   const vec<char,4> & inz) {
        f[0]=inx;
        if(size>1)f[1]=iny;
        if(size>2)f[2]=inz;
    }
     vec (const vec<char,4> & inx, const vec<char,4> & iny) {
        f[0]=inx;
        if (size>1) f[1]=iny;
    }
     vec (const vec<char,4> & scalar) {
        (*this)=scalar;
    }
     operator vec<char,4>  () const{
      return InitializeClass<vec<char,4> >()(getAt(0),getAt(1),getAt(2),getAt(3));
    }        
#define ASSIGN_OP(op)  \
         vec<VALUE,tsize>& operator op (const vec<char,4>  & in) {  \
        f[0] op (VALUE)(GetAt<vec<char,4> >(in,0));  \
        if (tsize>1) f[1] op (VALUE)(GetAt<vec<char,4> >(in,1));  \
        if (tsize>2) f[2] op (VALUE)(GetAt<vec<char,4> >(in,2));  \
        if (tsize>3) f[3] op (VALUE)(GetAt<vec<char,4> >(in,3));  \
        return *this;  \
    }
    ASSIGN_OP(=);
    ASSIGN_OP(/=);
    ASSIGN_OP(+=);
    ASSIGN_OP(-=);
    ASSIGN_OP(*=);
    ASSIGN_OP(%=);
#undef ASSIGN_OP
#undef GENERAL_TEMPLATIZED_FUNCTIONS
#line 322 "brtvector.hpp"

#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<float,1> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<float,1> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<float,1> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<float,1> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<float,tsize> questioncolon(const vec<float,1>  &b, 
						const vec<float,1>  &c)const {
        return vec<float,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<float,1> )/sizeof(float)
#else
#define TEMPL_TYPESIZE 1
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<float,1>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<float,1>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<float,1> >(b,0), \
                 getAt(1) op GetAt<vec<float,1> >(b,1), \
                 getAt(2) op GetAt<vec<float,1> >(b,2), \
                 getAt(3) op GetAt<vec<float,1> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<float,VALUE>::type,1> 
    dot (const vec<float,1>  &b) const{
      return vec< LCM<float,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<float,1> >(b,0) + 
                                     getAt(1) * GetAt<vec<float,1> >(b,1) + 
                                     getAt(2) * GetAt<vec<float,1> >(b,2) + 
                                     getAt(3) * GetAt<vec<float,1> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<float,1> >(b,0) + 
                                      getAt(1) * GetAt<vec<float,1> >(b,1) +
                                      getAt(2) * GetAt<vec<float,1> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<float,1> >(b,0) + 
                                      getAt(1) * GetAt<vec<float,1> >(b,1)):
                 getAt(0) * GetAt<vec<float,1> >(b,0));
                 
    }


    vec<typename LCM<float,VALUE>::type,1> 
    distance (const vec<float,1>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<float,1>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<float,1> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<float,1> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<float,1> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<float,1> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<int,1> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<int,1> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<int,1> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<int,1> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<int,tsize> questioncolon(const vec<int,1>  &b, 
						const vec<int,1>  &c)const {
        return vec<int,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<int,1> )/sizeof(int)
#else
#define TEMPL_TYPESIZE 1
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<int,1>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<int,1>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<int,1> >(b,0), \
                 getAt(1) op GetAt<vec<int,1> >(b,1), \
                 getAt(2) op GetAt<vec<int,1> >(b,2), \
                 getAt(3) op GetAt<vec<int,1> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<int,VALUE>::type,1> 
    dot (const vec<int,1>  &b) const{
      return vec< LCM<int,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<int,1> >(b,0) + 
                                     getAt(1) * GetAt<vec<int,1> >(b,1) + 
                                     getAt(2) * GetAt<vec<int,1> >(b,2) + 
                                     getAt(3) * GetAt<vec<int,1> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<int,1> >(b,0) + 
                                      getAt(1) * GetAt<vec<int,1> >(b,1) +
                                      getAt(2) * GetAt<vec<int,1> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<int,1> >(b,0) + 
                                      getAt(1) * GetAt<vec<int,1> >(b,1)):
                 getAt(0) * GetAt<vec<int,1> >(b,0));
                 
    }


    vec<typename LCM<int,VALUE>::type,1> 
    distance (const vec<int,1>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<int,1>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<int,1> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<int,1> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<int,1> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<int,1> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<char,1> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<char,1> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<char,1> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<char,1> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<char,tsize> questioncolon(const vec<char,1>  &b, 
						const vec<char,1>  &c)const {
        return vec<char,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<char,1> )/sizeof(char)
#else
#define TEMPL_TYPESIZE 1
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<char,1>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<char,1>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<char,1> >(b,0), \
                 getAt(1) op GetAt<vec<char,1> >(b,1), \
                 getAt(2) op GetAt<vec<char,1> >(b,2), \
                 getAt(3) op GetAt<vec<char,1> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<char,VALUE>::type,1> 
    dot (const vec<char,1>  &b) const{
      return vec< LCM<char,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<char,1> >(b,0) + 
                                     getAt(1) * GetAt<vec<char,1> >(b,1) + 
                                     getAt(2) * GetAt<vec<char,1> >(b,2) + 
                                     getAt(3) * GetAt<vec<char,1> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<char,1> >(b,0) + 
                                      getAt(1) * GetAt<vec<char,1> >(b,1) +
                                      getAt(2) * GetAt<vec<char,1> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<char,1> >(b,0) + 
                                      getAt(1) * GetAt<vec<char,1> >(b,1)):
                 getAt(0) * GetAt<vec<char,1> >(b,0));
                 
    }


    vec<typename LCM<char,VALUE>::type,1> 
    distance (const vec<char,1>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<char,1>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<char,1> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<char,1> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<char,1> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<char,1> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<float,2> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<float,2> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<float,2> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<float,2> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<float,tsize> questioncolon(const vec<float,2>  &b, 
						const vec<float,2>  &c)const {
        return vec<float,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<float,2> )/sizeof(float)
#else
#define TEMPL_TYPESIZE 2
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<float,2>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<float,2>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<float,2> >(b,0), \
                 getAt(1) op GetAt<vec<float,2> >(b,1), \
                 getAt(2) op GetAt<vec<float,2> >(b,2), \
                 getAt(3) op GetAt<vec<float,2> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<float,VALUE>::type,1> 
    dot (const vec<float,2>  &b) const{
      return vec< LCM<float,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<float,2> >(b,0) + 
                                     getAt(1) * GetAt<vec<float,2> >(b,1) + 
                                     getAt(2) * GetAt<vec<float,2> >(b,2) + 
                                     getAt(3) * GetAt<vec<float,2> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<float,2> >(b,0) + 
                                      getAt(1) * GetAt<vec<float,2> >(b,1) +
                                      getAt(2) * GetAt<vec<float,2> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<float,2> >(b,0) + 
                                      getAt(1) * GetAt<vec<float,2> >(b,1)):
                 getAt(0) * GetAt<vec<float,2> >(b,0));
                 
    }


    vec<typename LCM<float,VALUE>::type,1> 
    distance (const vec<float,2>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<float,2>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<float,2> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<float,2> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<float,2> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<float,2> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<int,2> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<int,2> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<int,2> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<int,2> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<int,tsize> questioncolon(const vec<int,2>  &b, 
						const vec<int,2>  &c)const {
        return vec<int,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<int,2> )/sizeof(int)
#else
#define TEMPL_TYPESIZE 2
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<int,2>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<int,2>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<int,2> >(b,0), \
                 getAt(1) op GetAt<vec<int,2> >(b,1), \
                 getAt(2) op GetAt<vec<int,2> >(b,2), \
                 getAt(3) op GetAt<vec<int,2> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<int,VALUE>::type,1> 
    dot (const vec<int,2>  &b) const{
      return vec< LCM<int,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<int,2> >(b,0) + 
                                     getAt(1) * GetAt<vec<int,2> >(b,1) + 
                                     getAt(2) * GetAt<vec<int,2> >(b,2) + 
                                     getAt(3) * GetAt<vec<int,2> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<int,2> >(b,0) + 
                                      getAt(1) * GetAt<vec<int,2> >(b,1) +
                                      getAt(2) * GetAt<vec<int,2> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<int,2> >(b,0) + 
                                      getAt(1) * GetAt<vec<int,2> >(b,1)):
                 getAt(0) * GetAt<vec<int,2> >(b,0));
                 
    }


    vec<typename LCM<int,VALUE>::type,1> 
    distance (const vec<int,2>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<int,2>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<int,2> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<int,2> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<int,2> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<int,2> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<char,2> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<char,2> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<char,2> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<char,2> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<char,tsize> questioncolon(const vec<char,2>  &b, 
						const vec<char,2>  &c)const {
        return vec<char,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<char,2> )/sizeof(char)
#else
#define TEMPL_TYPESIZE 2
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<char,2>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<char,2>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<char,2> >(b,0), \
                 getAt(1) op GetAt<vec<char,2> >(b,1), \
                 getAt(2) op GetAt<vec<char,2> >(b,2), \
                 getAt(3) op GetAt<vec<char,2> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<char,VALUE>::type,1> 
    dot (const vec<char,2>  &b) const{
      return vec< LCM<char,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<char,2> >(b,0) + 
                                     getAt(1) * GetAt<vec<char,2> >(b,1) + 
                                     getAt(2) * GetAt<vec<char,2> >(b,2) + 
                                     getAt(3) * GetAt<vec<char,2> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<char,2> >(b,0) + 
                                      getAt(1) * GetAt<vec<char,2> >(b,1) +
                                      getAt(2) * GetAt<vec<char,2> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<char,2> >(b,0) + 
                                      getAt(1) * GetAt<vec<char,2> >(b,1)):
                 getAt(0) * GetAt<vec<char,2> >(b,0));
                 
    }


    vec<typename LCM<char,VALUE>::type,1> 
    distance (const vec<char,2>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<char,2>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<char,2> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<char,2> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<char,2> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<char,2> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<float,3> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<float,3> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<float,3> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<float,3> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<float,tsize> questioncolon(const vec<float,3>  &b, 
						const vec<float,3>  &c)const {
        return vec<float,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<float,3> )/sizeof(float)
#else
#define TEMPL_TYPESIZE 3
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<float,3>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<float,3>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<float,3> >(b,0), \
                 getAt(1) op GetAt<vec<float,3> >(b,1), \
                 getAt(2) op GetAt<vec<float,3> >(b,2), \
                 getAt(3) op GetAt<vec<float,3> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<float,VALUE>::type,1> 
    dot (const vec<float,3>  &b) const{
      return vec< LCM<float,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<float,3> >(b,0) + 
                                     getAt(1) * GetAt<vec<float,3> >(b,1) + 
                                     getAt(2) * GetAt<vec<float,3> >(b,2) + 
                                     getAt(3) * GetAt<vec<float,3> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<float,3> >(b,0) + 
                                      getAt(1) * GetAt<vec<float,3> >(b,1) +
                                      getAt(2) * GetAt<vec<float,3> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<float,3> >(b,0) + 
                                      getAt(1) * GetAt<vec<float,3> >(b,1)):
                 getAt(0) * GetAt<vec<float,3> >(b,0));
                 
    }


    vec<typename LCM<float,VALUE>::type,1> 
    distance (const vec<float,3>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<float,3>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<float,3> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<float,3> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<float,3> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<float,3> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<int,3> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<int,3> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<int,3> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<int,3> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<int,tsize> questioncolon(const vec<int,3>  &b, 
						const vec<int,3>  &c)const {
        return vec<int,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<int,3> )/sizeof(int)
#else
#define TEMPL_TYPESIZE 3
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<int,3>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<int,3>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<int,3> >(b,0), \
                 getAt(1) op GetAt<vec<int,3> >(b,1), \
                 getAt(2) op GetAt<vec<int,3> >(b,2), \
                 getAt(3) op GetAt<vec<int,3> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<int,VALUE>::type,1> 
    dot (const vec<int,3>  &b) const{
      return vec< LCM<int,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<int,3> >(b,0) + 
                                     getAt(1) * GetAt<vec<int,3> >(b,1) + 
                                     getAt(2) * GetAt<vec<int,3> >(b,2) + 
                                     getAt(3) * GetAt<vec<int,3> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<int,3> >(b,0) + 
                                      getAt(1) * GetAt<vec<int,3> >(b,1) +
                                      getAt(2) * GetAt<vec<int,3> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<int,3> >(b,0) + 
                                      getAt(1) * GetAt<vec<int,3> >(b,1)):
                 getAt(0) * GetAt<vec<int,3> >(b,0));
                 
    }


    vec<typename LCM<int,VALUE>::type,1> 
    distance (const vec<int,3>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<int,3>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<int,3> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<int,3> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<int,3> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<int,3> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<char,3> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<char,3> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<char,3> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<char,3> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<char,tsize> questioncolon(const vec<char,3>  &b, 
						const vec<char,3>  &c)const {
        return vec<char,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<char,3> )/sizeof(char)
#else
#define TEMPL_TYPESIZE 3
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<char,3>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<char,3>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<char,3> >(b,0), \
                 getAt(1) op GetAt<vec<char,3> >(b,1), \
                 getAt(2) op GetAt<vec<char,3> >(b,2), \
                 getAt(3) op GetAt<vec<char,3> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<char,VALUE>::type,1> 
    dot (const vec<char,3>  &b) const{
      return vec< LCM<char,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<char,3> >(b,0) + 
                                     getAt(1) * GetAt<vec<char,3> >(b,1) + 
                                     getAt(2) * GetAt<vec<char,3> >(b,2) + 
                                     getAt(3) * GetAt<vec<char,3> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<char,3> >(b,0) + 
                                      getAt(1) * GetAt<vec<char,3> >(b,1) +
                                      getAt(2) * GetAt<vec<char,3> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<char,3> >(b,0) + 
                                      getAt(1) * GetAt<vec<char,3> >(b,1)):
                 getAt(0) * GetAt<vec<char,3> >(b,0));
                 
    }


    vec<typename LCM<char,VALUE>::type,1> 
    distance (const vec<char,3>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<char,3>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<char,3> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<char,3> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<char,3> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<char,3> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<float,4> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<float,4> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<float,4> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<float,4> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<float,tsize> questioncolon(const vec<float,4>  &b, 
						const vec<float,4>  &c)const {
        return vec<float,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<float,4> )/sizeof(float)
#else
#define TEMPL_TYPESIZE 4
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<float,4>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<float,4>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<float,4> >(b,0), \
                 getAt(1) op GetAt<vec<float,4> >(b,1), \
                 getAt(2) op GetAt<vec<float,4> >(b,2), \
                 getAt(3) op GetAt<vec<float,4> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<float,VALUE>::type,1> 
    dot (const vec<float,4>  &b) const{
      return vec< LCM<float,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<float,4> >(b,0) + 
                                     getAt(1) * GetAt<vec<float,4> >(b,1) + 
                                     getAt(2) * GetAt<vec<float,4> >(b,2) + 
                                     getAt(3) * GetAt<vec<float,4> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<float,4> >(b,0) + 
                                      getAt(1) * GetAt<vec<float,4> >(b,1) +
                                      getAt(2) * GetAt<vec<float,4> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<float,4> >(b,0) + 
                                      getAt(1) * GetAt<vec<float,4> >(b,1)):
                 getAt(0) * GetAt<vec<float,4> >(b,0));
                 
    }


    vec<typename LCM<float,VALUE>::type,1> 
    distance (const vec<float,4>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<float,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<float,4>  &b)const{ \
      return vec< TYPESPECIFIER<float, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<float,4> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<float,4> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<float,4> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<float,4> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<int,4> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<int,4> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<int,4> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<int,4> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<int,tsize> questioncolon(const vec<int,4>  &b, 
						const vec<int,4>  &c)const {
        return vec<int,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<int,4> )/sizeof(int)
#else
#define TEMPL_TYPESIZE 4
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<int,4>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<int,4>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<int,4> >(b,0), \
                 getAt(1) op GetAt<vec<int,4> >(b,1), \
                 getAt(2) op GetAt<vec<int,4> >(b,2), \
                 getAt(3) op GetAt<vec<int,4> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<int,VALUE>::type,1> 
    dot (const vec<int,4>  &b) const{
      return vec< LCM<int,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<int,4> >(b,0) + 
                                     getAt(1) * GetAt<vec<int,4> >(b,1) + 
                                     getAt(2) * GetAt<vec<int,4> >(b,2) + 
                                     getAt(3) * GetAt<vec<int,4> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<int,4> >(b,0) + 
                                      getAt(1) * GetAt<vec<int,4> >(b,1) +
                                      getAt(2) * GetAt<vec<int,4> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<int,4> >(b,0) + 
                                      getAt(1) * GetAt<vec<int,4> >(b,1)):
                 getAt(0) * GetAt<vec<int,4> >(b,0));
                 
    }


    vec<typename LCM<int,VALUE>::type,1> 
    distance (const vec<int,4>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<int,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<int,4>  &b)const{ \
      return vec< TYPESPECIFIER<int, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<int,4> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<int,4> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<int,4> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<int,4> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 323 "brtvector.hpp"
#define VECTOR_TEMPLATIZED_FUNCTIONS
    
      vec<VALUE,4> mask4 (const vec<char,4> &in,int X, int Y,int Z,int W) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        f[W]=in.getAt(3);
        return vec<VALUE,4>(unsafeGetAt(X),
                            unsafeGetAt(Y),
                            unsafeGetAt(Z),
                            unsafeGetAt(W));
    }
    
      vec<VALUE,3> mask3 (const vec<char,4> &in,int X,int Y,int Z) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        f[Z]=in.getAt(2);
        return vec<VALUE,3>(unsafeGetAt(X),unsafeGetAt(Y),unsafeGetAt(Z));
    }
     
      vec<VALUE,2> mask2 (const vec<char,4> &in,int X,int Y) {
        f[X]=in.getAt(0);
        f[Y]=in.getAt(1);
        return vec<VALUE,2>(unsafeGetAt(X),unsafeGetAt(Y));
    }
     
      vec<VALUE,1> mask1 (const vec<char,4> &in,int X) {
        f[X]=in.getAt(0);
        return vec<VALUE,1>(unsafeGetAt(X));
    }    
     
      vec<char,tsize> questioncolon(const vec<char,4>  &b, 
						const vec<char,4>  &c)const {
        return vec<char,tsize>
            (singlequestioncolon(getAt(0),b.getAt(0),c.getAt(0)),
             singlequestioncolon(getAt(1),b.getAt(1),c.getAt(1)),
             singlequestioncolon(getAt(2),b.getAt(2),c.getAt(2)),
             singlequestioncolon(getAt(3),b.getAt(3),c.getAt(3)));
    }
#if defined (_ARRGH) && (_ARRGH <= 1200)
#define TEMPL_TYPESIZE sizeof(vec<char,4> )/sizeof(char)
#else
#define TEMPL_TYPESIZE 4
#endif
#define BROOK_BINARY_OP(op,opgets,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<char,4>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                                VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size>(*this) opgets b; \
    }
    BROOK_BINARY_OP(*,*=,LCM);
    BROOK_BINARY_OP(/,/=,LCM);
    BROOK_BINARY_OP(+,+=,LCM);
    BROOK_BINARY_OP(-,-=,LCM);
    BROOK_BINARY_OP(%,%=,LCM);
#undef BROOK_BINARY_OP
#define BROOK_BINARY_OP(op,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> operator op (const vec<char,4>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (getAt(0) op GetAt<vec<char,4> >(b,0), \
                 getAt(1) op GetAt<vec<char,4> >(b,1), \
                 getAt(2) op GetAt<vec<char,4> >(b,2), \
                 getAt(3) op GetAt<vec<char,4> >(b,3)); \
    }
    BROOK_BINARY_OP(||,LCM);
    BROOK_BINARY_OP(&&,LCM);
    BROOK_BINARY_OP(<,COMMON_CHAR)
    BROOK_BINARY_OP(>,COMMON_CHAR)        
    BROOK_BINARY_OP(<=,COMMON_CHAR)
    BROOK_BINARY_OP(>=,COMMON_CHAR)        
    BROOK_BINARY_OP(!=,COMMON_CHAR)
    BROOK_BINARY_OP(==,COMMON_CHAR)
#undef BROOK_BINARY_OP

    vec<typename LCM<char,VALUE>::type,1> 
    dot (const vec<char,4>  &b) const{
      return vec< LCM<char,
                                           VALUE>::type, 1> 
                ((LUB<TEMPL_TYPESIZE,
                     tsize>::size)==4?(getAt(0) * GetAt<vec<char,4> >(b,0) + 
                                     getAt(1) * GetAt<vec<char,4> >(b,1) + 
                                     getAt(2) * GetAt<vec<char,4> >(b,2) + 
                                     getAt(3) * GetAt<vec<char,4> >(b,3)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==3?(getAt(0) * GetAt<vec<char,4> >(b,0) + 
                                      getAt(1) * GetAt<vec<char,4> >(b,1) +
                                      getAt(2) * GetAt<vec<char,4> >(b,2)):
                 (LUB<TEMPL_TYPESIZE,
                     tsize>::size)==2?(getAt(0) * GetAt<vec<char,4> >(b,0) + 
                                      getAt(1) * GetAt<vec<char,4> >(b,1)):
                 getAt(0) * GetAt<vec<char,4> >(b,0));
                 
    }


    vec<typename LCM<char,VALUE>::type,1> 
    distance (const vec<char,4>  &b) const{
      return (b-*this).length();
    }



#define BROOK_BINARY_OP(op, subop,TYPESPECIFIER)           \
    vec<typename TYPESPECIFIER<char,VALUE>::type, \
       LUB<TEMPL_TYPESIZE,tsize>::size> op (const vec<char,4>  &b)const{ \
      return vec< TYPESPECIFIER<char, \
                                           VALUE>::type, \
		 LUB<TEMPL_TYPESIZE,tsize>::size> \
                (::subop(getAt(0) , GetAt<vec<char,4> >(b,0)), \
                 ::subop(getAt(1) , GetAt<vec<char,4> >(b,1)), \
                 ::subop(getAt(2) , GetAt<vec<char,4> >(b,2)), \
                 ::subop(getAt(3) , GetAt<vec<char,4> >(b,3))); \
    }
    BROOK_BINARY_OP(atan2,atan2,LCM)
    BROOK_BINARY_OP(fmod,fmod,LCM)
    BROOK_BINARY_OP(pow,pow,LCM);
    BROOK_BINARY_OP(step_float,step_float,LCM);
    BROOK_BINARY_OP(min_float,min_float,LCM);
    BROOK_BINARY_OP(max_float,max_float,LCM);
#undef TEMPL_TYPESIZE
#undef BROOK_BINARY_OP    
#undef VECTOR_TEMPLATIZED_FUNCTIONS
#line 448 "brtvector.hpp"
#line 448 "brtvector.hpp"

#line 448 "brtvector.hpp"

#line 448 "brtvector.hpp"

#line 448 "brtvector.hpp"

#line 448 "brtvector.hpp"

};


template <> inline vec<float,1> singlequestioncolon (const vec<float,1> &a,
                                                     const vec<float,1> &b,
                                                     const vec<float,1> &c) {
    return a.unsafeGetAt(0)?b:c;
}

template <> inline vec<float,2> singlequestioncolon (const vec<float,2> & a,
                                                     const vec<float,2> &b,
                                                     const vec<float,2> &c) {
    return vec<float,2> (a.unsafeGetAt(0)?b.unsafeGetAt(0):c.unsafeGetAt(0),
                         a.unsafeGetAt(1)?b.unsafeGetAt(1):c.unsafeGetAt(1));
}
template <> inline vec<float,3> singlequestioncolon (const vec<float,3> &a,
                                              const vec<float,3> &b,
                                              const vec<float,3> &c) {
    return vec<float,3> (a.unsafeGetAt(0)?b.unsafeGetAt(0):c.unsafeGetAt(0),
                         a.unsafeGetAt(1)?b.unsafeGetAt(1):c.unsafeGetAt(1),
                         a.unsafeGetAt(2)?b.unsafeGetAt(2):c.unsafeGetAt(2));
}
template <> inline vec<float,4> singlequestioncolon (const vec<float,4> &a,
                                              const vec<float,4> &b,
                                              const vec<float,4> &c) {
    return vec<float,4> (a.unsafeGetAt(0)?b.unsafeGetAt(0):c.unsafeGetAt(0),
                         a.unsafeGetAt(1)?b.unsafeGetAt(1):c.unsafeGetAt(1),
                         a.unsafeGetAt(2)?b.unsafeGetAt(2):c.unsafeGetAt(2),
                         a.unsafeGetAt(3)?b.unsafeGetAt(3):c.unsafeGetAt(3));
}




template <class T> 
  std::ostream& operator^ (std::ostream& os, const T & a){
    if (T::size==1) {
        os << a.getAt(0);
    }else {
        os << "{";
        for (unsigned int i=0;i<T::size;++i) {
            os << a.getAt(i)<<(i!=T::size-1?", ":"");
        }
        os << "}";
    }
    return os;
}

#define VECX_CLASS(NAME,TYPE,X) \
inline std::ostream& operator << (std::ostream&a,const vec<TYPE,X> & b) { \
    return a^b; \
}   \
typedef vec<TYPE,X> NAME

VECX_CLASS(__BrtFloat1,float,1);
VECX_CLASS(__BrtFloat2,float,2);
VECX_CLASS(__BrtFloat3,float,3);
VECX_CLASS(__BrtFloat4,float,4);
VECX_CLASS(__BrtChar1,char,1);
VECX_CLASS(__BrtChar2,char,2);
VECX_CLASS(__BrtChar3,char,3);
VECX_CLASS(__BrtChar4,char,4);
VECX_CLASS(__BrtInt1,int,1);
VECX_CLASS(__BrtInt2,int,2);
VECX_CLASS(__BrtInt3,int,3);
VECX_CLASS(__BrtInt4,int,4);
#undef VECX_CLASS
#define MATRIXXY_CLASS(TYPE,X,Y) \
inline std::ostream& operator << (std::ostream&a, \
                                  const vec<TYPE##X,Y> & b) { \
    return a^b; \
}   \
typedef vec<TYPE##X,Y> TYPE##X##x##Y

MATRIXXY_CLASS(__BrtFloat,4,4);
MATRIXXY_CLASS(__BrtFloat,4,3);
MATRIXXY_CLASS(__BrtFloat,4,2);
MATRIXXY_CLASS(__BrtFloat,4,1);
MATRIXXY_CLASS(__BrtFloat,3,4);
MATRIXXY_CLASS(__BrtFloat,3,3);
MATRIXXY_CLASS(__BrtFloat,3,2);
MATRIXXY_CLASS(__BrtFloat,3,1);
MATRIXXY_CLASS(__BrtFloat,2,4);
MATRIXXY_CLASS(__BrtFloat,2,3);
MATRIXXY_CLASS(__BrtFloat,2,2);
MATRIXXY_CLASS(__BrtFloat,2,1);
MATRIXXY_CLASS(__BrtFloat,1,4);
MATRIXXY_CLASS(__BrtFloat,1,3);
MATRIXXY_CLASS(__BrtFloat,1,2);
MATRIXXY_CLASS(__BrtFloat,1,1);

MATRIXXY_CLASS(__BrtInt,4,4);
MATRIXXY_CLASS(__BrtInt,4,3);
MATRIXXY_CLASS(__BrtInt,4,2);
MATRIXXY_CLASS(__BrtInt,4,1);
MATRIXXY_CLASS(__BrtInt,3,4);
MATRIXXY_CLASS(__BrtInt,3,3);
MATRIXXY_CLASS(__BrtInt,3,2);
MATRIXXY_CLASS(__BrtInt,3,1);
MATRIXXY_CLASS(__BrtInt,2,4);
MATRIXXY_CLASS(__BrtInt,2,3);
MATRIXXY_CLASS(__BrtInt,2,2);
MATRIXXY_CLASS(__BrtInt,2,1);
MATRIXXY_CLASS(__BrtInt,1,4);
MATRIXXY_CLASS(__BrtInt,1,3);
MATRIXXY_CLASS(__BrtInt,1,2);
MATRIXXY_CLASS(__BrtInt,1,1);

MATRIXXY_CLASS(__BrtChar,4,4);
MATRIXXY_CLASS(__BrtChar,4,3);
MATRIXXY_CLASS(__BrtChar,4,2);
MATRIXXY_CLASS(__BrtChar,4,1);
MATRIXXY_CLASS(__BrtChar,3,4);
MATRIXXY_CLASS(__BrtChar,3,3);
MATRIXXY_CLASS(__BrtChar,3,2);
MATRIXXY_CLASS(__BrtChar,3,1);
MATRIXXY_CLASS(__BrtChar,2,4);
MATRIXXY_CLASS(__BrtChar,2,3);
MATRIXXY_CLASS(__BrtChar,2,2);
MATRIXXY_CLASS(__BrtChar,2,1);
MATRIXXY_CLASS(__BrtChar,1,4);
MATRIXXY_CLASS(__BrtChar,1,3);
MATRIXXY_CLASS(__BrtChar,1,2);
MATRIXXY_CLASS(__BrtChar,1,1);

#undef MATRIXXY_CLASS


/*

inline static __BrtFloat4 computeIndexOf(unsigned int i,
                                         unsigned int dim,
                                         const unsigned int *extents){
   return __BrtFloat4((float)(i%extents[dim-1]),
                      dim>1?(float)((i/extents[dim-1])%extents[dim-2]):0.0f,
                      dim>2?(float)((i/(extents[dim-2]*extents[dim-1]))
                            %extents[dim-3]):0.0f,
                      dim>3?(float)(i/(extents[dim-3]
                                       *extents[dim-2]
                                       *extents[dim-1])):0.0f
                      );
}

inline static void incrementIndexOf(__BrtFloat4 &indexof,
                                    unsigned int dim, 
                                    const unsigned int *extents) {
   indexof.unsafeGetAt(0)+=1;
   if (indexof.unsafeGetAt(0)>=extents[dim-1]) {
      indexof.unsafeGetAt(0)-=extents[dim-1];
      indexof.unsafeGetAt(1)+=1;
      if (dim>1&&indexof.unsafeGetAt(1)>=extents[dim-2]) {
         indexof.unsafeGetAt(1)-=extents[dim-2];
         indexof.unsafeGetAt(2)+=1;
         if (dim>2&&indexof.unsafeGetAt(2)>=extents[dim-3]) {
            indexof.unsafeGetAt(2)-=extents[dim-3];
            indexof.unsafeGetAt(3)+=1;
         }
      }
   }
   }*/
#endif
#endif


