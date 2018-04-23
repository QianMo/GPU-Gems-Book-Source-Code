#ifndef BRT_HPP
#define BRT_HPP
#ifndef _WIN32
#include <stdlib.h>
#endif
extern "C" {
#include <assert.h>
};

#include <brook/kerneldesc.hpp>
#include <brook/brtvector.hpp>
typedef struct fixed {
  fixed(unsigned char _x) { x =_x;}
  fixed(float _x) {if (_x>1) _x=1; if (_x<0)_x=0; x = (unsigned char)(_x*255);}
   fixed(float _x,float _y,float _z,float _w) {if (_x>1) _x=1; if (_x<0)_x=0; x = (unsigned char)(_x*255);}
  fixed(void) {}
  operator __BrtFloat1()const {return __BrtFloat1(((float)x)/255.0f);}
  template <class T> T castToArg(const T&dummy)const{return T(((float)x)/255.0f);}
  fixed& operator = (const __BrtFloat1&input){
    return (*this=fixed(input.unsafeGetAt(0)));
  }
      unsigned char x;//unsigned char pad1,pad2,pad3;
} fixed;

typedef struct fixed2 {
  fixed2(float _x, float _y) { x = (unsigned char)(_x*255); y = (unsigned char)(_y*255); }
   fixed2(float _x, float _y, float _z, float _w) { x = (unsigned char)(_x*255); y = (unsigned char)(_y*255); }
  fixed2(unsigned char _x, unsigned char _y) { x = _x;y=_y;}
  fixed2(void) {}
  unsigned char x,y,pad2,pad3;
  operator __BrtFloat2() const{return __BrtFloat2(((float)x)/255.0f,((float)y)/255.0f);}
  template <class T> T castToArg(const T &dummy) const{return T(((float)x)/255.0f,((float)y)/255.0f);}
} fixed2;

typedef struct fixed3 {
  fixed3(float _x, float _y, float _z) { x = (unsigned char)(_x*255); y = (unsigned char)(_y*255); z=(unsigned char)(_z*255);}
   fixed3(float _x, float _y, float _z, float _w) { x = (unsigned char)(_x*255); y = (unsigned char)(_y*255); z=(unsigned char)(_z*255);}
  fixed3(unsigned char _x, unsigned char _y, unsigned char _z) { x = _x;y=_y;z=_z;}
  fixed3(void) {}
  operator __BrtFloat3() const{return __BrtFloat3(((float)x)/255.0f,((float)y)/255.0f,((float)z)/255.0f);}
  template <class T> T castToArg(const T&dummy) const {return T(((float)x)/255.0f,((float)y)/255.0f,((float)z)/255.0f);}
  unsigned char x,y,z,pad3;
} fixed3;

typedef struct fixed4 {
  fixed4(float _x, float _y, float _z, float _w) { x = (unsigned char)(_x*255); y = (unsigned char)(_y*255); z=(unsigned char)(_z*255); w=(unsigned char)(_w*255);}
  fixed4(unsigned char _x, unsigned char _y, unsigned char _z, unsigned char _w) { x = _x;y=_y;z=_z;w=_w;}
  fixed4(void) {}
  unsigned char x,y,z,w;
   operator __BrtFloat4() const{return __BrtFloat4(((float)x)/255.0f,((float)y)/255.0f,((float)z)/255.0f,((float)w)/255.0f);}
  template <class T> T castToArg(const T&dummy)const{return T(((float)x)/255.0f,((float)y)/255.0f,((float)z)/255.0f,((float)w)/255.0f);}

} fixed4;

typedef struct float2 {
  float2(float _x, float _y) { x = _x; y = _y; }
  float2(void) {}
  operator __BrtFloat2()const {return __BrtFloat2(x,y);}
  float x,y;
} float2;


typedef struct double2 {
  double2(double _x, double _y) { x = _x; y = _y; }
  double2(void) {}
  operator __BrtDouble2()const {return __BrtDouble2(x,y);}
  double x,y;
} double2;


typedef struct float3 {
  float3(float _x, float _y, float _z) { x = _x; y = _y; z = _z; }
  float3(void) {}
  operator __BrtFloat3()const {return __BrtFloat3(x,y,z);}
  float x,y,z;
} float3;

typedef struct float4 {
  float4(float _x, float _y, float _z, float _w) {
     x = _x; y = _y; z = _z; w = _w;
  }
  float4(void) {}
  operator __BrtFloat4()const {return __BrtFloat4(x,y,z,w);}
  float x,y,z,w;
} float4;

typedef struct int2
{
    int2(void) {}
    int2( int inX, int inY )
        : x(inX), y(inY) {}
    operator int*() { return (int*)this; }
    operator const int*() const { return (const int*)this; }
    int x, y;
} int2;

typedef struct int3
{
    int3(void) {}
    int3( int inX, int inY, int inZ )
        : x(inX), y(inY), z(inZ) {}
    operator int*() { return (int*)this; }
    operator const int*() const { return (const int*)this; }
    int x, y, z;
} int3;

typedef struct int4
{
    int4(void) {}
    int4( int inX, int inY, int inZ, int inW )
        : x(inX), y(inY), z(inZ), w(inW) {}
    operator int*() { return (int*)this; }
    operator const int*() const { return (const int*)this; }
    int x, y, z, w;
} int4;

enum __BRTStreamType {
   __BRTNONE=-1,
   __BRTSTREAM=0,//stream of stream is illegal. Used in reduce to stream.
   __BRTFLOAT=1,
   __BRTFLOAT2=2,
   __BRTFLOAT3=3,
   __BRTFLOAT4=4,
    __BRTFIXED=5,
    __BRTFIXED2=6,
    __BRTFIXED3=7,
    __BRTFIXED4=8,
    __BRTHALF=9,
    __BRTHALF2=10,
    __BRTHALF3=11,
    __BRTHALF4=12,
    __BRTDOUBLE=13,
    __BRTDOUBLE2=14,
};
float getSentinel();

namespace brook {

  class StreamInterface;
  class Stream;
  class Iter;
  class Kernel;

  class stream;
  class iter;
  class kernel;

  static const unsigned int MAXPROGLENGTH = 1024*32;
  static const unsigned int MAXSTREsAMDIMS = 8;

  enum StreamType {
    __BRTNONE=-1,
    __BRTSTREAM=0, //stream of stream is illegal. Used in reduce to stream.
    __BRTFLOAT=1,
    __BRTFLOAT2=2,
    __BRTFLOAT3=3,
    __BRTFLOAT4=4,
    __BRTFIXED=5,
    __BRTFIXED2=6,
    __BRTFIXED3=7,
    __BRTFIXED4=8,
    __BRTHALF=9,
    __BRTHALF2=10,
    __BRTHALF3=11,
    __BRTHALF4=12,
    __BRTDOUBLE=13,
    __BRTDOUBLE2=14,
  };
  unsigned int getElementSize(StreamType);
  template<typename T>
  const ::brook::StreamType* getStreamType(T* unused=0);

  template<>
  inline const ::brook::StreamType* getStreamType(float*) {
    static const ::brook::StreamType result[] = {__BRTFLOAT, __BRTNONE};
    return result;
  }

  template<>
  inline const ::brook::StreamType* getStreamType(float2*) {
    static const ::brook::StreamType result[] = {__BRTFLOAT2, __BRTNONE};
    return result;
  }

  template<>
  inline const StreamType* getStreamType(float3*) {
    static const ::brook::StreamType result[] = {__BRTFLOAT3, __BRTNONE};
    return result;
  }

  template<>
  inline const ::brook::StreamType* getStreamType(float4*) {
    static const ::brook::StreamType result[] = {__BRTFLOAT4, __BRTNONE};
    return result;
  }
  template<>
  inline const ::brook::StreamType* getStreamType(fixed*) {
     static const ::brook::StreamType result[] = {__BRTFIXED,__BRTNONE};
     return result;
  }

  template<>
  inline const ::brook::StreamType* getStreamType(fixed2*) {
     static const ::brook::StreamType result[] = {__BRTFIXED2,__BRTNONE};
     return result;
  }

  template<>
  inline const ::brook::StreamType* getStreamType(fixed3*) {
     static const ::brook::StreamType result[] = {__BRTFIXED3,__BRTNONE};
     return result;
  }

  template<>
  inline const ::brook::StreamType* getStreamType(fixed4*) {
     static const ::brook::StreamType result[] = {__BRTFIXED4,__BRTNONE};
     return result;
  }
  template<>
  inline const ::brook::StreamType* getStreamType(double*) {
     static const ::brook::StreamType result[] = {__BRTDOUBLE,__BRTNONE};
     return result;
  }
  template<>
  inline const ::brook::StreamType* getStreamType(double2*) {
     static const ::brook::StreamType result[] = {__BRTDOUBLE2,__BRTNONE};
     return result;
  }
 
  /****************************************************/
  /*****           Runtime routines            ********/

  class Runtime;

  void initialize( const char* inRuntimeName, 
                   void* inContextValue = 0 );
  Runtime* createRuntime( bool useAddressTranslation );


  /************************************************/
  /**********       Stream classes      ***********/

  class StreamInterface {
  private:
    size_t referenceCount;

  protected:
    StreamInterface()
      : referenceCount(1) {}
    virtual ~StreamInterface(){}

  public:
    void acquireReference() { referenceCount++; }
    void releaseReference() {
      if( --referenceCount == 0 ) delete this;
    }
    
    enum USAGEFLAGS {NONE=0x0,READ=0x1,WRITE=0x2,READWRITE=0x3};

    virtual void * getData (unsigned int flags)=0;
    virtual void releaseData(unsigned int flags)=0;
    virtual void readItem(void * p,unsigned int * index);
    virtual void * fetchItemPtr (void *data, unsigned int * index);

    virtual const unsigned int * getExtents() const=0;
    virtual unsigned int getDimension() const {return 0;}
    virtual const unsigned int * getDomainMin() const {assert(0); return 0;}
    virtual const unsigned int * getDomainMax() const {assert(0); return 0;}

    unsigned int getElementSize() const;
    virtual unsigned int getFieldCount() const = 0;
    virtual StreamType getIndexedFieldType(unsigned int i) const=0;

    virtual unsigned int         getTotalSize() const {
       unsigned int ret=1;
       unsigned int dim=getDimension();
       const unsigned int * extents = getExtents();
       for (unsigned int i=0;i<dim;++i) {
          ret*=extents[i];
       }
       return ret;
    }

    // functions for getting at low-level representation,
    // so that an application can render and simulate
    virtual void * getIndexedFieldRenderData(unsigned int i);
    virtual void   synchronizeRenderData();

    /* Used CPU Runtime Only */
    virtual void * fetchElem(const unsigned int pos[], 
                             const unsigned int bounds[],
                             unsigned int dim) {
      assert (0); return (void *) 0;
    }

    __BrtFloat4 
    StreamInterface::computeIndexOf(unsigned int linear_index);

    virtual bool   isCPU() const  { return false; }

  };

  class Stream : public StreamInterface {
  public:
    Stream () {}
    virtual void Read(const void* inData) = 0;
    virtual void Write(void* outData) = 0;
    virtual Stream* Domain(int min, int max) {
        assert(0); return 0;
    }
    virtual Stream* Domain(const int2& min, const int2& max) {
        assert(0); return 0;
    }
    virtual Stream* Domain(const int3& min, const int3& max) {
        assert(0); return 0;
    }
    virtual Stream* Domain(const int4& min, const int4& max) {
        assert(0); return 0;
    }
    
    //virtual unsigned int getStride() const {return sizeof(float)*getStreamType();}
    //virtual __BRTStreamType getStreamType ()const{return type;}

  protected:
    virtual ~Stream() {}
  };

  class Iter : public StreamInterface {
  public:
    Iter (::brook::StreamType type) {this->type=type;}
    virtual unsigned int getFieldCount() const { return 1; }
    virtual ::brook::StreamType getIndexedFieldType(unsigned int i) const {
      assert(i == 0);
      return type;
    }
    
  protected:
    ::brook::StreamType type;
    virtual ~Iter() {}
  };

  /*** Main Kernel Class ***/

  class Kernel {
  public:
    Kernel() {}
    virtual void PushStream(::brook::Stream *s) = 0;
    virtual void PushIter(class Iter * v) = 0;
    virtual void PushConstant(const float &val) = 0;  
    virtual void PushConstant(const float2 &val) = 0;  
    virtual void PushConstant(const float3 &val) = 0; 
    virtual void PushConstant(const float4 &val) = 0;
    virtual void PushReduce (void * val,  ::brook::StreamType type)=0;
    virtual void PushGatherStream(::brook::Stream *s) = 0;
    virtual void PushOutput(::brook::Stream *s) = 0;
    virtual void Map() = 0;
    virtual void Reduce() = 0;
    virtual void Release() {delete this;}

    // CPU version only
    virtual bool   Continue() {assert (0); return false;}
    virtual void * FetchElem(StreamInterface *s) {
      assert (0); return (void *) 0;
    }

  protected:
    virtual ~Kernel() {}
  };


  /*************************************************************/
  /****     Stub classes for initialization               ******/

  class stream
  {
  public:
    stream();
    stream( const ::brook::stream& );
    stream& operator=( const ::brook::stream& );

    // easy-to-use constructors for C++ interface
    template<typename T>
    static ::brook::stream create( int inExtent0 ) {
      return stream( getStreamType((T*)0), inExtent0, -1 );
    }

    template<typename T>
    static ::brook::stream create( int y, int x ) {
      return stream( ::brook::getStreamType((T*)0), y, x, -1 );
    }

    // for domain
    stream(::brook::Stream *s) {_stream = s;}

    // standard constructors for BRCC-generated code
    stream(const ::brook::StreamType*,...);
    stream(const unsigned int extents[],
           unsigned int dims,
           const ::brook::StreamType *type);
    stream( const ::brook::iter& );

    ~stream();

    void swap(::brook::stream& other) {
      ::brook::Stream* s = other._stream;
      other._stream = _stream;
      _stream = s;
    }

    operator ::brook::Stream*() const {
      return _stream;
    }

    operator ::brook::StreamInterface*() const {
      return _stream;
    }

    ::brook::Stream* operator->() const {
      return _stream;
    }

    void read (const void *p) {
      _stream->Read(p);
    }

    void write (void *p) {
      _stream->Write(p);
    }

    ::brook::stream domain(int min, int max) {
      return stream(_stream->Domain(min, max));
    }

    ::brook::stream domain(const int2 &min, const int2 &max) {
      return stream(_stream->Domain(min, max));
    }

    ::brook::stream domain(const int3 &min, const int3 &max) {
      return stream(_stream->Domain(min, max));
    }

    ::brook::stream domain(const int4 &min, const int4 &max) {
      return stream(_stream->Domain(min, max));
    }

  private:

    ::brook::Stream* _stream;
  };

  class iter {
  public:
    iter(::brook::StreamType, ...);
    ~iter() {
      if(_iter) _iter->releaseReference();
    }

    operator ::brook::Iter*() const {
      return _iter;
    }

    operator ::brook::StreamInterface*() const {
      return _iter;
    }

    ::brook::Iter* operator->() const {
      return _iter;
    }

  private:
    iter( const ::brook::iter& ); // no copy constructor
    ::brook::Iter* _iter;
  };


  class kernel {
  public:
    
    kernel(const void* code[]);
    
    ~kernel() {
      if( _kernel != 0 )
        _kernel->Release();
    }
    
    operator brook::Kernel*() const {
      return _kernel;
    }
    
    brook::Kernel* operator->() const {
      return _kernel;
    }
    
  private:
    kernel( const kernel& ); // no copy constructor
    brook::Kernel* _kernel;
  };


  /* For vout */
  inline static void maxDimension(unsigned int * out, 
                                  const unsigned int * in,
                                  int dims) {
    for (int i=0;i<dims;++i) {
      if (in[i]>(unsigned int)out[i])out[i]=in[i];
    }
  }
  float getSentinel();
  ::brook::stream* sentinelStream(int dim);


  void readItem(brook::StreamInterface *s, void * p, ... );
}

/***********************************************************/
/*******      S T R E A M      O P E R A T O R S       *****/

void streamPrint(brook::StreamInterface*s, bool flatten=false);

inline static float4 streamSize(::brook::stream &s) {
  unsigned int i;
  const unsigned int * extents = s->getExtents();
  unsigned int dim             = s->getDimension();
  
  float4 ret(0.0f,0.0f,0.0f,0.0f);
  
  switch (s->getDimension()) {
  case 3:
    ret.z=(float)extents[dim-3];
  case 2:
    ret.y=(float)extents[dim-2];
  case 1:
    ret.x=(float)extents[dim-1];
    break;
  case 4:
  default:
    for (i=0;i<dim-3;++i) ret.w+=(float)extents[i];
    ret.z=(float)extents[dim-3];
    ret.y=(float)extents[dim-2];
    ret.x=(float)extents[dim-1];
    break;
  }
  return ret;
}

typedef ::brook::iter __BRTIter;
  
inline static void streamRead( ::brook::Stream *s, void *p) {
  s->Read(p);
}

inline static void streamWrite( ::brook::Stream *s, void *p) {
  s->Write(p);
}

inline static void streamSwap( ::brook::stream &x, ::brook::stream &y) {
  x.swap(y);
}


#define streamGatherOp(a,b,c,d) \
  __streamGatherOrScatterOp(a,b,c, \
    (void (*)(void *, void *))__##d##_cpu_inner,true)

#define streamScatterOp(a,b,c,d) \
  __streamGatherOrScatterOp(a,b,c, \
    (void (*)(void *, void *))__##d##_cpu_inner,false)

void __streamGatherOrScatterOp (::brook::StreamInterface *dst, 
                                ::brook::StreamInterface *index,
                                ::brook::StreamInterface *src,
                                void (*func) (void *, void *),
                                bool GatherNotScatter);
template <class T> class Addressable: public T {
 public:
  mutable void *address;
  template <typename U>Addressable(U* Address)
    : T(Address->castToArg(T())){
    this->address=Address;
  }
  //  Addressable(const T &t, void * Address):T(t){this->address=Address;}
  Addressable(const T&t):T(t){address=NULL;} 
  Addressable(const Addressable<T>& b):T(b){ 
    this->address=b.address;    
  }
  Addressable<T>&operator = (const T&b) {
    *static_cast<T*>(this)=static_cast<const T&>(b);
    return *this;
  }
  Addressable<T>& operator = (const Addressable<T>& b) {
    *static_cast<T*>(this)=static_cast<const T&>(b);
    if (address==NULL) this->address=b.address;
    return *this;
  }
};
template <class T> const void * getStreamAddress(const T* a) {
   return static_cast<const Addressable<T>* > (a)->address;
}
#if 0
#define indexof(a) __indexof(a.address)
#else
#define indexof(a) __indexof(getStreamAddress(&a))
#endif
__BrtFloat4 __indexof (const void *);

// TIM: adding conditional magick for raytracer
void streamEnableWriteMask();
void streamDisableWriteMask();
void streamSetWriteMask( ::brook::stream& );
void streamBeginWriteQuery();
int streamEndWriteQuery();

void hackStreamRestoreContext();

#endif

