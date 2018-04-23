#ifndef _BROOK_CPU_HPP
#define _BROOK_CPU_HPP
#include <vector>
#include "../runtime.hpp"
#include <brook/brtarray.hpp>

#ifdef _WIN32
#define THREADRETURNTYPE unsigned long 
#else
#define THREADRETURNTYPE void *
#endif

namespace brook {
  extern const char * CPU_RUNTIME_STRING;	

  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  class CPUKernel : public Kernel {
  public:
    CPUKernel(const void * source []);
    virtual void PushStream(Stream *s);
    virtual void PushConstant(const float &val);  
    virtual void PushConstant(const float2 &val);  
    virtual void PushConstant(const float3 &val); 
    virtual void PushConstant(const float4 &val);
    virtual void PushIter(class Iter * i);
    virtual void PushGatherStream(Stream *s);
    virtual void PushReduce (void * val, StreamType type);
    virtual void PushOutput(Stream *s);
    virtual void Map();
    virtual void Reduce();

    virtual void   PushStreamInterface(StreamInterface * s);
    virtual void * FetchElem(StreamInterface *s);
    virtual bool   Continue();

    virtual void Release();

    // public for indexof
    std::vector<StreamInterface *> input_args;
    std::vector<StreamInterface *> output_args;

  protected:

    virtual ~CPUKernel();

    typedef void callable(::brook::Kernel *__k, 
                          const std::vector<void *>&args);

    callable * func;

    std::vector<void *> args;
    std::vector<StreamInterface *> freeme;
    std::vector<__BrtArray<unsigned char> *> freeme_array;

    unsigned int dims;
    unsigned int *curpos;
    unsigned int *extents;
    unsigned int *minpos;
    unsigned int *maxpos;

    bool         is_reduce;
    bool         reduce_is_scalar;
    void         *reduce_arg;
    void         *reduce_value;



    void Cleanup();
  };

  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  class CPUStream: public Stream {
  public:
    
    CPUStream (unsigned int inFieldCount, 
               const StreamType* inFieldTypes ,
               unsigned int dims, 
               const unsigned int extents[]);
 
    // Domain Constructor
    CPUStream (const CPUStream &a, 
               const int *min, 
               const int *max);
 
    // Default Constructor
    CPUStream () {}
    
    /* Stream Operators */
    virtual void     Read(const void* inData);
    virtual void     Write(void* outData);
    virtual Stream* Domain(int min, int max);
    virtual Stream* Domain(const int2& min, const int2& max);
    virtual Stream* Domain(const int3& min, const int3& max);
    virtual Stream* Domain(const int4& min, const int4& max);
    
    /* Internal runtime functions */
    virtual void *getData(unsigned int flags) {return data;}
    virtual void releaseData(unsigned int flags){}

    virtual const unsigned int * getExtents()   const { return extents; }
    virtual unsigned int         getDimension() const { return dims; }
    virtual unsigned int         getTotalSize() const { return totalsize; }
    virtual const unsigned int * getDomainMin() const { return domain_min; }
    virtual const unsigned int * getDomainMax() const { return domain_max; }

    virtual unsigned int         getFieldCount() const {
      return elementType.size();
    }

    virtual StreamType           getIndexedFieldType(unsigned int i) const {
      return elementType[i];
    }

    virtual void *  fetchElem(const unsigned int pos[], 
                              const unsigned int bounds[],
                              unsigned int dim);
    virtual bool    isCPU() const {return true;}

    virtual ~CPUStream();

    // public for indexof
    unsigned int malloced_size;
    unsigned int stride;

    protected:
  
    std::vector<StreamType> elementType;
    void * data;

    unsigned int dims;
    unsigned int totalsize;

    unsigned int *extents;
    unsigned int *domain_min;
    unsigned int *domain_max;
    unsigned int *pos;

    bool isDerived;
  };
   
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  class CPUIter:public Iter {
  protected:
    CPUStream stream;
    virtual ~CPUIter() {}
    void allocateStream(unsigned int dims, 
                        const unsigned int extents[],
                        const float ranges[]);
  public:
    CPUIter(StreamType type, unsigned int dims, 
            const unsigned int extents[], const float ranges[])
      :Iter(type),stream(1,&type,dims,extents){
      allocateStream(dims,extents,ranges);//now we always have this
    }
    
    virtual void * getData (unsigned int flags){return stream.getData(flags);}
    virtual void releaseData(unsigned int flags){stream.releaseData(flags);}
    virtual const unsigned int * getExtents()const {return stream.getExtents();}
    virtual const unsigned int * getDomainMin() const { return stream.getDomainMin(); }
    virtual const unsigned int * getDomainMax() const { return stream.getDomainMax(); }
    virtual unsigned int getDimension()const {return stream.getDimension();}
    virtual unsigned int getTotalSize() const {return stream.getTotalSize();}

    virtual void *  fetchElem(const unsigned int pos[], 
                              const unsigned int bounds[],
                              unsigned int dim) {
      return stream.fetchElem(pos, bounds, dim);
    }

    // Totally bogus... I know.  But it is to force the 
    // iter stream to create a CPUStreamShadow.
    virtual bool    isCPU() const {return false;}
  };

  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  class CPURuntime: public brook::Runtime {
  public:
    CPURuntime();
    virtual Kernel * CreateKernel(const void*[]);
    virtual Stream * CreateStream(unsigned int fieldCount, 
                                  const StreamType fieldTypes[],
                                  unsigned int dims, 
                                  const unsigned int extents[]);
    virtual Iter * CreateIter(StreamType type, 
                              unsigned int dims, 
                              const unsigned int extents[],
                              const float ranges[]);
    virtual ~CPURuntime(){}
  };

  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  class CPUStreamShadow : public CPUStream {
  public:
    
    CPUStreamShadow (StreamInterface *s,
                     unsigned int flags);
    virtual ~CPUStreamShadow ();
    
  private:
    StreamInterface *shadow;
    unsigned int flags;
  };

}

#endif





