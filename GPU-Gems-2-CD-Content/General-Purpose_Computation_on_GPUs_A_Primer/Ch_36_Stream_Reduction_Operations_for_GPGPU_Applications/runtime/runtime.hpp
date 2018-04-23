// runtime.hpp
#ifndef __RUNTIME_HPP__
#define __RUNTIME_HPP__

#include <brook/brt.hpp>

namespace brook {

  class Runtime {
  public:
    Runtime() {}

    virtual Kernel* CreateKernel(const void*[]) = 0;

    virtual Stream* CreateStream(unsigned int fieldCount, 
                                 const StreamType fieldTypes[],
                                 unsigned int dims, 
                                 const unsigned int extents[]) = 0;

    virtual Iter* CreateIter(StreamType type, 
                             unsigned int dims, 
                             const unsigned int extents[], 
                             const float ranges[])=0;
    virtual ~Runtime() {}

    // TIM: hacky magick for raytracer
    virtual void hackEnableWriteMask() { assert(false); throw 1; }
    virtual void hackDisableWriteMask() { assert(false); throw 1; }
    virtual void hackSetWriteMask( Stream* ) { assert(false); throw 1; }
    virtual void hackBeginWriteQuery() { assert(false); throw 1; }
    virtual int hackEndWriteQuery() { assert(false); throw 1; }

    static Runtime* GetInstance( const char* inRuntimeName = 0, 
                                 void* inContextValue = 0, 
                                 bool addressTranslation = false );

    // TIM: magick to allow re-setting the context after
    // rendering operations
    virtual void hackRestoreContext() { assert(false); throw 1; }

  private:
    static Runtime* CreateInstance( const char* inRuntimeName, 
                                    void* inContextValue, 
                                    bool addressTranslation );
  };
}
#endif

