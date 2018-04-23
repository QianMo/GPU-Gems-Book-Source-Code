// gpukernel.hpp
#ifndef GPU_KERNEL_HPP
#define GPU_KERNEL_HPP

#include "gpubase.hpp"
#include "gpucontext.hpp"

namespace brook {

  class GPURuntime;

  class GPUKernel : public Kernel {
  public:
    typedef GPUContext::TextureHandle TextureHandle;
    typedef GPUContext::PixelShaderHandle PixelShaderHandle;
    typedef GPUContext::VertexShaderHandle VertexShaderHandle;

    static GPUKernel* create( GPURuntime* inRuntime, const void* inSource[] );

    // BRT Interface
    virtual void PushStream( Stream* inStream );
    virtual void PushIter( Iter* inIterator );
    virtual void PushConstant( const float& inValue );  
    virtual void PushConstant( const float2& inValue );  
    virtual void PushConstant( const float3& inValue ); 
    virtual void PushConstant( const float4& inValue );
    virtual void PushGatherStream( Stream* inStream );
    virtual void PushReduce( void* outValue, StreamType inType );
    virtual void PushOutput(Stream *s);
    virtual void Map();
    virtual void Reduce();

  private:
    GPURuntime* _runtime;
    GPUContext* _context;

    GPUKernel( GPURuntime* inRuntime );
    bool initialize( const void* inSource[] );
    bool initialize( const ::brook::desc::gpu_kernel_desc* inDescriptor );
    virtual ~GPUKernel();


    /******************************************************************/
    /**********     Argument Type Classes                **************/

    class ArgumentType
    {
    public:
      virtual TextureHandle getTexture( GPUKernel* inKernel, 
                                        size_t inIndex, 
                                        size_t inComponent );
      virtual void getInterpolant( GPUKernel* inKernel, 
                                   size_t inIndex, 
                                   size_t inComponent,
                                   GPUInterpolant &outInterpolant);
      virtual float4 getConstant( GPUKernel* inKernel, 
                                  size_t inIndex, 
                                  size_t inComponent );
    };

    class StreamArgumentType : public ArgumentType
    {
    public:
      TextureHandle getTexture( GPUKernel* inKernel, 
                                size_t inIndex, 
                                size_t inComponent );
      void getInterpolant( GPUKernel* inKernel, 
                           size_t inIndex, 
                           size_t inComponent,
                           GPUInterpolant &outInterpolant );
      float4 getConstant( GPUKernel* inKernel, 
                          size_t inIndex, 
                          size_t inComponent );
    };
    
    class IteratorArgumentType : public ArgumentType
    {
    public:
      void getInterpolant( GPUKernel* inKernel, 
                           size_t inIndex, 
                           size_t inComponent,
                           GPUInterpolant &outInterpolant );
      float4 getConstant( GPUKernel* inKernel, 
                          size_t inIndex, 
                          size_t inComponent );
    };

    class ConstantArgumentType : public ArgumentType
    {
    public:
      float4 getConstant( GPUKernel* inKernel, 
                          size_t inIndex, 
                          size_t inComponent );
    };

    class GatherArgumentType : public ArgumentType
    {
    public:
      TextureHandle getTexture( GPUKernel* inKernel, 
                                size_t inIndex, 
                                size_t inComponent );
      float4 getConstant( GPUKernel* inKernel, 
                          size_t inIndex, 
                          size_t inComponent );
    };
    
    class OutputArgumentType : public ArgumentType
    {
    public:
      TextureHandle getTexture( GPUKernel* inKernel, 
                                size_t inIndex, 
                                size_t inComponent );
      void getInterpolant( GPUKernel* inKernel, 
                           size_t inIndex, 
                           size_t inComponent,
                           GPUInterpolant &outInterpolant);
      float4 getConstant( GPUKernel* inKernel,
                          size_t inIndex,
                          size_t inComponent );
    };

    class ReduceArgumentType : public ArgumentType
    {
    public:
    };

    //internal classes are not friendly enough
    friend class StreamArgumentType;
    friend class IteratorArgumentType; 
    friend class ConstantArgumentType;
    friend class GatherArgumentType; 
    friend class OutputArgumentType; 
    friend class ReduceArgumentType;

    /***************************************************************/

    class ArgumentInfo
    {
    public:
      ArgumentInfo( ArgumentType* inType, 
                    size_t inIndex )
        : type(inType), index(inIndex)
      {}

      TextureHandle getTexture( GPUKernel* inKernel, 
                                size_t inComponent ) {
        return type->getTexture( inKernel, index, inComponent );
      }

      void getInterpolant( GPUKernel* inKernel, 
                           size_t inComponent,
                           GPUInterpolant &outInterpolant) {
        type->getInterpolant( inKernel, index, inComponent, 
                              outInterpolant );
      }

      float4 getConstant( GPUKernel* inKernel, 
                          size_t inComponent ) {
        return type->getConstant( inKernel, index, inComponent );
      }

      ArgumentType* type;
      size_t index;
    };

    class Input
    {
    public:
      Input()
      {
      }

      Input( const ::brook::desc::gpu_input_desc& inDescriptor )
        : argumentIndex(inDescriptor._argumentIndex), 
          componentIndex(inDescriptor._componentIndex)
      {
      }

      size_t argumentIndex;
      size_t componentIndex;
    };
    typedef std::vector<Input> InputList;


    class Pass
    {
    public:
      void addConstant( const ::brook::desc::gpu_input_desc& inDescriptor ) {
        constants.push_back( Input(inDescriptor) );
      }

      void addSampler( const ::brook::desc::gpu_input_desc& inDescriptor ) {
        samplers.push_back( Input(inDescriptor) );
      }

      void addInterpolant( const ::brook::desc::gpu_input_desc& inDescriptor ) {
        interpolants.push_back( Input(inDescriptor) );
      }

      void addOutput( const ::brook::desc::gpu_input_desc& inDescriptor ) {
        outputs.push_back( Input(inDescriptor) );
      }

      PixelShaderHandle pixelShader;

      InputList constants;
      InputList samplers;
      InputList interpolants;
      InputList outputs;
    };
    typedef std::vector<Pass> PassList;

    class Technique
    {
    public:
      int reductionFactor;
      bool outputAddressTranslation;
      bool inputAddressTranslation;

      PassList passes;
    };

    class ReduceArgumentInfo
    {
    public:
      ReduceArgumentInfo( void* inData, StreamType inType )
        : data(inData), type(inType)
      {}

      void* data;
      StreamType type;
    };

    // The full generality of BrookGPU reductions
    // is overwhelming. The ReductionState structure
    // keeps track of all the buffers and dimensions
    // needed during a multipass 2D reduction.
    class ReductionState
    {
    public:
      TextureHandle inputTexture; // the texture we are reducing from
      TextureHandle outputTexture; // the texture we are reducing to

      int whichBuffer; // -1 for inputTexture, 0,1 for reductionBuffers
      TextureHandle reductionBuffers[2];
      size_t reductionBufferWidths[2];
      size_t reductionBufferHeights[2];
      TextureHandle slopBuffer;
      size_t slopBufferWidth;
      size_t slopBufferHeight;

      size_t currentDimension; // 0 for X, 1 for Y

      // the sizes of the various buffers
      size_t targetExtents[2];
      size_t inputExtents[2];
      size_t currentExtents[2];
      size_t slopCount;

      size_t componentCount; // float1/2/3/4?
    };

    // internal methods
    void pushArgument( ArgumentType* inType, size_t inIndex );
    void clearArguments();
    void clearInputs();
    void executeTechnique( const Technique& inTechnique );
    void executePass( const Pass& inPass );
    void reduceToStream( TextureHandle inOutputBuffer, size_t inExtentX, size_t inExtentY );
    void bindConstant( PixelShaderHandle inPixelShader, size_t inIndex, const Input& inInput );
    void bindSampler( size_t inIndex, const Input& inInput );
    void bindInterpolant( size_t inIndex, const Input& inInput );
    void bindOutput( size_t inIndex, const Input& inInput );

    float4 getGlobalConstant( size_t inComponentIndex );
    TextureHandle getGlobalSampler( size_t inComponentIndex );
    void getGlobalInterpolant( size_t inComponentIndex, GPUInterpolant& outInterpolant );
    TextureHandle getGlobalOutput( size_t inComponentIndex );

    float4 getATIndexofNumerConstant( unsigned int rank, const unsigned int* domainMin, const unsigned int* domainMax );
    float4 getATIndexofDenomConstant( unsigned int rank, const unsigned int* domainMin, const unsigned int* domainMax );

    // multipass reduction helpers
    void executeReductionTechnique( size_t inFactor );
    void beginReduction( ReductionState& ioState );
    void executeReductionStep( ReductionState& ioState );
    void executeSlopStep( ReductionState& ioState );
    void endReduction( ReductionState& ioState );
    void dumpReductionState( ReductionState& ioState );
    void dumpReductionBuffer( TextureHandle inBuffer, size_t inBufferWidth, size_t inBufferHeight, size_t inWidth, size_t inHeight );


    // shared data
    static ArgumentType* sStreamArgumentType;
    static ArgumentType* sIteratorArgumentType;
    static ArgumentType* sConstantArgumentType;
    static ArgumentType* sGatherArgumentType;
    static ArgumentType* sOutputArgumentType;
    static ArgumentType* sReduceArgumentType;

    // instance data
    std::vector<Technique> _techniques;

    typedef std::vector<GPUStream*> StreamList;
    typedef std::vector<GPUIterator*> IteratorList;
    typedef std::vector<float4> ConstantList;
    typedef std::vector<ArgumentInfo> ArgumentList;
    typedef std::vector<ReduceArgumentInfo> ReduceArgumentList;
    typedef std::vector<GPUInterpolant> InterpolantList;
    typedef std::vector<TextureHandle> TextureList;

    StreamList _streamArguments;
    IteratorList _iteratorArguments;
    ConstantList _constantArguments;
    StreamList _gatherArguments;
    StreamList _outputArguments;
    ArgumentList _arguments;
    ReduceArgumentList _reduceArguments;

    // 'global' arguments
    ConstantList _globalConstants;
    TextureList _globalSamplers;
    TextureList _globalOutputs;
    InterpolantList _globalInterpolants;

    unsigned int _outTextureWidth;
    unsigned int _outTextureHeight;
    unsigned int _outReversedExtents[4];

    GPURegion _outputRegion;
    InterpolantList _inputInterpolants;
  
  };
}

#endif
