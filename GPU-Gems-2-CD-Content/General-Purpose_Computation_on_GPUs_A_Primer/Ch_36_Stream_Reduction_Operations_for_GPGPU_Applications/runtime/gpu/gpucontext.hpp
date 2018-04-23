// gpucontext.hpp
#ifndef GPU_CONTEXT_HPP
#define GPU_CONTEXT_HPP

#include "gpubase.hpp"

namespace brook {

  /*
  This is the abstract interface that each GPU
  implementation platform must target.
  */
  class GPUContext
  {
  public:
    typedef void* TextureHandle;
    typedef void* PixelShaderHandle;
    typedef void* VertexShaderHandle;

    enum TextureFormat
    {
      kTextureFormat_Float1,
      kTextureFormat_Float2,
      kTextureFormat_Float3,
      kTextureFormat_Float4,
      kTextureFormat_Fixed1,
      kTextureFormat_Fixed2,
      kTextureFormat_Fixed3,
      kTextureFormat_Fixed4,
      kTextureFormat_Half1,
      kTextureFormat_Half2,
      kTextureFormat_Half3,
      kTextureFormat_Half4
    };

    virtual int
    getShaderFormatRank (const char *name) const = 0;

    /* Test to see if we can create a texture with
    ** a dimension of this size
    */
    virtual bool 
    isTextureExtentValid( unsigned int inExtent ) const = 0;   
    
    /* Returns the number of shader outputs supported by
    ** the hardware.
    */
    virtual unsigned int 
    getMaximumOutputCount() const { return 1; }

    virtual float4 getStreamIndexofConstant( TextureHandle inTexture ) const = 0;
    virtual float4 getStreamGatherConstant(
      unsigned int inRank, const unsigned int* inDomainMin,
      const unsigned int* inDomainMax, const unsigned int* inExtents ) const = 0;

    virtual float4 getATLinearizeConstant(
        unsigned int texWidth, unsigned int texHeight,
        unsigned int rank, const unsigned int* reversedExtents )
    {
        float4 result = float4(0,0,0,0);
        result.x = (float)reversedExtents[0] / (float)texWidth;
        if( rank > 1 )
            result.y = (float)reversedExtents[1] / (float)texWidth;
        if( rank > 2 )
            result.z = (float)reversedExtents[2] / (float)texWidth;
        if( rank > 3 )
            result.w = (float)reversedExtents[3] / (float)texWidth;
        return result;
    }

    virtual float4 getATReshapeConstant(
        unsigned int texWidth, unsigned int texHeight,
        unsigned int rank, const unsigned int* reversedExtents )
    {
        float4 result = float4(0,0,0,0);
        result.x = 1.0f;
        result.y = 1.0f / (float)texHeight;
        return result;
    }
    
    virtual void
    get1DInterpolant( const float4 &start, 
                      const float4 &end,
                      const unsigned int outputWidth,
                      GPUInterpolant &interpolant) const = 0;

    virtual void 
    get2DInterpolant( const float2 &start, 
                      const float2 &end,
                      const unsigned int outputWidth,
                      const unsigned int outputHeight, 
                      GPUInterpolant &interpolant) const = 0;

    virtual void 
    getStreamInterpolant( const TextureHandle texture,
                          unsigned int rank,
                          const unsigned int* domainMin,
                          const unsigned int* domainMax,
                          const unsigned int outputWidth,
                          const unsigned int outputHeight, 
                          GPUInterpolant &interpolant) const = 0;
    
    virtual void
    getStreamOutputRegion( const TextureHandle texture,
                           unsigned int rank,
                           const unsigned int* domainMin,
                           const unsigned int* domainMax,
                           GPURegion &region) const = 0; 

    // for reductions
    virtual void 
      getStreamReduceInterpolant( const TextureHandle texture,
      const unsigned int outputWidth,
      const unsigned int outputHeight, 
      const unsigned int minX,
      const unsigned int maxX, 
      const unsigned int minY,
      const unsigned int maxY,
      GPUInterpolant &interpolant) const = 0; 

    void getStreamReduceInterpolant( const TextureHandle texture,
      const unsigned int outputWidth,
      const unsigned int outputHeight, 
      const unsigned int minPrimary,
      const unsigned int maxPrimary, 
      const unsigned int minSecondary,
      const unsigned int maxSecondary,
      const unsigned int primaryAxis,
      GPUInterpolant &interpolant) const
    {
      if( primaryAxis == 0 )
        getStreamReduceInterpolant( texture, outputWidth, outputHeight, minPrimary, maxPrimary, minSecondary, maxSecondary, interpolant );
      else
        getStreamReduceInterpolant( texture, outputWidth, outputHeight, minSecondary, maxSecondary, minPrimary, maxPrimary, interpolant );
    }

    virtual void
      getStreamReduceOutputRegion( const TextureHandle texture,
      const unsigned int minX,
      const unsigned int maxX, 
      const unsigned int minY,
      const unsigned int maxY,
      GPURegion &region) const = 0; 

   void getStreamReduceOutputRegion( const TextureHandle texture,
     const unsigned int minPrimary,
     const unsigned int maxPrimary, 
     const unsigned int minSecondary,
     const unsigned int maxSecondary,
     const unsigned int primaryAxis,
      GPURegion &region) const
   {
     if( primaryAxis == 0 )
       getStreamReduceOutputRegion( texture, minPrimary, maxPrimary, minSecondary, maxSecondary, region );
     else
       getStreamReduceOutputRegion( texture, minSecondary, maxSecondary, minPrimary, maxPrimary, region );
   }

    /* Create a texture */
    virtual TextureHandle 
    createTexture2D( unsigned int inWidth, 
                     unsigned int inHeight, 
                     TextureFormat inFormat ) = 0;
    
    /* Free a texture */
    virtual void 
    releaseTexture( TextureHandle inTexture ) = 0;

    /* Load up the texture with data */
    virtual void 
    setTextureData( TextureHandle inTexture, 
                    const float* inData, 
                    unsigned int inStrideBytes, 
                    unsigned int inElemCount,
                    unsigned int inRank,
                    const unsigned int* inDomainMin,
                    const unsigned int* inDomainMax,
                    const unsigned int* inExtents, bool inUsesAddressTranslation ) = 0;
    
    /* Fetch the texture data */
    virtual void 
    getTextureData( TextureHandle inTexture, 
                    float* outData, 
                    unsigned int inStrideBytes, 
                    unsigned int inElemCount,
                    unsigned int inRank,
                    const unsigned int* inDomainMin,
                    const unsigned int* inDomainMax,
                    const unsigned int* inExtents, bool inUsesAddressTranslation ) = 0;
    
    /* Create a pixel shader */
    virtual PixelShaderHandle 
    createPixelShader( const char* inSource ) = 0;

    /* Handles to the passthrough shaders */
    virtual VertexShaderHandle getPassthroughVertexShader() = 0;
    virtual PixelShaderHandle  getPassthroughPixelShader() = 0;

    /* To use around rendering operations */
    virtual void beginScene() = 0;
    virtual void endScene() = 0;

    /* Binding commands */
    virtual void bindConstant( PixelShaderHandle ps, 
                               unsigned int inIndex, 
                               const float4& inValue ) = 0;
    virtual void bindTexture( unsigned int inIndex, 
                              TextureHandle inTexture ) = 0;
    virtual void bindOutput( unsigned int inIndex, 
                             TextureHandle inSurface ) = 0;
    virtual void bindPixelShader( PixelShaderHandle inPixelShader ) = 0;
    virtual void bindVertexShader( VertexShaderHandle inVertexShader ) = 0;
    
    /* Turn off a shader output */
    virtual void disableOutput( unsigned int inIndex ) = 0;
    
    virtual void setOutputDomainMode( bool inUsingOutputDomain ) = 0;
    virtual void setAddressTranslationMode( bool inUsingAddressTranslation ) = 0;

    /* Issue a shader */
    virtual void drawRectangle( const GPURegion &outputRegion, 
                                const GPUInterpolant *interpolants, 
                                unsigned int numInterpolants ) = 0;

    /* hacky functions for rendering - will be deprecated soon */
    virtual void* getTextureRenderData( TextureHandle inTexture ) = 0;
    virtual void synchronizeTextureRenderData( TextureHandle inTexture ) = 0;

    // TIM: hacky magick for raytracer
    virtual void hackEnableWriteMask() { GPUError("unimplemented"); throw 1; }
    virtual void hackDisableWriteMask() { GPUError("unimplemented"); throw 1; }
    virtual void hackSetWriteMask( TextureHandle ) { GPUError("unimplemented"); throw 1; }
    virtual void hackBeginWriteQuery() { GPUError("unimplemented"); throw 1; }
    virtual int hackEndWriteQuery() { GPUError("unimplemented"); throw 1; }

    virtual void hackRestoreContext() { GPUError("unimplemented"); throw 1; }
  };
}

#endif
