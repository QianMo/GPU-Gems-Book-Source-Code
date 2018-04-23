#include "gpu.hpp"

namespace brook
{

  // GPUStreamData
  
  GPUStreamData::GPUStreamData( GPURuntime* inRuntime )
    : _referenceCount(1),
      _cpuData(0),
      _cpuDataSize(0),
      _requiresAddressTranslation(false)
  {
    _runtime = inRuntime;
    _context = _runtime->getContext();
  }
  
  GPUStreamData::~GPUStreamData()
  {
    GPUAssert( _referenceCount == 0, "Deleting referenced stream data" );
    
    if( _context )
      {
        size_t fieldCount = _fields.size();
        for( size_t f = 0; f < fieldCount; f++ )
          {
            _context->releaseTexture( _fields[f].texture );
          }
      }
    if( _cpuData )
      {
        delete[] (unsigned char*)_cpuData;
      }
  }

  void GPUStreamData::acquireReference() {
    _referenceCount++;
  }
  
  void GPUStreamData::releaseReference()
  {
    if( --_referenceCount == 0 )
      delete this;
  }
  
  bool GPUStreamData::initialize(unsigned int inFieldCount, 
                                 const StreamType* inFieldTypes,
                                 unsigned int inDimensionCount, 
                                 const unsigned int* inExtents )
  {
    _rank = (unsigned int)inDimensionCount;

    if( _rank == 0 || _rank > 4 )
    {
      GPUWARN << "Unable to create stream with " << _rank << " dimensions.\n"
              << "Dimensions must be between 1 and 4." << std::endl;
      return false;
    }
    if( _rank > 2 )
    {
      GPULOG(1) << "Performance Warning: stream with rank " << _rank
                << " requires address translation." << std::endl;
      _requiresAddressTranslation = true;
    }
    
    _totalSize = 1;
    unsigned int d;
    for( d = 0; d < _rank; d++ )
      {
        unsigned int extent = (unsigned int)inExtents[d];
        if( !_context->isTextureExtentValid( extent )
            && !_requiresAddressTranslation )
          {
            GPULOG(1) << "Performance Warning: stream with extent " << extent
                      << " in dimension " << d
                      << " requires address translation." << std::endl;
            _requiresAddressTranslation = true;
          }
        _extents.push_back(extent);
        _totalSize *= _extents[d];
      }
    
    d = _rank;
    while( d-- > 0 )
      {
        _reversedExtents.push_back(_extents[d]);
      }
    
    if( !_requiresAddressTranslation )
    {
        switch( _rank )
        {
        case 1:
            _textureWidth = _extents[0]>0?_extents[0]:1;
            _textureHeight = 1;
            break;
        case 2:
            _textureWidth = _extents[1]>0?_extents[1]:1;
            _textureHeight = _extents[0]>0?_extents[0]:1;
            break;
        default:
            GPUAssert( false, "Should be unreachable" );
            return false;
        }
    }
    else
    {
        // address-translation - tasty
        bool foundValidShape = false;
        unsigned int bestTextureWidth = 0;
        unsigned int bestTextureHeight = 0;
        float bestAspect = 0;

        unsigned int trialTextureWidth = 2;
        for(; _context->isTextureExtentValid( trialTextureWidth ); trialTextureWidth *= 2 )
        {
            unsigned int neededTextureHeight =
                (_totalSize + (trialTextureWidth-1)) / trialTextureWidth;

            unsigned int trialTextureHeight = 1;
            while( trialTextureHeight < neededTextureHeight )
              trialTextureHeight *= 2;

            if( !_context->isTextureExtentValid( trialTextureHeight ) )
                continue;

            float trialAspect = fabsf( logf(
                (float)(trialTextureWidth) / (float)(trialTextureHeight) ) );

            if( !foundValidShape || trialAspect < bestAspect )
            {
                foundValidShape = true;
                bestTextureWidth = trialTextureWidth;
                bestTextureHeight = trialTextureHeight;
                bestAspect = trialAspect;
            }
        }

        if( !foundValidShape )
        {
            GPUWARN << "Couldn't find valid texture shape to hold"
                << " stream with " << _totalSize << " elements total.";
            return false;
        }

        _textureWidth = bestTextureWidth;
        _textureHeight = bestTextureHeight;
    }
    
    _elementSize = 0;
    for( unsigned int i = 0; i < inFieldCount; i++ )
      {
        StreamType fieldType = inFieldTypes[i];
        TextureFormat fieldTextureFormat;
        size_t fieldComponentCount;
        
        switch (fieldType) {
        case __BRTFLOAT:
          fieldComponentCount=1;
          fieldTextureFormat = GPUContext::kTextureFormat_Float1;
          break;
        case __BRTFLOAT2:
          fieldComponentCount=2;
          fieldTextureFormat = GPUContext::kTextureFormat_Float2;
          break;
        case __BRTFLOAT3:
          fieldComponentCount=3;
          fieldTextureFormat = GPUContext::kTextureFormat_Float3;
          break;
        case __BRTFLOAT4:
          fieldComponentCount=4;
          fieldTextureFormat = GPUContext::kTextureFormat_Float4;
          break;
        case __BRTFIXED:
          fieldComponentCount=1;
          fieldTextureFormat = GPUContext::kTextureFormat_Fixed1;
          break;
        case __BRTFIXED2:
          fieldComponentCount=2;
          fieldTextureFormat = GPUContext::kTextureFormat_Fixed2;
          break;
        case __BRTFIXED3:
          fieldComponentCount=3;
          fieldTextureFormat = GPUContext::kTextureFormat_Fixed3;
          break;
        case __BRTFIXED4:
          fieldComponentCount=4;
          fieldTextureFormat = GPUContext::kTextureFormat_Fixed4;
          break;
        case __BRTHALF:
          fieldComponentCount=1;
          fieldTextureFormat = GPUContext::kTextureFormat_Half1;
          break;
        case __BRTHALF2:
          fieldComponentCount=2;
          fieldTextureFormat = GPUContext::kTextureFormat_Half2;
          break;
        case __BRTHALF3:
          fieldComponentCount=3;
          fieldTextureFormat = GPUContext::kTextureFormat_Half3;
          break;
        case __BRTHALF4:
          fieldComponentCount=4;
          fieldTextureFormat = GPUContext::kTextureFormat_Half4;
          break;
        default:
          GPUWARN << "Invalid element type for stream.\n"
                  << "Only float, float2, float3 and float4 elements are supported." << std::endl;
          return false;
        }
        _elementSize += ::brook::getElementSize(fieldType);
        
        TextureHandle fieldTexture = _context->createTexture2D(_textureWidth, 
                                                               _textureHeight, 
                                                               fieldTextureFormat );
        if( fieldTexture == NULL )
          {
            GPUWARN << "Texture allocation failed during stream initialization.";
            return false;
          }
        
        _fields.push_back( Field(fieldType,
                                 fieldComponentCount,
                                 fieldTexture) );
      }
    
    _indexofConstant = _context->getStreamIndexofConstant( getIndexedFieldTexture( 0 ) );
    
    return true;
  }

  void GPUStreamData::setData( const void* inData )
  {
    unsigned int domainMin[4] = {0,0,0,0};
    const unsigned int* domainMax = getExtents();

    setDomainData( inData, domainMin, domainMax );
  }

  void GPUStreamData::getData( void* outData )
  {
    unsigned int domainMin[4] = {0,0,0,0};
    const unsigned int* domainMax = getExtents();

    getDomainData( outData, domainMin, domainMax );
  }

  void GPUStreamData::setDomainData( const void* inData, const unsigned int* inDomainMin, const unsigned int* inDomainMax )
  {
    // TIM: pain in the ass
    unsigned int domainSize =1;
    for( unsigned int r = 0; r < _rank; r++ )
      domainSize *= (inDomainMax[r] - inDomainMin[r]);

    const unsigned char* data = (const unsigned char*) inData;
    size_t stride = getElementSize();
    size_t fieldCount = _fields.size();
    for( size_t f = 0; f < fieldCount; f++ )
    {
      _context->setTextureData( _fields[f].texture, (const float*)data, stride, domainSize,
        _rank, inDomainMin, inDomainMax, getExtents(), _requiresAddressTranslation );
      data += ::brook::getElementSize(getIndexedFieldType(f));
    }
  }

  void GPUStreamData::getDomainData( void* outData, const unsigned int* inDomainMin, const unsigned int* inDomainMax )
  {
    // TIM: pain in the ass
    unsigned int domainSize =1;
    for( unsigned int r = 0; r < _rank; r++ )
      domainSize *= (inDomainMax[r] - inDomainMin[r]);

    unsigned char* data = (unsigned char*) outData;
    size_t stride = getElementSize();
    size_t fieldCount = _fields.size();
    for( size_t f = 0; f < fieldCount; f++ )
    {
      _context->getTextureData( _fields[f].texture, (float*)data, stride, domainSize,
        _rank, inDomainMin, inDomainMax, getExtents(), _requiresAddressTranslation );
      data += ::brook::getElementSize(getIndexedFieldType(f));
    }
  }

  // TIM: map/unmap are going to be broken when applied
  // to a domain. Thus CPU fallback for domains will
  // be broken

  void* GPUStreamData::map(unsigned int flags)
  {
    if( _cpuData == NULL )
      {
        _cpuDataSize = _totalSize * getElementSize();
        _cpuData = new unsigned char[ _cpuDataSize ];
      }

    if( flags & Stream::READ )
      getData( _cpuData );
    return _cpuData;
  }
  
  
  void GPUStreamData::unmap(unsigned int flags)
  {
    if( flags & Stream::WRITE )
      {
        setData( _cpuData );
      }
  }
  
  void GPUStreamData::getOutputRegion(
                                      const unsigned int* inDomainMin,
                                      const unsigned int* inDomainMax,
                                      GPURegion &outRegion )
  {
    if( _requiresAddressTranslation )
    {
        const unsigned int domainMin[4] = {0,0,0,0};
        const unsigned int domainMax[4] = {_textureHeight,_textureWidth,0,0};
        _context->getStreamOutputRegion( getIndexedFieldTexture( 0 ),
                                        2,
                                        domainMin,
                                        domainMax,
                                        outRegion );
    }
    else
    {
        _context->getStreamOutputRegion(
                                        getIndexedFieldTexture( 0 ),
                                        _rank,
                                        inDomainMin,
                                        inDomainMax,
                                        outRegion );
    }
  }
  
  void GPUStreamData::getStreamInterpolant(
                                           const unsigned int* inDomainMin,
                                           const unsigned int* inDomainMax,
                                           unsigned int inOutputWidth,
                                           unsigned int inOutputHeight,
                                           GPUInterpolant &outInterpolant )
  {
    _context->getStreamInterpolant(
                                   getIndexedFieldTexture( 0 ),
                                   _rank,
                                   inDomainMin,
                                   inDomainMax,
                                   inOutputWidth,
                                   inOutputHeight,
                                   outInterpolant );
    
  }
  
  // GPUStream
  
  GPUStream* GPUStream::create( GPURuntime* inRuntime,
                                unsigned int inFieldCount, 
                                const StreamType* inFieldTypes,
                                unsigned int inDimensionCount, 
                                const unsigned int* inExtents )
  {
    GPUStreamData* data = new GPUStreamData( inRuntime );
    if( data->initialize( inFieldCount, inFieldTypes, inDimensionCount, inExtents ) )
      {
        GPUStream* result = new GPUStream( data );
        data->releaseReference();
        return result;
      }
    data->releaseReference();
    return NULL;
  }
  
  GPUStream::GPUStream( GPUStreamData* inData )
    : _data(inData)
  {
    _data->acquireReference();
    
    unsigned int rank = _data->getRank();
    const unsigned int* extents = _data->getExtents();
    unsigned int r;
    for( r = 0; r < rank; r++ )
      {
        _domainMin[r] = 0;
        _domainMax[r] = extents[r];
      }
    for( ; r < kMaximumRank; r++ )
      {
        _domainMin[r] = 0;
        _domainMax[r] = 1;
      }
  }
  
  GPUStream::GPUStream( GPUStreamData* inData,
                        const unsigned int* inDomainMin,
                        const unsigned int* inDomainMax )
    : _data(inData)
  {
    _data->acquireReference();
    
    unsigned int rank = _data->getRank();
    unsigned int r;
    for( r = 0; r < rank; r++ )
      {
        _domainMin[r] = inDomainMin[r];
        _domainMax[r] = inDomainMax[r];
      }
    for( ; r < kMaximumRank; r++ )
      {
        _domainMin[r] = 0;
        _domainMax[r] = 1;
      }
  }
  
    GPUStream::~GPUStream()
    {
        if( _data )
            _data->releaseReference();
    }

    void GPUStream::getOutputRegion( GPURegion& outRegion )
    {
        _data->getOutputRegion( _domainMin, _domainMax, outRegion );
    }

  void
  GPUStream::getStreamInterpolant (unsigned int _textureWidth,
                                   unsigned int _textureHeight,
                                   GPUInterpolant &_interpolant) {
    
    _data->getStreamInterpolant( _domainMin, _domainMax,
        _textureWidth, _textureHeight, _interpolant );
  }

    float4 GPUStream::getATLinearizeConstant()
    {
        float4 result(0,0,0,0);
        unsigned int rank = getRank();
        const unsigned int* reversedExtents = getReversedExtents();
        unsigned int stride = 1;
        for( unsigned int r = 0; r < rank; r++ )
        {
            ((float*)&result)[r] = (float)(stride);
            stride *= reversedExtents[r];
        }
        return result;
    }

    float4 GPUStream::getATTextureShapeConstant()
    {
        float4 result(0,0,0,0);
        unsigned int textureWidth = getTextureWidth();
        unsigned int textureHeight = getTextureHeight();
        result.x = 1.0f / (float)textureWidth;
        result.y = 1.0f / (float)textureHeight;
        result.z = (float)textureWidth;
        result.w = (float)textureHeight;
        return result;
    }

    float4 GPUStream::getATDomainMinConstant()
    {
        float4 result(0,0,0,0);
        unsigned int rank = getRank();
        const unsigned int* domainMin = getDomainMin();
        for( unsigned int r = 0; r < rank; r++ )
        {
            unsigned int d = rank - (r+1);
            ((float*)&result)[r] = (float)domainMin[d];
        }
        return result;
    }

    float4 GPUStream::getGatherConstant() const
    {
      return _data->_context->getStreamGatherConstant( getRank(), _domainMin, _domainMax, getExtents() );
    }

  Stream* GPUStream::Domain(int inMin, int inMax)
  {
      GPUAssert( getDimension() == 1, "Expected stream with rank of 1" );
      unsigned int newDomainMin[4];
      unsigned int newDomainMax[4];
      newDomainMin[0] = _domainMin[0] + (unsigned int)inMin;
      newDomainMax[0] = _domainMin[0] + (unsigned int)inMax;
      return new GPUStream( _data, newDomainMin, newDomainMax );
  }

  Stream* GPUStream::Domain(const int2& inMin, const int2& inMax)
  {
      GPUAssert( getDimension() == 2, "Expected stream with rank of 2" );
      unsigned int newDomainMin[4];
      unsigned int newDomainMax[4];
      for( int i = 0; i < 2; i++ )
      {
          unsigned int d = 2 - (i+1);
          newDomainMin[i] = _domainMin[i] + (unsigned int)inMin[d];
          newDomainMax[i] = _domainMin[i] + (unsigned int)inMax[d];
      }
      return new GPUStream( _data, newDomainMin, newDomainMax );
  }

  Stream* GPUStream::Domain(const int3& inMin, const int3& inMax)
  {
      GPUAssert( getDimension() == 3, "Expected stream with rank of 3" );
      unsigned int newDomainMin[4];
      unsigned int newDomainMax[4];
      for( int i = 0; i < 3; i++ )
      {
          unsigned int d = 3 - (i+1);
          newDomainMin[i] = _domainMin[i] + (unsigned int)inMin[d];
          newDomainMax[i] = _domainMin[i] + (unsigned int)inMax[d];
      }
      return new GPUStream( _data, newDomainMin, newDomainMax );
  }

  Stream* GPUStream::Domain(const int4& inMin, const int4& inMax)
  {
      GPUAssert( getDimension() == 4, "Expected stream with rank of 4" );
      unsigned int newDomainMin[4];
      unsigned int newDomainMax[4];
      for( int i = 0; i < 4; i++ )
      {
          unsigned int d = 4 - (i+1);
          newDomainMin[i] = _domainMin[i] + (unsigned int)inMin[d];
          newDomainMax[i] = _domainMin[i] + (unsigned int)inMax[d];
      }
      return new GPUStream( _data, newDomainMin, newDomainMax );
  }


  void * 
  GPUStream::getData (unsigned int flags)
  {
    return _data->map( flags );
  }


  void GPUStream::releaseData(unsigned int flags)
  {
    _data->unmap( flags );
  }

  // TIM: hacky magic stuff for rendering
  void* GPUStream::getIndexedFieldRenderData(unsigned int i)
  {
    GPUContext* context = _data->_context;
    TextureHandle texture = getIndexedFieldTexture( i );
    return context->getTextureRenderData( texture );
  }

  void   GPUStream::synchronizeRenderData()
  {
    GPUContext* context = _data->_context;
    unsigned int fieldCount = getFieldCount();
    for( unsigned int f = 0; f < fieldCount; f++ )
    {
      TextureHandle texture = getIndexedFieldTexture( f );
      context->synchronizeTextureRenderData( texture );
    }
  }

}
