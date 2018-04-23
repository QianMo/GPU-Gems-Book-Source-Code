// gpukernel.cpp
#include "gpukernel.hpp"

#include "gpu.hpp"

#include <brook/kerneldesc.hpp>
#include <string>

namespace brook
{
  GPUKernel::ArgumentType* GPUKernel::sStreamArgumentType = new GPUKernel::StreamArgumentType();
  GPUKernel::ArgumentType* GPUKernel::sIteratorArgumentType = new GPUKernel::IteratorArgumentType();
  GPUKernel::ArgumentType* GPUKernel::sConstantArgumentType = new GPUKernel::ConstantArgumentType();
  GPUKernel::ArgumentType* GPUKernel::sGatherArgumentType = new GPUKernel::GatherArgumentType();
  GPUKernel::ArgumentType* GPUKernel::sOutputArgumentType = new GPUKernel::OutputArgumentType();
  GPUKernel::ArgumentType* GPUKernel::sReduceArgumentType = new GPUKernel::ReduceArgumentType();
 
  GPUKernel* GPUKernel::create( GPURuntime* inRuntime, 
                                const void* inSource[] )
  {
    BROOK_PROFILE("GPUKernel::create");
    
    GPUKernel* result = new GPUKernel( inRuntime );
    if( result->initialize( inSource ) )
      return result;
    delete result;
    return NULL;
  }
  
  GPUKernel::GPUKernel( GPURuntime* inRuntime )
    : _runtime(inRuntime), _context(NULL)
  {
    _context = inRuntime->getContext();
  }
  
  GPUKernel::~GPUKernel()
  {
    // TODO: implement
  }
  
  bool GPUKernel::initialize( const void* inSource[] )
  {
    //    clearInputs();
    //    clearArguments();

    int bestRank = -1;
    const ::brook::desc::gpu_kernel_desc* 
          bestDescriptor = NULL;
    
    size_t i = 0;
    while( inSource[i] != NULL )
      {
        const char* nameString = (const char*)inSource[i];
        const ::brook::desc::gpu_kernel_desc* 
          descriptor = (::brook::desc::gpu_kernel_desc*)inSource[i+1];
        
        if( descriptor != NULL
            && nameString != NULL
            && _context->getShaderFormatRank( nameString ) >= 0 )
          {
            int rank = _context->getShaderFormatRank( nameString );
            if( rank > bestRank )
            {
                bestRank = rank;
                bestDescriptor = descriptor;
            }
          }
        
        i += 2;
      }

    if( bestDescriptor != NULL )
        return initialize( bestDescriptor );
    
    GPUWARN << "Unable to find appropriate GPU kernel descriptor.";
    return false;
  }
  
  bool GPUKernel::initialize( const ::brook::desc::gpu_kernel_desc* inDescriptor )
  {
    using namespace ::brook::desc;
    
    _techniques.resize( inDescriptor->_techniques.size() );
    
    std::vector<gpu_technique_desc>::const_iterator 
      ti = inDescriptor->_techniques.begin();
    std::vector<Technique>::iterator 
      tj = _techniques.begin();
    for(; ti != inDescriptor->_techniques.end(); ++ti, ++tj )
      {
        const gpu_technique_desc& inputTechnique = *ti;
        Technique& outputTechnique = *tj;
        
        outputTechnique.reductionFactor = 
          inputTechnique._reductionFactor;
        outputTechnique.outputAddressTranslation = 
          inputTechnique._outputAddressTranslation;
        outputTechnique.inputAddressTranslation = 
          inputTechnique._inputAddressTranslation;
        
        outputTechnique.passes.resize( inputTechnique._passes.size() );
        std::vector<gpu_pass_desc>::const_iterator 
          pi = inputTechnique._passes.begin();
        std::vector<Pass>::iterator 
          pj = outputTechnique.passes.begin();
        for(; pi != inputTechnique._passes.end(); ++pi, ++pj )
          {
            const gpu_pass_desc& inputPass = *pi;
            Pass& outputPass = *pj;
            
            outputPass.pixelShader = 
              _context->createPixelShader( inputPass._shaderString );
            if( outputPass.pixelShader == NULL )
              {
                GPUWARN << "Failed to create a kernel pass pixel shader";
                return false;
              }
            
            std::vector<gpu_input_desc>::const_iterator k;
            for( k = inputPass.constants.begin(); 
                 k != inputPass.constants.end(); k++ )
              outputPass.addConstant( *k );
            for( k = inputPass.samplers.begin(); 
                 k != inputPass.samplers.end(); k++ )
              outputPass.addSampler( *k );
            for( k = inputPass.interpolants.begin(); 
                 k != inputPass.interpolants.end(); k++ )
              outputPass.addInterpolant( *k );
            for( k = inputPass.outputs.begin(); 
                 k != inputPass.outputs.end(); k++ )
              outputPass.addOutput( *k );
          }
      }
    
    // initialize the output rects, just in case
    //    outputRect = DX9Rect(0,0,0,0);
    return true;
  }

  void GPUKernel::PushStream( Stream* inStream )
  {
    GPUStream* stream = (GPUStream*)inStream;
    
    size_t streamArgumentIndex = _streamArguments.size();
    _streamArguments.push_back( stream );
    pushArgument( sStreamArgumentType, streamArgumentIndex );
  }
  
  void GPUKernel::PushIter( Iter* inIterator )
  {
    GPUIterator* iterator = (GPUIterator*)inIterator;
    
    size_t iteratorArgumentIndex = _iteratorArguments.size();
    _iteratorArguments.push_back( iterator );
    pushArgument( sIteratorArgumentType, iteratorArgumentIndex );
  }
  
  void GPUKernel::PushConstant( const float& inValue ) {
    PushConstant( float4( inValue, 0, 0, 0 ) );
  }
  
  void GPUKernel::PushConstant( const float2& inValue ) {
    PushConstant( float4( inValue.x, inValue.y, 0, 0 ) );
  }  
  void GPUKernel::PushConstant( const float3& inValue ) {
    PushConstant( float4( inValue.x, inValue.y, inValue.z, 0 ) );
  }

  void GPUKernel::PushConstant( const float4& inValue )
  {
    size_t constantArgumentIndex = _constantArguments.size();
    _constantArguments.push_back( inValue );
    pushArgument( sConstantArgumentType, constantArgumentIndex );
  }
  
  void GPUKernel::PushGatherStream( Stream* inStream )
  {
    GPUStream* stream = (GPUStream*)inStream;
    
    size_t gatherArgumentIndex = _gatherArguments.size();
    _gatherArguments.push_back( stream );
    pushArgument( sGatherArgumentType, gatherArgumentIndex );
  }

  void GPUKernel::PushReduce( void* outValue, StreamType inType )
  {
    ReduceArgumentInfo argument( outValue, inType );

    size_t reduceArgumentIndex = _reduceArguments.size();
    _reduceArguments.push_back( argument );
    pushArgument( sReduceArgumentType, reduceArgumentIndex );
  }

  void GPUKernel::PushOutput( Stream* inStream )
  {
    GPUStream* stream = (GPUStream*)inStream;
    
    size_t outputArgumentIndex = _outputArguments.size();
    _outputArguments.push_back( stream );
    pushArgument( sOutputArgumentType, outputArgumentIndex );
  }

  void GPUKernel::Map()
  {
    GPUAssert( _outputArguments.size() != 0, 
               "Must have at least some outputs for Map." );

    GPUStream* outputStream = _outputArguments[0];
    outputStream->getOutputRegion(_outputRegion);
    
    _outTextureWidth = outputStream->getTextureWidth();
    _outTextureHeight = outputStream->getTextureHeight();
    unsigned int outRank = outputStream->getRank();
    const unsigned int* outReversed = outputStream->getReversedExtents();
    const unsigned int* outDomainMin = outputStream->getDomainMin();
    const unsigned int* outDomainMax = outputStream->getDomainMax();
    bool usingOutputDomain = false;
    unsigned int r;
    for( r = 0; r < outRank; r++ )
    {
        unsigned int d = outRank - (r+1);
        _outReversedExtents[r] = outDomainMax[d] - outDomainMin[d];
        if( _outReversedExtents[r] != outReversed[r] )
          usingOutputDomain = true;
    }
    for( r = outRank; r < 4; r++ )
        _outReversedExtents[r] = 1;

    bool requiresBaseAddressTranslation = false;
    bool shapeMismatch = false;

    if( outputStream->requiresAddressTranslation() )
        requiresBaseAddressTranslation = true;

    size_t arg;

    size_t streamArgumentCount = _streamArguments.size();
    for( arg = 0; arg < streamArgumentCount; arg++ )
    {
        GPUStream* stream = _streamArguments[arg];
        if( stream->requiresAddressTranslation() )
            requiresBaseAddressTranslation = true;

        unsigned int rank = stream->getRank();
        if( rank != outRank )
			GPUError( "input and output stream ranks do not match" );

        if( !shapeMismatch )
        {
            const unsigned int* domainMin = stream->getDomainMin();
            const unsigned int* domainMax = stream->getDomainMax();

            for( unsigned int r = 0; r < rank; r++ )
            {
				if( domainMin[r] != outDomainMin[r] )
				{
					shapeMismatch = true;
					break;
				}

				if( domainMax[r] != outDomainMax[r] )
				{
					shapeMismatch = true;
					break;
				}
            }
        }
    }
    size_t gatherArgumentCount = _gatherArguments.size();
    for( arg = 0; arg < gatherArgumentCount; arg++ )
    {
        if( _gatherArguments[arg]->requiresAddressTranslation() )
        {
            requiresBaseAddressTranslation = true;
            break;
        }
    }

    bool requiresInputAddressTranslation = requiresBaseAddressTranslation && shapeMismatch;

    // Find and execute an appropriate technique
    Technique* foundTechnique = NULL;
    bool foundTechniqueTrans = false;
    bool foundTechniqueInputTrans = false;
    size_t techniqueCount = _techniques.size();
    for( size_t t = 0; t < techniqueCount && !foundTechnique; t++ )
    {
      Technique& technique = _techniques[t];

      bool techniqueTrans = technique.outputAddressTranslation || technique.inputAddressTranslation;
      bool techniqueInputTrans = technique.inputAddressTranslation;

      if( requiresBaseAddressTranslation && !techniqueTrans )
        continue;

      if( requiresInputAddressTranslation && !techniqueInputTrans )
        continue;

      foundTechnique = & technique;
      foundTechniqueTrans = techniqueTrans;
      foundTechniqueInputTrans = techniqueInputTrans;
    }
    if( !foundTechnique )
      GPUAssert( false, "No appropriate map technique found" );


    if( foundTechniqueTrans )
    {
        // set up additional "global" constants/interpolants

        float4 outputLinearize = float4(0,0,0,0);
        float4 outputStride = float4(0,0,0,0);
        float4 outputInvStride = float4(0,0,0,0);
        float4 outputInvExtent = float4(0,0,0,0);
        float4 outputDomainMin = float4(0,0,0,0);
        float4 outputDomainSize = float4(0.5,0.5,0.5,0.5);
        float4 outputInvShape = float4(0,0,0,0);

        outputLinearize.x = 1.0f;
        outputLinearize.y = (float)_outTextureWidth;

        unsigned int strides[4] = {0,0,0,0};
        unsigned int extents[4] = {0,0,0,0};
        unsigned int s = 1;
        for( unsigned int r = 0; r < outRank; r++ )
        {
            extents[r] = s;
            s *= outReversed[r];
            strides[r] = s;
        }
        for( unsigned int r = 0; r < outRank; r++ )
        {
            ((float*)&outputStride)[r] = (float) strides[r];
            ((float*)&outputInvStride)[r] = 1.0f / (float) strides[r];
            ((float*)&outputInvExtent)[r] = 1.0f / (float) extents[r];

            // must reversed the domain ranges since they are in [w][z][y][x] order
            unsigned int domainMin = outDomainMin[ outRank - (r+1) ];
            unsigned int domainMax = outDomainMax[ outRank - (r+1) ];

            ((float*)&outputDomainMin)[r] = (float)domainMin;
            ((float*)&outputDomainSize)[r] = (float)(domainMax - domainMin) - 0.5f;

            ((float*)&outputInvShape)[r] = 1.0f / (float) outReversed[r];
        }

        _globalConstants.push_back( outputLinearize );
        _globalConstants.push_back( outputStride );
        _globalConstants.push_back( outputInvStride );
        _globalConstants.push_back( outputInvExtent );
        _globalConstants.push_back( outputDomainMin );
        _globalConstants.push_back( outputDomainSize );
        _globalConstants.push_back( outputInvShape );

        float4 hackConstant = float4(1,1,1,1);
        _globalConstants.push_back( hackConstant );

        GPUInterpolant outputInterpolant;
		unsigned int fakeDomainMin[2] = { 0, 0 };
		unsigned int fakeDomainMax[2] = { _outTextureHeight, _outTextureWidth };
		_context->getStreamInterpolant( outputStream->getIndexedFieldTexture(0),
			2, fakeDomainMin, fakeDomainMax, _outTextureWidth, _outTextureHeight, outputInterpolant );
        _globalInterpolants.push_back( outputInterpolant );

        float2 addressStart = float2(0.5f,0.5f);
        float2 addressEnd = float2((float)_outTextureWidth+0.5f,(float)_outTextureHeight+0.5f);
        GPUInterpolant addressInterpolant;
        _context->get2DInterpolant( addressStart, addressEnd, _outTextureWidth, _outTextureHeight, addressInterpolant );
        _globalInterpolants.push_back( addressInterpolant );
    }
    
    // Check to see if the output size matches the input size

#if 0
    size_t streamArgumentIndex = _streamArguments.size();
    _streamArguments.push_back( stream );
    pushArgument( sStreamArgumentType, streamArgumentIndex );
#endif

    _context->setOutputDomainMode( usingOutputDomain );
    _context->setAddressTranslationMode( foundTechniqueTrans || foundTechniqueInputTrans );

    _context->beginScene();
    
    executeTechnique( *foundTechnique );
    
    _context->endScene();
    
    //    size_t outputStreamCount = _outputStreams.size();
    //    for( size_t o = 0; o < outputStreamCount; o++ )
    //      outputStreams[o]->markGPUDataChanged();
    
    clearArguments();
  }
  
  void GPUKernel::Reduce()
  {
    GPUAssert( _reduceArguments.size() == 1,
      "Must have one and only one reduction output." );

    ReduceArgumentInfo reduceArgument = _reduceArguments[0];
    StreamType outputReductionType = reduceArgument.type;
    void* outputReductionData = reduceArgument.data;

    if( outputReductionType == __BRTSTREAM )
    {
      Stream* outputStreamBase = *((const ::brook::stream*)outputReductionData);
      GPUStream* outputStream = (GPUStream*)outputStreamBase;
      GPUAssert( outputStream->getFieldCount() == 1,
        "Reductions to streams of structure type is currently unsupported.");

      TextureHandle outputTexture = outputStream->getIndexedFieldTexture(0);
      reduceToStream( outputTexture, outputStream->getTextureWidth(), outputStream->getTextureHeight() );
    }
    else
    {
      size_t componentCount = 0;
      switch( outputReductionType )
      {
      case __BRTFLOAT:
          componentCount = 1;
          break;
      case __BRTFLOAT2:
          componentCount = 2;
          break;
      case __BRTFLOAT3:
          componentCount = 3;
          break;
      case __BRTFLOAT4:
          componentCount = 4;
          break;
      default:
          GPUError("reduce output type isn't float1/2/3/4");
          break;
      }

      TextureHandle outputTexture = _runtime->getReductionTargetBuffer( componentCount );
      reduceToStream( outputTexture, 1, 1 );

      float4 reductionResult;
      unsigned int domainMin = 0;
      unsigned int domainMax = 1;
      unsigned int extents = 1;
      _context->getTextureData( outputTexture, (float*)&reductionResult, sizeof(reductionResult), 1, 1, &domainMin, &domainMax, &extents, false );
      if( outputReductionType == __BRTFLOAT )
        *((float*)outputReductionData) = *((float*)&reductionResult);
      else if( outputReductionType == __BRTFLOAT2 )
        *((float2*)outputReductionData) = *((float2*)&reductionResult);
      else if( outputReductionType == __BRTFLOAT3 )
        *((float3*)outputReductionData) = *((float3*)&reductionResult);
      else if( outputReductionType == __BRTFLOAT4 )
        *((float4*)outputReductionData) = *((float4*)&reductionResult);
      else
      {
        GPUError("Invalid reduction target type.\n"
          "Only float, float2, float3, and float4 outputs allowed.");
      }
    }
  }
  
  /// Internal methods
  void GPUKernel::pushArgument( ArgumentType* inType, size_t inIndex )
  {
    _arguments.push_back( ArgumentInfo( inType, inIndex ) );
  }
  
  void GPUKernel::clearArguments()
  {
    _streamArguments.clear();
    _iteratorArguments.clear();
    _constantArguments.clear();
    _gatherArguments.clear();
    _outputArguments.clear();
    _reduceArguments.clear();
    _arguments.clear();

    _globalConstants.clear();
    _globalSamplers.clear();
    _globalOutputs.clear();
    _globalInterpolants.clear();
  }
  
  void GPUKernel::clearInputs()
  {
    _inputInterpolants.clear();
  }

  void GPUKernel::executeTechnique( const Technique& inTechnique )
  {
    PassList::const_iterator i;
    for( i = inTechnique.passes.begin(); i != inTechnique.passes.end(); i++ )
      {
        executePass( *i );
        //      clearInputs();
      }
  }
  
  void GPUKernel::executePass( const Pass& inPass )
  {
    PixelShaderHandle pixelShader = inPass.pixelShader;
    VertexShaderHandle vertexShader = _context->getPassthroughVertexShader();

    // Bind all the arguments for this pass
    size_t i;
    
    size_t constantCount = inPass.constants.size();
    for( i = 0; i < constantCount; i++ )
      bindConstant( pixelShader, i, inPass.constants[i] );
    
    size_t samplerCount = inPass.samplers.size();
    for( i = 0; i < samplerCount; i++ )
      bindSampler( i, inPass.samplers[i] );
    
    size_t maximumOutputCount = _context->getMaximumOutputCount();
    for( i = 1; i < maximumOutputCount; i++ )
      _context->disableOutput( i );

	size_t outputCount = inPass.outputs.size();
    for( i = 0; i < outputCount; i++ )
      bindOutput( i, inPass.outputs[i] );

    size_t interpolantCount = inPass.interpolants.size();
    _inputInterpolants.resize( interpolantCount );
    for( i = 0; i < interpolantCount; i++ )
      bindInterpolant( i, inPass.interpolants[i] );
    
    // Execute
    _context->bindVertexShader( vertexShader );
    _context->bindPixelShader( pixelShader );
    
    _context->drawRectangle( _outputRegion, 
                             &(_inputInterpolants[0]), 
                             _inputInterpolants.size() );
    clearInputs();
  }

  void GPUKernel::reduceToStream( TextureHandle inOutputBuffer, size_t inExtentX, size_t inExtentY )
  {
    TextureHandle outputBuffer = inOutputBuffer;
    size_t outputWidth = inExtentX;
    size_t outputHeight = inExtentY;


    GPUAssert( _streamArguments.size() == 1,
      "Reductions must have one and only one input stream." );

    GPUStream* inputStream = _streamArguments[0];
    size_t inputWidth = inputStream->getTextureWidth();
    size_t inputHeight = inputStream->getTextureHeight();

    GPUAssert( inputStream->getFieldCount() == 1,
      "Reductions from structures are not currently supported." );

    size_t xFactor = inputWidth / outputWidth;
    size_t yFactor = inputHeight / outputHeight;

    GPUAssert( inputWidth % outputWidth == 0,
      "Reduction output width is not an integer divisor of input width" );
    GPUAssert( inputHeight % outputHeight == 0,
      "Reduction output height is not an integer divisor of input height" );

    // we try to reduce in whatever direction
    // has the greater factor first
    // so that we can hopefully reduce
    // the size of the buffers allocated
    size_t firstDimension = 0;
    size_t secondDimension = 1;
    if( yFactor > xFactor )
    {
      firstDimension = 1;
      secondDimension = 0;
    }

    ReductionState state;
    state.inputTexture = inputStream->getIndexedFieldTexture(0);
    state.outputTexture = outputBuffer;
    state.whichBuffer = -1; // data starts in the input
    state.reductionBuffers[0] = NULL;
    state.reductionBuffers[1] = NULL;
    state.reductionBufferWidths[0] = 0;
    state.reductionBufferWidths[1] = 0;
    state.reductionBufferHeights[0] = 0;
    state.reductionBufferHeights[1] = 0;
    state.slopBuffer = NULL;
    state.slopBufferWidth = 0;
    state.slopBufferHeight = 0;
    state.currentDimension = firstDimension;
    state.targetExtents[0] = outputWidth;
    state.targetExtents[1] = outputHeight;
    state.inputExtents[0] = inputWidth;
    state.inputExtents[1] = inputHeight;
    state.currentExtents[0] = inputWidth;
    state.currentExtents[1] = inputHeight;
    state.slopCount = 0;

    StreamType fieldType = inputStream->getIndexedFieldType(0);
    size_t componentCount = 0;
    switch( fieldType )
    {
    case __BRTFLOAT:
        componentCount = 1;
        break;
    case __BRTFLOAT2:
        componentCount = 2;
        break;
    case __BRTFLOAT3:
        componentCount = 3;
        break;
    case __BRTFLOAT4:
        componentCount = 4;
        break;
    default:
        GPUError("cannot reduce non float1/2/3/4 stream");
        break;
    }
    state.componentCount = componentCount;

    beginReduction( state );

    // execute reduction passes in the first dimension
    // until the stream is the proper size
    while( state.currentExtents[firstDimension] != state.targetExtents[firstDimension] )
      executeReductionStep( state );
    executeSlopStep( state );

    // now repeat in the second dimension
    state.currentDimension = secondDimension;
    while( state.currentExtents[secondDimension] != state.targetExtents[secondDimension] )
      executeReductionStep( state );
    executeSlopStep( state );

    endReduction( state );
  }

  void GPUKernel::executeReductionTechnique( size_t inFactor )
  {
    // TIM: We currently make the very strong
    // assumption that reduction techniques
    // will be stored sequentially in our
    // technique array starting with the 2-way
    // reduction technique...

    GPUAssert( inFactor >= 2, "Attempt to reduce by a factor of less than 2" );
    size_t techniqueIndex = inFactor - 2;
    GPUAssert( techniqueIndex < _techniques.size(), "Attempt to reduce by too large of a factor" );

    _context->setOutputDomainMode( false );
    _context->setAddressTranslationMode( false );

    // we use the standard technique-mapping code to make things easier...
    executeTechnique( _techniques[techniqueIndex] );
  }

  void GPUKernel::beginReduction( ReductionState& ioState )
  {
    // TIM: this routine used to do a lot more work
    // (like copying data into reduction buffers
    // or validating input texture data)
    _context->beginScene();

#ifdef BROOK_GPU_ENABLE_REDUCTION_LOG
    dumpReductionState( ioState );
#endif
  }

  void GPUKernel::executeReductionStep( ReductionState& ioState )
  {
    // TIM: this is the meat of the reduction implementation
    // and it is really ugly, gone-off meat...

    // read state values into temporaries
    // so that our code can be less ridiculously verbose
    size_t dim = ioState.currentDimension; // the dimension we are reducing
    size_t remainingExtent = ioState.currentExtents[dim]; // how big the to-be-reduced buffer is
    size_t outputExtent = ioState.targetExtents[dim]; // how big we want it to be
    size_t remainingFactor = remainingExtent / outputExtent; // how much is left to reduce by
    size_t otherExtent = ioState.currentExtents[1-dim]; // how big the other dimension is...
    size_t reductionComponents = ioState.componentCount; // float1/2/3/4?

    // First we must find an appropriate technqiue
    // execute. We will always try to find one
    // that is a perfect divisor of the reduction
    // factor, and failing that we chose the largest
    // applicable factor
    size_t bestTechnique = 0; // base 2-fold reduction should always work
    size_t t = _techniques.size();
    while( t-- > 0 )
    {
      Technique& technique = _techniques[t];
      size_t passFactor = technique.reductionFactor;

      size_t quotient = remainingFactor / passFactor;
      size_t remainder = remainingFactor % passFactor;

      // The logic used here bears explaining. Effectively
      // we have an input buffer consisting of groups
      // of stream elements of size <remainingFactor>
      // each of which will become a single element
      // of the reduced stream

      // as an example, imagine the sum reduction of:
      // 0 1 2 3 4 5 6 7 8
      // to
      // 3 12 21
      // In this case <remainingFactor> is 3 and we
      // can imagine the input grouped as:
      // [0 1 2] [3 4 5] [6 7 8]

      // Now suppose we have techniques that can
      // reduce any span of <passFactor> consecutive values
      // into a single value for 2 <= passFactor <= N
      // How do we know if we can apply the
      // technique with factor passFactor when we have
      // a remaining factor of remainingFactor?

      // There is one very obvious failure case:
      // if the passFactor is greater than the remainingFactor
      // then there isn't going to be enough data
      // to run that technique
      if( quotient == 0 ) continue; // the factor is larger than the data

      // There are four obvious success cases:
      // 1 - passFactor is a perfect divisor of remainingFactor.
      //    clearly in this case we just set up interlaced
      //    texture coordinates and go.
      // 2 - passFactor only divides into the remainingFactor once.
      //    in this case we are only reducing the 'left' side of
      //    the buffer and just need reduce the 'slop' on the
      //    right by some factor P < passFactor
      // 3 - remainingFactor == remainingExtent
      //    in this case we are reducing the whole buffer to
      //    a single value, so we can just reduce the leftmost
      //    stuff and then deal with the slop on the right.
      // 4 - the 'slop' per group of elements is 1
      //    building a texcoord interpolant to skip one texel
      //    every N pixels is fairly doable. For example if
      //    remainingFactor is 5 and passFactor is 2 we need
      //    texcoords "A" and "B" to sample as follows:
      //       A0  B0  A1  B1      A2  B2  A3  B3
      //       0   1   2   3   4   5   6   7   8   9
      //    we can achieve this by setting:
      //       A[n] = floor( n*( 5 / 2 ) )
      //       B[n] = floor( 1 + n*( 5/ 2 ) )
      //    where the 'floor' operation is provided by nearest-
      //    neighbor texturefiltering and the constant spacing
      //    between texcoords is provided by the linear interpolation

      // we currently only deal with cases 1, 2 and 4
      // so our N-to-1 reductions may not be as aggresive
      // as possible.

      if( remainder == 0 )
      {
          // perfect divisors are better than
          // all other options
          bestTechnique = t;
          break;
      }

      if( quotient == 1 || remainder <= 1 )
      {
          if( bestTechnique < t )
              bestTechnique = t;
      }
    }

    // pull information out of the chosen technique
    Technique& technique = _techniques[ bestTechnique ];
    size_t reductionFactor = technique.reductionFactor;
    size_t slopFactor = (remainingFactor % reductionFactor);

    GPULOG(3) << "reduction factor: " << reductionFactor << "slop factor: " << slopFactor << std::endl;

    // calculate the new size of the result buffer
    size_t resultExtents[2];
    resultExtents[0] = ioState.currentExtents[0];
    resultExtents[1] = ioState.currentExtents[1];
    resultExtents[dim] = outputExtent * (remainingFactor / reductionFactor);

    TextureHandle slopBuffer = ioState.slopBuffer;
    TextureHandle inputBuffer = NULL;
    if( ioState.whichBuffer == -1 )
      inputBuffer = ioState.inputTexture; // this the first pass, the data is still in the input
    else
      inputBuffer = ioState.reductionBuffers[ioState.whichBuffer];

    // nextBuffer is where we will be placing the data
    size_t nextBuffer = (ioState.whichBuffer + 1) % 2;
    TextureHandle outputBuffer = ioState.reductionBuffers[nextBuffer];
    size_t outputWidth = ioState.reductionBufferWidths[nextBuffer];
    size_t outputHeight = ioState.reductionBufferHeights[nextBuffer];
    if( outputBuffer == NULL )
    {
      size_t ignoreComponents;
      outputBuffer = _runtime->getReductionTempBuffer(
        kGPUReductionTempBuffer_Swap0 + nextBuffer,
        resultExtents[0], resultExtents[1], reductionComponents,
        &outputWidth, &outputHeight, &ignoreComponents );
      ioState.reductionBuffers[nextBuffer] = outputBuffer;
      ioState.reductionBufferWidths[nextBuffer] = outputWidth;
      ioState.reductionBufferHeights[nextBuffer] = outputHeight;
    }

    // The crazy argument-annotation magic in the pass descriptors
    // will look for reduction-related data in the "global"
    // part of things. We therefore set up the global arguments
    // as needed...

    _globalSamplers.resize(2);
    _globalSamplers[0] = inputBuffer;
    _globalSamplers[1] = inputBuffer;

    _globalOutputs.resize(1);
    _globalOutputs[0] = outputBuffer;

    _globalInterpolants.resize( reductionFactor );
    for( size_t i = 0; i < reductionFactor; i++ )
    {
      _context->getStreamReduceInterpolant( inputBuffer, resultExtents[0], resultExtents[1],
        i, remainingExtent+i, 0, otherExtent, dim, _globalInterpolants[i] );
    }
    size_t newExtent = resultExtents[dim];
    ioState.currentExtents[dim] = newExtent;

    _context->getStreamReduceOutputRegion( outputBuffer, 0, newExtent, 0, otherExtent, dim, _outputRegion );

    // use the existing map functionality to execute the pass/passes...
    executeTechnique( technique );

    // move any slop out to the slop buffer
    if( slopFactor )
    {
      size_t slopWidth = ioState.slopBufferWidth;
      size_t slopHeight = ioState.slopBufferHeight;
      size_t slopExtents[2];
      if( slopBuffer == NULL )
      {
        slopExtents[dim] = outputExtent;
        slopExtents[1-dim] = otherExtent;

        size_t ignoreComponents;
        slopBuffer = _runtime->getReductionTempBuffer( kGPUReductionTempBuffer_Slop,
          slopExtents[0], slopExtents[1], reductionComponents, &slopWidth, &slopHeight, &ignoreComponents );
        ioState.slopBuffer = slopBuffer;
        ioState.slopBufferWidth = slopWidth;
        ioState.slopBufferHeight = slopHeight;
      }
      slopExtents[0] = resultExtents[0];
      slopExtents[1] = resultExtents[1];
      slopExtents[dim] = outputExtent;

      if( ioState.slopCount == 0 && slopFactor == 1 )
      {
        // there is no existing slop data, and the
        // "reduction" factor for the new slop data
        // is one, so we can just copy it...

        _context->bindPixelShader( _context->getPassthroughPixelShader() );
        _context->bindTexture( 0, inputBuffer );

        size_t offset = remainingFactor-1;
        GPUInterpolant interpolant;
        _context->getStreamReduceInterpolant( inputBuffer, slopExtents[0], slopExtents[1],
          offset, remainingFactor+offset, 0, otherExtent, dim, interpolant );
        _inputInterpolants.push_back( interpolant );

        _context->bindOutput( 0, slopBuffer );
        
        _context->getStreamReduceOutputRegion( slopBuffer, 0, outputExtent, 0, otherExtent, dim, _outputRegion );

        _context->drawRectangle( _outputRegion, &_inputInterpolants[0], _inputInterpolants.size() );

        clearInputs();
        ioState.slopCount++;
      }
      else if( ioState.slopCount == 0 )
      {
        // there is no existing slop data,
        // but we have to reduce the new data
        // by the slopFactor to make it fit

        _globalSamplers[0] = inputBuffer;
        _globalSamplers[1] = inputBuffer;
        _globalOutputs[0] = slopBuffer;

        for( size_t i = 0; i < slopFactor; i++ )
        {
          size_t offset = slopFactor - i;
          offset = remainingFactor - offset;

          _context->getStreamReduceInterpolant( inputBuffer, slopExtents[0], slopExtents[1],
            offset, remainingExtent+offset, 0, otherExtent, dim, _globalInterpolants[i] );
        }
        _context->getStreamReduceOutputRegion( slopBuffer, 0, outputExtent, 0, otherExtent, dim, _outputRegion );

        executeReductionTechnique( slopFactor );
        ioState.slopCount++;
      }
      else
      {
        // there is already data in the
        // slop buffer, so we'll need
        // to combine one value from
        // the old slop buffer with
        // one or more from the new data

        _globalSamplers[0] = inputBuffer;
        _globalSamplers[1] = slopBuffer;
        _globalOutputs[0] = slopBuffer;

        for( size_t i = 0; i < slopFactor; i++ )
        {
          size_t offset = slopFactor - i;
          offset = remainingFactor - offset;
          _context->getStreamReduceInterpolant( inputBuffer, slopExtents[0], slopExtents[1],
            offset, remainingExtent+offset, 0, otherExtent, dim, _globalInterpolants[i] );
        }
        _context->getStreamReduceInterpolant( inputBuffer, slopExtents[0], slopExtents[1],
          0, outputExtent, 0, otherExtent, dim, _globalInterpolants[slopFactor] );
        _context->getStreamReduceOutputRegion( slopBuffer, 0, outputExtent, 0, otherExtent, dim, _outputRegion );

        executeReductionTechnique( slopFactor+1 );
        ioState.slopCount++;
      }
    }

    ioState.whichBuffer = nextBuffer;

#ifdef BROOK_GPU_ENABLE_REDUCTION_LOG
    dumpReductionState( ioState );
#endif
  }


  void GPUKernel::executeSlopStep( ReductionState& ioState )
  {
    // we have finished reducing the "bulk"
    // of the data in a given dimension
    // and now we just need to composite
    // the remaining output-sized
    // slop buffer over our results...

    if( ioState.slopCount == 0 ) return;

    size_t dim = ioState.currentDimension;
    size_t outputWidth = ioState.currentExtents[0];
    size_t outputHeight = ioState.currentExtents[1];
    size_t outputExtent = ioState.currentExtents[dim];
    size_t otherExtent = ioState.currentExtents[1-dim];

    TextureHandle slopBuffer = ioState.slopBuffer;
    TextureHandle inputBuffer = ioState.reductionBuffers[ioState.whichBuffer];

    // TIM: we are using the destination buffer both as an input and an output
    // for simplicity. This is not necesarily future proof, and is probably
    // avoidable. It's the only place in the reduction code where we
    // perpetrate such a hack...
    TextureHandle outputBuffer = inputBuffer;

    _globalSamplers[0] = inputBuffer;
    _globalSamplers[1] = slopBuffer;
    _globalOutputs[0] = outputBuffer;

    _context->getStreamReduceInterpolant( inputBuffer, outputWidth, outputHeight,
      0, outputExtent, 0, otherExtent, dim, _globalInterpolants[0] );
    _context->getStreamReduceInterpolant( slopBuffer, outputWidth, outputHeight,
      0, outputExtent, 0, otherExtent, dim, _globalInterpolants[1] );
    _context->getStreamReduceOutputRegion( outputBuffer, 0, outputExtent, 0, otherExtent, dim, _outputRegion );

    // execute the 2-argument reduction technique
    executeReductionTechnique( 2 );

    ioState.slopCount = 0;

#ifdef BROOK_GPU_ENABLE_REDUCTION_LOG
    dumpReductionState( ioState );
#endif
  }

  void GPUKernel::endReduction( ReductionState& ioState )
  {
    size_t outputWidth = ioState.targetExtents[0];
    size_t outputHeight = ioState.targetExtents[1];

    TextureHandle inputBuffer;
    if( ioState.whichBuffer == -1 )
      inputBuffer = ioState.inputTexture; // this should only happen if they didn't actually reduce things
    else
      inputBuffer = ioState.reductionBuffers[ioState.whichBuffer];

    TextureHandle outputBuffer = ioState.outputTexture;

    _context->bindPixelShader( _context->getPassthroughPixelShader() );
    _context->bindTexture( 0, inputBuffer );
    _context->bindOutput( 0, outputBuffer );

    GPUInterpolant interpolant;
    _context->getStreamReduceInterpolant( inputBuffer,
      outputWidth, outputHeight, 0, outputWidth, 0, outputHeight, interpolant );
    _inputInterpolants.push_back( interpolant );

    _context->getStreamReduceOutputRegion( outputBuffer, 0, outputWidth, 0, outputHeight, _outputRegion );

    _context->drawRectangle( _outputRegion, &_inputInterpolants[0], _inputInterpolants.size() );
    clearInputs();

    // final cleanup
    _context->endScene();

    // TIM: used to have to deal with flushing caches here
    // but I think the DX9 context no longer needs it

#ifdef BROOK_GPU_ENABLE_REDUCTION_LOG
    GPULOG(3) << "************ Result *************";
    dumpReductionBuffer( outputBuffer, 1, 1, 1, 1 );
#endif

    clearArguments();
  }

  void GPUKernel::dumpReductionState( ReductionState& ioState )
  {
    GPULOG(3) << "********************* Reduction Dump *************";
    size_t dim = ioState.currentDimension;
    int buffer = ioState.whichBuffer;

    if( buffer == -1 )
    {
      GPULOG(3) << "Input";
      dumpReductionBuffer( ioState.inputTexture,
        ioState.inputExtents[0], ioState.inputExtents[1],
        ioState.currentExtents[0], ioState.currentExtents[1] );
    }
    else
    {
      GPULOG(3) << "Buffer";
//      ioState.reductionBuffers[buffer]->markCachedDataChanged();
      dumpReductionBuffer( ioState.reductionBuffers[buffer],
        ioState.reductionBufferWidths[buffer], ioState.reductionBufferHeights[buffer],
        ioState.currentExtents[0], ioState.currentExtents[1] );
    }

    if( ioState.slopCount )
    {
      int slopExtents[2];
      slopExtents[0] = ioState.currentExtents[0];
      slopExtents[1] = ioState.currentExtents[1];
      slopExtents[dim] = ioState.targetExtents[dim];

      GPULOG(3) << "Slop";
//      ioState.slopBuffer->markCachedDataChanged();
      dumpReductionBuffer( ioState.slopBuffer, ioState.slopBufferWidth, ioState.slopBufferHeight, slopExtents[0], slopExtents[1] );
    }
  }

  void GPUKernel::dumpReductionBuffer( TextureHandle inBuffer, size_t inBufferWidth, size_t inBufferHeight, size_t inWidth, size_t inHeight )
  {
    static float4* data = new float4[2048*2048];

    int bufferWidth = inBufferWidth;
    int bufferHeight = inBufferHeight;

    int w = inWidth;
    int h = inHeight;
    if( w == 0 )
      w = bufferWidth;
    if( h == 0 )
      h = bufferHeight;

    unsigned int domainMin[2] = { 0, 0 };
    unsigned int domainMax[2] = { bufferHeight, bufferWidth };
    unsigned int extents[2] = { bufferHeight, bufferWidth };
    _context->getTextureData( inBuffer, (float*)data, sizeof(float4), bufferWidth*bufferHeight, 2, domainMin, domainMax, extents, false );

    float4* line = data;
    for( int y = 0; y < h; y++ )
    {
      float4* pixel = line;
      for( int x = 0; x < w; x++ )
      {
        if( x > 0 && x % 5 == 0 )
          GPULOGPRINT(3) << "\n\t";

        float4 value = *pixel++;
        GPULOGPRINT(3) << "{" << value.x
          << " " << value.y
          << " " << value.z
          << " " << value.w << "}";
      }
      line += bufferWidth;
      GPULOGPRINT(3) << std::endl;
    }
  }

  void GPUKernel::bindConstant( PixelShaderHandle inPixelShader, 
                                size_t inIndex, const Input& inInput )
  {
    if( inInput.argumentIndex > 0 )
    {
      int arg = inInput.argumentIndex-1;
      ArgumentInfo& argument = _arguments[ arg ];
      _context->bindConstant( inPixelShader, inIndex, 
                              argument.getConstant( this, 
                                                    inInput.componentIndex ) );
    }
    else
    {
      // global constant
      _context->bindConstant( inPixelShader, inIndex, getGlobalConstant( inInput.componentIndex ) );
    }
  }

  void GPUKernel::bindSampler( size_t inIndex, const Input& inInput )
  {
    if( inInput.argumentIndex > 0 )
    {
      int arg = inInput.argumentIndex-1;
      ArgumentInfo& argument = _arguments[ arg ];
      _context->bindTexture( inIndex, 
                             argument.getTexture( this, 
                                                  inInput.componentIndex ) );
    }
    else
    {
      // global sampler
      _context->bindTexture( inIndex, getGlobalSampler( inInput.componentIndex ) );
    }
  }

  void GPUKernel::bindInterpolant( size_t inIndex, 
                                   const Input& inInput )
  {
    if( inInput.argumentIndex > 0 )
    {
      int arg = inInput.argumentIndex-1;
      ArgumentInfo& argument = _arguments[ arg ];
      argument.getInterpolant( this, 
                               inInput.componentIndex,
                               _inputInterpolants[inIndex] );
    }
    else
    {
      // global interpolant
      getGlobalInterpolant( inInput.componentIndex, _inputInterpolants[inIndex] );
    }
  }

  void GPUKernel::bindOutput( size_t inIndex, const Input& inInput )
  {
    if( inInput.argumentIndex > 0 )
    {
      int arg = inInput.argumentIndex-1;
      ArgumentInfo& argument = _arguments[ arg ];
      _context->bindOutput( inIndex, 
                            argument.getTexture( this, 
                                                 inInput.componentIndex ) );
    }
    else
    {
      // global output
      _context->bindOutput( inIndex, getGlobalOutput( inInput.componentIndex ) );
    }
  }

  float4 GPUKernel::getGlobalConstant( size_t inComponentIndex )
  {
    GPUAssert( inComponentIndex < _globalConstants.size(), "Invalid global constant index" );
    return _globalConstants[inComponentIndex];
  }

  GPUKernel::TextureHandle GPUKernel::getGlobalSampler( size_t inComponentIndex )
  {
    GPUAssert( inComponentIndex < _globalSamplers.size(), "Invalid global sampler index" );
    return _globalSamplers[inComponentIndex];
  }

  void GPUKernel::getGlobalInterpolant( size_t inComponentIndex, GPUInterpolant& outInterpolant )
  {
    GPUAssert( inComponentIndex < _globalInterpolants.size(), "Invalid global interpolant index" );
    outInterpolant = _globalInterpolants[inComponentIndex];
  }

  GPUKernel::TextureHandle GPUKernel::getGlobalOutput( size_t inComponentIndex )
  {
    GPUAssert( inComponentIndex < _globalOutputs.size(), "Invalid global output index" );
    return _globalOutputs[inComponentIndex];
  }

  /// Argument Types
  GPUKernel::TextureHandle GPUKernel::ArgumentType::getTexture( GPUKernel* inKernel, size_t inIndex, size_t inComponent )
  {
    GPUAssert( false, "No textures in argument" );
    return NULL;
  }

  void GPUKernel::ArgumentType::getInterpolant( GPUKernel* inKernel, 
                                                size_t inIndex, 
                                                size_t inComponent,
                                                GPUInterpolant &outInterpolant)
  {
    GPUAssert( false, "No interpolant in argument" );
  }

  float4 GPUKernel::ArgumentType::getConstant( GPUKernel* inKernel, 
                                               size_t inIndex, 
                                               size_t inComponent)
  {
    GPUAssert( false, "No constants in argument" );
    return float4(0,0,0,0);
  }

  // Stream
  GPUKernel::TextureHandle 
  GPUKernel::StreamArgumentType::getTexture( GPUKernel* inKernel, 
                                             size_t inIndex, 
                                             size_t inComponent )
  {
    return inKernel->_streamArguments[ inIndex ]->
      getIndexedFieldTexture( inComponent );
  }

  void GPUKernel::StreamArgumentType::getInterpolant( GPUKernel* inKernel, 
                                                      size_t inIndex, 
                                                      size_t inComponent,
                                                      GPUInterpolant &outInterpolant)
  {
    using namespace ::brook::desc;
    GPUStream* stream = inKernel->_streamArguments[ inIndex ];
    switch( inComponent )
    {
    case kStreamInterpolant_Position:
      stream->getStreamInterpolant(inKernel->_outTextureWidth, 
                                   inKernel->_outTextureHeight, 
                                   outInterpolant);
      return;
    }
    GPUError("not implemented");
  }

      float4 GPUKernel::getATIndexofNumerConstant( unsigned int rank, const unsigned int* domainMin, const unsigned int* domainMax )
    {
        float4 result(0,0,0,0);
        for( unsigned int r = 0; r < rank; r++ )
        {
            unsigned int d = rank - (r+1);
            unsigned int streamExtent = domainMax[d] - domainMin[d];
            unsigned int outputExtent = _outReversedExtents[r];

            unsigned int numer = 1;
            if( streamExtent > outputExtent )
                numer = streamExtent / outputExtent;

            ((float*)&result)[r] = (float)numer;
        }
        return result;
    }

    float4 GPUKernel::getATIndexofDenomConstant( unsigned int rank, const unsigned int* domainMin, const unsigned int* domainMax )
    {
        float4 result(0,0,0,0);
        for( unsigned int r = 0; r < rank; r++ )
        {
            unsigned int d = rank - (r+1);
            unsigned int streamExtent = domainMax[d] - domainMin[d];
            unsigned int outputExtent = _outReversedExtents[r];

            unsigned int denom = 1;
            if( streamExtent < outputExtent )
                denom = outputExtent / streamExtent;

            ((float*)&result)[r] = 1.0f / (float)denom;
        }
        return result;
    }

  float4 GPUKernel::StreamArgumentType::getConstant( GPUKernel* inKernel, 
                                                     size_t inIndex, 
                                                     size_t inComponent )
  {
    using namespace ::brook::desc;
    GPUStream* stream = inKernel->_streamArguments[ inIndex ];
    switch( inComponent )
    {
    case kStreamConstant_Indexof:
      return stream->getIndexofConstant();
      break;
    case kStreamConstant_ATIndexofNumer:
        {
            unsigned int rank = stream->getRank();
            const unsigned int* domainMin = stream->getDomainMin();
            const unsigned int* domainMax = stream->getDomainMax();
            return inKernel->getATIndexofNumerConstant( rank, domainMin, domainMax );
        }
        break;
    case kStreamConstant_ATIndexofDenom:
        {
            unsigned int rank = stream->getRank();
            const unsigned int* domainMin = stream->getDomainMin();
            const unsigned int* domainMax = stream->getDomainMax();
            return inKernel->getATIndexofDenomConstant( rank, domainMin, domainMax );
        }
        break;
    case kStreamConstant_ATLinearize:
        return stream->getATLinearizeConstant();
        break;
    case kStreamConstant_ATTextureShape:
        return stream->getATTextureShapeConstant();
        break;
    case kStreamConstant_ATDomainMin:
        return stream->getATDomainMinConstant();
        break;
    }
    GPUError("not implemented");
    return float4(0,0,0,0);
  }

  // Iterator
  void 
  GPUKernel::IteratorArgumentType::getInterpolant( GPUKernel* inKernel, 
                                                   size_t inIndex, 
                                                   size_t inComponent,
                                                   GPUInterpolant &outInterpolant )
  {
    GPUIterator* iterator = inKernel->_iteratorArguments[ inIndex ];
    iterator->getInterpolant( inKernel->_outTextureWidth, inKernel->_outTextureHeight, outInterpolant );
  }

  float4 GPUKernel::IteratorArgumentType::getConstant( GPUKernel* inKernel, 
                                                       size_t inIndex, 
                                                       size_t inComponent )
  {
    using namespace ::brook::desc;
    GPUIterator* iter = inKernel->_iteratorArguments[ inIndex ];
    switch( inComponent )
    {
    case kIteratorConstant_ATIndexofNumer:
        {
            unsigned int rank = iter->getRank();
            const unsigned int* domainMin = iter->getDomainMin();
            const unsigned int* domainMax = iter->getDomainMax();
            return inKernel->getATIndexofNumerConstant( rank, domainMin, domainMax );
        }
        break;
    case kIteratorConstant_ATIndexofDenom:
        {
            unsigned int rank = iter->getRank();
            const unsigned int* domainMin = iter->getDomainMin();
            const unsigned int* domainMax = iter->getDomainMax();
            return inKernel->getATIndexofDenomConstant( rank, domainMin, domainMax );
        }
        break;
    case kIteratorConstant_ATValueBase:
        return iter->getValueBaseConstant();
        break;
    case kIteratorConstant_ATValueOffset1:
        return iter->getValueOffset1Constant();
        break;
    case kIteratorConstant_ATValueOffset4:
        return iter->getValueOffset4Constant();
        break;
    }
    GPUError("not implemented");
    return float4(0,0,0,0);
  }

  // Constant
  float4 GPUKernel::ConstantArgumentType::getConstant( GPUKernel* inKernel, 
                                                       size_t inIndex, 
                                                       size_t inComponent )
  {
    return inKernel->_constantArguments[ inIndex ];
  }

  // Gather
  GPUKernel::TextureHandle 
  GPUKernel::GatherArgumentType::getTexture( GPUKernel* inKernel, 
                                             size_t inIndex, 
                                             size_t inComponent )
  {
    return inKernel->_gatherArguments[ inIndex ]->
      getIndexedFieldTexture( inComponent );
  }

  float4 GPUKernel::GatherArgumentType::getConstant( GPUKernel* inKernel, 
                                                     size_t inIndex, 
                                                     size_t inComponent )
  {
    using namespace ::brook::desc;
    GPUStream* stream = inKernel->_gatherArguments[ inIndex ];
    switch( inComponent )
    {
    case kGatherConstant_Shape:
      return stream->getGatherConstant();
      break;
    case kGatherConstant_ATLinearize:
        return stream->getATLinearizeConstant();
        break;
    case kGatherConstant_ATTextureShape:
        return stream->getATTextureShapeConstant();
        break;
    case kGatherConstant_ATDomainMin:
        {
            float4 result = stream->getATDomainMinConstant();
            // extra biasing for the round-to-nearest
            // step that is done for gather indices, but
            // not general stream indices:
            result.x += 0.5f;
            result.y += 0.5f;
            result.z += 0.5f;
            result.w += 0.5f;
            return result;
        }
        break;
      break;
    }
    GPUError("not implemented");
    return float4(0,0,0,0);
  }

  // Output
  GPUKernel::TextureHandle 
  GPUKernel::OutputArgumentType::getTexture( GPUKernel* inKernel, 
                                             size_t inIndex, 
                                             size_t inComponent )
  {
    return inKernel->_outputArguments[ inIndex ]->
      getIndexedFieldTexture( inComponent );
  }

  void GPUKernel::OutputArgumentType::getInterpolant( GPUKernel* inKernel, 
                                                      size_t inIndex, 
                                                      size_t inComponent,
                                                      GPUInterpolant &outInterpolant)
  {
    using namespace ::brook::desc;
    GPUStream* stream = inKernel->_outputArguments[ inIndex ];
    switch( inComponent )
    {
    case kOutputInterpolant_Position:
      stream->getStreamInterpolant(
        inKernel->_outTextureWidth, 
        inKernel->_outTextureHeight, 
        outInterpolant);
      return;
    }
    GPUError("not implemented");
  }

  float4 GPUKernel::OutputArgumentType::getConstant( GPUKernel* inKernel, 
                                                     size_t inIndex, 
                                                     size_t inComponent )
  {
    using namespace ::brook::desc;
    GPUStream* stream = inKernel->_outputArguments[ inIndex ];
    switch( inComponent )
    {
    case kOutputConstant_Indexof:
      return stream->getIndexofConstant();
      break;
    }
    GPUError("not implemented");
    return float4(0,0,0,0);
  }
}
