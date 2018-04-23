// gpuiterator.cpp
#include "gpuiterator.hpp"

#include "gpuruntime.hpp"

using namespace brook;

static float lerp (unsigned int i, unsigned int end,float lower,float upper) {
   float frac=end>0?((float)i)/(float)end:(float)upper;
   return (1-frac)*lower+frac*upper;
}

GPUIterator* GPUIterator::create( GPURuntime* inRuntime, 
                                  StreamType inElementType,
                                  unsigned int inDimensionCount, 
                                  const unsigned int inExtents[],
                                  const float inRanges[] )
{
  GPUIterator* result = new GPUIterator( inRuntime, inElementType );
  if( result->initialize( inDimensionCount, inExtents, inRanges ) )
    return result;
  delete result;
  return NULL;
}

GPUIterator::GPUIterator( GPURuntime* inRuntime, StreamType inElementType )
  : Iter(inElementType), _context(inRuntime->getContext()), _cpuBuffer(NULL), _requiresAddressTranslation(false)
{
}

bool GPUIterator::initialize( unsigned int inDimensionCount, 
                              const unsigned int inExtents[], 
                              const float inRanges[] )
{
  _dimensionCount = inDimensionCount;
  if( (_dimensionCount <= 0) || (_dimensionCount > 4) )
  {
    GPUWARN << "Invalid dimension for iterator stream "
      << inDimensionCount << ".\n"
      << "Dimension must be greater than 0 and less than 5.";
    return false;
  }
  if( _dimensionCount > 2 )
  {
      _requiresAddressTranslation = true;
  }

  switch( type )
  {
  case __BRTFLOAT:
    _componentCount = 1;
    break;
  case __BRTFLOAT2:
    _componentCount = 2;
    break;
  case __BRTFLOAT3:
    _componentCount = 3;
    break;
  case __BRTFLOAT4:
    _componentCount = 4;
    break;
  default:
    GPUWARN << "Unknown iterator element type.\n"
      "Element type must be one of float, float2, float3, float4.";
    return false;
    break;
  }

  if( _dimensionCount != 1 && _dimensionCount != _componentCount )
  {
      GPUWARN << "Cannot create a " << _dimensionCount << "D iterator"
          << " with float" << _componentCount << " elements." << std::endl;
      return false;
  }

  _totalSize = 1;
  for( unsigned int i = 0; i < _dimensionCount; i++ )
  {
    int extent = inExtents[i];
    if( extent <= 0 )
    {
      GPUWARN << "Invalid iterator extent " << extent << " in dimension " << i << ".\n"
        << "The extent must be greater than 0.";
      return false;
    }

    _extents[i] = extent;
    _totalSize *= _extents[i];
    _domainMax[i] = _extents[i];
    _domainMin[i] = 0;
  }

  unsigned int rangeCount = _componentCount*2;
  unsigned int r;
  for( r = 0; r < rangeCount; r++ )
    _ranges[r] = inRanges[r];

  if( _dimensionCount == 1 )
  {
    float4 min = float4(0,0,0,0);
    float4 max = float4(0,0,0,0);

    unsigned int i;
    for( i = 0; i < _componentCount; i++ )
    {
      ((float*)&min)[i] = _ranges[i];
      ((float*)&max)[i] = _ranges[i+_componentCount];
    }
    // fill in remaining components just in case...
    for( ; i < 4; i++ )
    {
      ((float*)&min)[i] = (i==4) ? 1.0f : 0.0f;
      ((float*)&max)[i] = (i==4) ? 1.0f : 0.0f;
    }

    _min1D = min;
    _max1D = max;

    _context->get1DInterpolant( _min1D, _max1D, _extents[0], _defaultInterpolant );
//    _rect.vertices[0] = max;
//    _rect.vertices[1] = min;
//    _rect.vertices[2] = max;
//    _rect.vertices[3] = min;
  }
  else if( _dimensionCount == 2 )
  {
    float minX = _ranges[0];
    float minY = _ranges[1];
    float maxX = _ranges[2];
    float maxY = _ranges[3];

    _min2D = float2( minX, minY );
    _max2D = float2( maxX, maxY );

    _context->get2DInterpolant( _min2D, _max2D, _extents[1], _extents[0], _defaultInterpolant );
  }

  // TIM: TODO: figure out what to do with the rest of the cases... :(
  _valueBase = float4(0,0,0,0);
  _valueOffset1 = float4(0,0,0,0);
  _valueOffset4 = float4(0,0,0,0);

  unsigned int c;
  for( c = 0; c < _componentCount; c++ )
      ((float*)&_valueBase)[c] = _ranges[c];

  if( _dimensionCount == 1 )
  {
    for( c = 0; c < _componentCount; c++ )
      ((float*)&_valueOffset1)[c] = (_ranges[c + _componentCount] - _ranges[c]) / (float)_extents[0];
  }
  else
  {
    for( c = 0; c < _componentCount; c++ )
    {
      unsigned int d = _componentCount - (c+1);
//      GPUWARN << "extent " << _extents[d] << std::endl;
      ((float*)&_valueOffset4)[c] = (_ranges[c + _componentCount] - _ranges[c]) / (float)_extents[d];
    }
  }

//  GPUWARN << "base " << _valueBase.x << " " << _valueBase.y << " " << _valueBase.z << " " << _valueBase.w << " " << std::endl;
//  GPUWARN << "offset1 " << _valueOffset1.x << " " << _valueOffset1.y << " " << _valueOffset1.z << " " << _valueOffset1.w << " " << std::endl;
//  GPUWARN << "offset4 " << _valueOffset4.x << " " << _valueOffset4.y << " " << _valueOffset4.z << " " << _valueOffset4.w << " " << std::endl;

  return true;
}

void GPUIterator::getInterpolant(
  unsigned int inOutputWidth,
  unsigned int inOutputHeight,
  GPUInterpolant& outInterpolant )
{
  if( _dimensionCount == 1 )
  {
    if( inOutputWidth == _extents[0] )
    {
      outInterpolant = _defaultInterpolant;
      return;
    }

    _context->get1DInterpolant( _min1D, _max1D, _extents[0], outInterpolant );
  }
  else
  {
    if( inOutputWidth == _extents[1] && inOutputHeight == _extents[0] )
    {
      outInterpolant = _defaultInterpolant;
      return;
    }

    _context->get2DInterpolant( _min2D, _max2D, _extents[1], _extents[0], outInterpolant );
  }
}

void* GPUIterator::getData (unsigned int flags)
{
  GPUAssert( !(flags & Stream::WRITE),
    "Attempted to write to an iterator.\n"
    "Iterators are strictly read-only." );

  if( _cpuBuffer != NULL ) return _cpuBuffer;

  size_t cpuBufferSize = _totalSize * _componentCount * sizeof(float);

  _cpuBuffer = malloc( cpuBufferSize );

  // fill in the data
  float* data = (float*)_cpuBuffer;
  if( _dimensionCount == 1 )
  {
    for( unsigned int i = 0; i < _extents[0]; i++ )
    {
      for( unsigned int j = 0; j < _componentCount; j++ )
        *data++ = lerp( i, _extents[0], _ranges[j], _ranges[j+_componentCount] );
    }
  }
  else if( _dimensionCount == 2 )
  {
    unsigned int i[2];
    for( i[0] = 0; i[0] < _extents[0]; i[0]++ )
    {
      for( i[1] = 0; i[1] < _extents[1]; i[1]++ )
      {
        for( unsigned int k = 0; k < 2; k++ )
          *data++ = lerp( i[1-k], _extents[1-k], _ranges[k], _ranges[2+k] );
      }
    }
  }
  else
  {
    GPUAssert( false, "Should be unreachable" );
  }
  
  return _cpuBuffer;
}

void GPUIterator::releaseData(unsigned int flags) {
  // empty
}
