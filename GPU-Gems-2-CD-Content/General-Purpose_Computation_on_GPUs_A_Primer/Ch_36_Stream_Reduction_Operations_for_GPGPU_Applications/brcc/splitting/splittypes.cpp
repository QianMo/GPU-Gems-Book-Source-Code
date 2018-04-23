// splittypes.cpp
#include "splittypes.h"

std::ostream& operator<<( std::ostream& inStream, SplitBasicType inType )
{
  switch(inType)
  {
  case kSplitBasicType_Float:
    inStream << "float";
    break;
  case kSplitBasicType_Float2:
    inStream << "float2";
    break;
  case kSplitBasicType_Float3:
    inStream << "float3";
    break;
  case kSplitBasicType_Float4:
    inStream << "float4";
    break;
  default:
    inStream << "unknown";
    break;
  }
  return inStream;
}


