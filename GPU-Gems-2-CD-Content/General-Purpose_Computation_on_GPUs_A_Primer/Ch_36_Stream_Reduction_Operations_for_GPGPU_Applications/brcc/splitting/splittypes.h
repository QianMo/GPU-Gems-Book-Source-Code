// splittypes.h
#ifndef __SPLITTYPES_H__
#define __SPLITTYPES_H__

#include <iostream>

enum SplitBasicType
{
  kSplitBasicType_Unknown = -1,
  kSplitBasicType_Float = 0,
  kSplitBasicType_Float2,
  kSplitBasicType_Float3,
  kSplitBasicType_Float4
};

std::ostream& operator<<( std::ostream&, SplitBasicType );

#endif

