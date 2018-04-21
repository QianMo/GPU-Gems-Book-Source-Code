#include <paramgl/param.h>

const Param<int> dummy("error");

float
ParamBase::GetFloatValue()
{
  return ((Param<float> *) this)->GetValue();
}

int
ParamBase::GetIntValue()
{
  return ((Param<int> *) this)->GetValue();
}
