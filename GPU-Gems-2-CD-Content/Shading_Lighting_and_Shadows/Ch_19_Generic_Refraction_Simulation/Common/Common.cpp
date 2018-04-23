///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Common.cpp
//  Desc : Common data/macros/functions
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include <stdio.h>
#include <cstdarg>

// output message
void OutputMsg(const char *pTitle, const char *pText, ...) 
{
  assert(pText && pText && "OutputMsg: Invalid Input message");

  va_list vl;
  char pStr[4096];

  // get variable-argument list
  va_start(vl, pText);  
  // write formatted output
  vsprintf(pStr, pText, vl);  
  va_end(vl);  

  // just do it!
  MessageBox(NULL,pStr, pTitle,MB_OK|MB_ICONINFORMATION);  
 }
