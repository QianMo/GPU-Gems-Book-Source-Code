/**
  @file Quake3Md3.h

  @maintainer Kevin Egan

*/

#ifndef _QUAKE_3_MD3_
#define _QUAKE_3_MD3_

#include <string>

class BasicModel;

BasicModel* loadQuake3Md3(
        const std::string &     modelName,
        const std::string &     fileName,
        int                     startAnimation);

#endif

