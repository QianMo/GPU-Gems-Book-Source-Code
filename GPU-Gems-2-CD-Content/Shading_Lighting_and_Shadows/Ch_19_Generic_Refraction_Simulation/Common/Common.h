///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Common.h
//  Desc : Common data/macros/functions
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

// enable memory leaks detection
//#ifdef DEBUG
//#define CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>
//#endif

// exclude redundant stuff
#define WIN32_LEAN_AND_MEAN
#define STRICT
#include <windows.h>

// directx9 
#include <d3dx9.h>
#pragma comment (lib, "d3d9.lib")
#pragma comment (lib, "d3dx9.lib")

// nvidia CG
#include <Cg/cgD3D9.h>
#pragma comment (lib, "cg.lib")
#pragma comment (lib, "cgD3D9.lib")

// stl 
#include <vector>
#include <list>
#include <deque>
#include <stack>
#include <iostream>
#include <cassert>
#include <string>
using namespace std;

// common unsigned data types
typedef unsigned int    uint;
typedef unsigned long   ulong;
typedef unsigned short  ushort;
typedef unsigned char   uchar;
typedef unsigned long   DWORD;

// safe release/delete macros

// use this in classes with a release method
#define SAFE_RELEASE(p) \
  if (p) { \
    (p)->Release(); \
    (p) = 0; \
  } \

// use this do delete a pointer
#define SAFE_DELETE(p) \
  if(p) { \
    delete (p); \
    (p) = 0; \
  } \

// use this to delete an array
#define SAFE_DELETE_ARRAY(p)  \
  if(p) { \
    delete[] (p); \
    (p) = 0; \
  } \

// macro to return 32bit color
#define CREATECOLOR(r, g, b, a) (((uchar)(a) << 24) | ((uchar)(r) << 16) |  ((uchar)(g) << 8) | ((uchar)(b)))

// error control
enum AppErrors {
  APP_OK,                   // it's ok  
  APP_ERR_UNKNOWN,          // unknown error
  APP_ERR_OUTOFMEMORY,      // out of memory  
  APP_ERR_INVALIDCALL,      // invalid function call
  APP_ERR_INVALIDPARAM,     // invalid parameter passed  
  APP_ERR_NOTINITIALIZED,   // not initialized
  APP_ERR_NOTSUPPORTED,     // not supported
  APP_ERR_NOTFOUND,         // data not found  
  APP_ERR_READFAIL,         // file read fail
  APP_ERR_WRITEFAIL,        // file write fail
  APP_ERR_INITFAIL,         // initialization fail
  APP_ERR_SHUTFAIL          // shutdown fail
};

#define APP_FAILED(exp)     ((exp) != APP_OK)
#define APP_SUCCEEDED(exp)  ((exp) == APP_OK)

// data paths
#define APP_DATAPATH_SHADERS "../Media/Shaders/"
#define APP_DATAPATH_MODELS "../Media/Models/"
#define APP_DATAPATH_TEXTURES "../Media/Textures/"
#define APP_DATAPATH_MUSICS "../Media/Musics/"

// output message
void OutputMsg(const char *pTitle, const char *pText, ...);

#include "Mathlib.h"
#include "Vector.h"
#include "Matrix.h"
#include "Color.h"
#include "Timer.h"  
