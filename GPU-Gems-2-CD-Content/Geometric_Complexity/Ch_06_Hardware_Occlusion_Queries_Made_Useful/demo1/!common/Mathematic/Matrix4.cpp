//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
#pragma hdrstop
#include "Matrix4.h"
//#pragma package(smart_init)
//---------------------------------------------------------------------------
namespace Math {

const Matrix4f Matrix4f::IDENTITY(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
const Matrix4f Matrix4f::ONE(1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1);
const Matrix4f Matrix4f::T05_S05(0.5,0,0,0.5, 0,0.5,0,0.5, 0,0,0.5,0.5, 0,0,0,1);
const Matrix4f Matrix4f::ZERO(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

const Matrix4d Matrix4d::IDENTITY(1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1);
const Matrix4d Matrix4d::ONE(1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1);
const Matrix4d Matrix4d::T05_S05(0.5,0,0,0.5, 0,0.5,0,0.5, 0,0,0.5,0.5, 0,0,0,1);
const Matrix4d Matrix4d::ZERO(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

//namespace
};
