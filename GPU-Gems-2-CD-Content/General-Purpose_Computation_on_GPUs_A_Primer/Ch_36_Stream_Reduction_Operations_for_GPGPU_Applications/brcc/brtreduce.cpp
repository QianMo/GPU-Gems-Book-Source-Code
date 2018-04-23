/*
 * brtreduce.cpp
 *
 *      Classes reflecting the body of Brook reduces for the different
 *      backends.  Each one knows how to build itself from a function
 *      definition and then how to emit C++ for itself.
 */
#ifdef _WIN32
#pragma warning(disable:4786)
//the above warning disables visual studio's annoying habit of warning when using the standard set lib
#endif

#include <cstring>
#include <cassert>
#include <sstream>

#include "brtreduce.h"
#include "brtexpress.h"
#include "codegen.h"
#include "main.h"


// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTFP30ReduceCode::printCode(std::ostream& out) const
{
  this->BRTFP30KernelCode::printCode(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTFP40ReduceCode::printCode(std::ostream& out) const
{
  this->BRTFP40KernelCode::printCode(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTARBReduceCode::printCode(std::ostream& out) const
{
  this->BRTARBKernelCode::printCode(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BRTPS20ReduceCode::BRTPS20ReduceCode(const FunctionDef& _fDef)
  : BRTPS20KernelCode(_fDef)//converts gathers
{
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BRTPS2BReduceCode::BRTPS2BReduceCode(const FunctionDef& _fDef)
  : BRTPS2BKernelCode(_fDef)//converts gathers
{
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BRTPS2AReduceCode::BRTPS2AReduceCode(const FunctionDef& _fDef)
  : BRTPS2AKernelCode(_fDef)//converts gathers
{
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BRTPS30ReduceCode::BRTPS30ReduceCode(const FunctionDef& _fDef)
  : BRTPS30KernelCode(_fDef)//converts gathers
{
}

void
BRTPS20ReduceCode::printCode(std::ostream& out) const
{
  this->BRTPS20KernelCode::printCode(out);
}

void
BRTPS2BReduceCode::printCode(std::ostream& out) const
{
	this->BRTPS2BKernelCode::printCode(out);
}

void
BRTPS2AReduceCode::printCode(std::ostream& out) const
{
	this->BRTPS2AKernelCode::printCode(out);
}

void
BRTPS30ReduceCode::printCode(std::ostream& out) const
{
  this->BRTPS30KernelCode::printCode(out);
}

