/*
 * brtreduce.h
 *
 *      Header for the various BRT Reduce classes (objects responsible for
 *      compiling and emitting reducers for the various backends).
 */
#ifndef    _BRTREDUCE_H_
#define    _BRTREDUCE_H_

#include "dup.h"
#include "stemnt.h"
#include "brtkernel.h"
#include "b2ctransform.h"

class BRTFP30ReduceCode : public BRTFP30KernelCode
{
  public:
    BRTFP30ReduceCode(const FunctionDef& _fDef) : BRTFP30KernelCode(_fDef) {};
   ~BRTFP30ReduceCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTFP30ReduceCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTFP40ReduceCode : public BRTFP40KernelCode
{
  public:
    BRTFP40ReduceCode(const FunctionDef& _fDef) : BRTFP40KernelCode(_fDef) {};
   ~BRTFP40ReduceCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTFP40ReduceCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTARBReduceCode : public BRTARBKernelCode
{
  public:
    BRTARBReduceCode(const FunctionDef& _fDef) : BRTARBKernelCode(_fDef) {};
   ~BRTARBReduceCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTARBReduceCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTPS20ReduceCode : public BRTPS20KernelCode
{
  public:
    BRTPS20ReduceCode(const FunctionDef& fDef);
   ~BRTPS20ReduceCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTPS20ReduceCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTPS2BReduceCode : public BRTPS2BKernelCode
{
  public:
    BRTPS2BReduceCode(const FunctionDef& fDef);
   ~BRTPS2BReduceCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTPS2BReduceCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTPS2AReduceCode : public BRTPS2AKernelCode
{
  public:
    BRTPS2AReduceCode(const FunctionDef& fDef);
   ~BRTPS2AReduceCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTPS2AReduceCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTPS30ReduceCode : public BRTPS30KernelCode
{
  public:
    BRTPS30ReduceCode(const FunctionDef& fDef);
   ~BRTPS30ReduceCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTPS30ReduceCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTCPUReduceCode : public BRTCPUKernelCode
{
public: 
  BRTCPUReduceCode(const FunctionDef& fDef) : BRTCPUKernelCode(fDef) {};
};

#endif  /* STEMNT_H */

