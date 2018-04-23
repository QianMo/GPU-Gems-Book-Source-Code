/*
 * brtkernel.h
 *
 *      Header for the various BRT Kernel classes (objects responsible for
 *      compiling and emitting kernels for the various backends).
 */
#ifndef    _BRTKERNEL_H_
#define    _BRTKERNEL_H_

#include "dup.h"
#include "stemnt.h"
#include "b2ctransform.h"
#include "codegen.h"

class BRTKernelCode;
typedef Dup<BRTKernelCode> DupableBRTKernelCode;

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class BRTKernelCode : public DupableBRTKernelCode
{
  public:
    BRTKernelCode(const FunctionDef& _fDef) {
       fDef = (FunctionDef *) _fDef.dup0();
    };

    /*
     * Deleting fDef, even though it's dupped, appears to free memory that
     * has already been freed.  I'm suspicious that one of the dup methods
     * doesn't go deep enough, but haven't chased it down.  --Jeremy.
     */
    virtual ~BRTKernelCode() { delete fDef; };

    BRTKernelCode *dup0() const = 0;
    virtual void printCode(std::ostream& out) const = 0;
    virtual void printInnerCode(std::ostream&out)const=0;
    friend std::ostream& operator<< (std::ostream& o, const BRTKernelCode& k);

    FunctionDef *fDef;
    bool standAloneKernel() const;

    virtual void onlyPrintInner(std::ostream &out) const{}
};


class BRTGPUKernelCode : public BRTKernelCode
{
public:
   BRTGPUKernelCode(const FunctionDef& fDef);
   ~BRTGPUKernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }
   
   void printInnerCode(std::ostream&out)const;

   /* static so it can be passed as a findExpr() callback */
   static Expression *ConvertGathers(Expression *e);
   
   void printCodeForType(std::ostream& out, 
                         CodeGenTarget target) const;
   
   virtual BRTKernelCode *dup0() const = 0;
   virtual void printCode(std::ostream& out) const = 0;
};

class BRTFP30KernelCode : public BRTGPUKernelCode
{
  public:
    BRTFP30KernelCode(const FunctionDef& fDef) : BRTGPUKernelCode(fDef) {}
   ~BRTFP30KernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTFP30KernelCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTFP40KernelCode : public BRTGPUKernelCode
{
  public:
    BRTFP40KernelCode(const FunctionDef& fDef) : BRTGPUKernelCode(fDef) {}
   ~BRTFP40KernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTFP40KernelCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTARBKernelCode : public BRTGPUKernelCode
{
  public:
    BRTARBKernelCode(const FunctionDef& fDef) : BRTGPUKernelCode(fDef) {}
   ~BRTARBKernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }

    BRTKernelCode *dup0() const { return new BRTARBKernelCode(*this->fDef); }
    void printCode(std::ostream& out) const;
};

class BRTPS20KernelCode : public BRTGPUKernelCode
{
  public:
   BRTPS20KernelCode(const FunctionDef& fDef) : BRTGPUKernelCode(fDef) {}
   ~BRTPS20KernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }
   
   BRTKernelCode *dup0() const { return new BRTPS20KernelCode(*this->fDef); }
   void printCode(std::ostream& out) const;
};

class BRTPS2BKernelCode : public BRTGPUKernelCode
{
  public:
   BRTPS2BKernelCode(const FunctionDef& fDef) : BRTGPUKernelCode(fDef) {}
   ~BRTPS2BKernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }
   
   BRTKernelCode *dup0() const { return new BRTPS2BKernelCode(*this->fDef); }
   void printCode(std::ostream& out) const;
};

class BRTPS2AKernelCode : public BRTGPUKernelCode
{
  public:
   BRTPS2AKernelCode(const FunctionDef& fDef) : BRTGPUKernelCode(fDef) {}
   ~BRTPS2AKernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }
   
   BRTKernelCode *dup0() const { return new BRTPS2AKernelCode(*this->fDef); }
   void printCode(std::ostream& out) const;
};

class BRTPS30KernelCode : public BRTGPUKernelCode
{
  public:
   BRTPS30KernelCode(const FunctionDef& fDef) : BRTGPUKernelCode(fDef) {}
   ~BRTPS30KernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }
   
   BRTKernelCode *dup0() const { return new BRTPS30KernelCode(*this->fDef); }
   void printCode(std::ostream& out) const;
};


class BRTCPUKernelCode : public BRTKernelCode
{
public:
  
  BRTCPUKernelCode(const FunctionDef& _fDef) : BRTKernelCode(_fDef) {}
  ~BRTCPUKernelCode() { /* Nothing, ~BRTKernelCode() does all the work */ }
  
  BRTKernelCode *dup0() const { 
    return new BRTCPUKernelCode(*this->fDef); 
  }

  void printCode(std::ostream& out) const;
  void onlyPrintInner(std::ostream& out) const;
  void printInnerCode(std::ostream& out) const;
};

#endif  /* STEMNT_H */

