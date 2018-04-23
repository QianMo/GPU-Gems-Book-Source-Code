/*
 * brtstemnt.h
 *
 *      Brook extensions to cTool's stemnt.h for kernel definitions.
 */
#ifndef    _BRTSTEMNT_H_
#define    _BRTSTEMNT_H_

#include "stemnt.h"
#include "b2ctransform.h"

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class BRTKernelDef : public FunctionDef
{
  public:
    BRTKernelDef(const FunctionDef& fDef);

    /* Pass ourselves (as a FunctionDef) to our own constructor */
    Statement *dup0() const { 
       return new BRTKernelDef(*static_cast<const FunctionDef*>(this)); };
    void print(std::ostream& out, int level) const;

    void printStub(std::ostream& out) const;
    virtual void PrintVoutPrefix(std::ostream & out)const;
    virtual void PrintVoutDimensionalShift(std::ostream & out,
                                           Decl * decl, 
                                           unsigned int dim)const;
    virtual void PrintVoutPostfix(std::ostream & out)const;
    virtual bool CheckSemantics(void) const;
};

class BRTMapKernelDef : public BRTKernelDef
{
  public:
    BRTMapKernelDef(const FunctionDef& fDef) : BRTKernelDef(fDef) {
       if (!CheckSemantics()) assert(false);
    }

    /* Pass ourselves (as a FunctionDef) to our own constructor */
    Statement *dup0() const {
       return new BRTMapKernelDef(*static_cast<const FunctionDef*>(this));
    };

    bool CheckSemantics(void) const;
};

class BRTReduceKernelDef : public BRTKernelDef
{
  public:
    BRTReduceKernelDef(const FunctionDef& fDef) : BRTKernelDef(fDef) {
       if (!CheckSemantics()) assert(false);
    }

    /* Pass ourselves (as a FunctionDef) to our own constructor */
    Statement *dup0() const {
       return new BRTReduceKernelDef(*static_cast<const FunctionDef*>(this));
    };

    bool CheckSemantics(void) const;
};

#endif  /* STEMNT_H */
