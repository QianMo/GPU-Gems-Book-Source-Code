/*
 * brtdecl.h  --
 * Stream type and gather type are operated on here
 */

#ifndef    BRTDECL_H
#define    BRTDECL_H

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cassert>

#include "symbol.h"
#include "callback.h"
#include "location.h"

#include "dup.h"

#include "decl.h"
#include "express.h"

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class BrtStreamType : public Type
{
public:
  ~BrtStreamType();

  Type* dup0() const = 0;

  bool printStructureStreamHelperType( std::ostream& out, const std::string& name, bool raw ) const {
    return false;
  }

  void printBase(std::ostream& out, int level ) const {assert (0);}
  void printBefore(std::ostream& out, Symbol *name, int level) const {
     assert (0);
  }
  void printAfter(std::ostream& out ) const {assert (0);}

  void printType(std::ostream& out, Symbol *name,
                 bool showBase, int level, bool raw=false ) const = 0;
  void printForm(std::ostream& out) const = 0;
  virtual void printInitializer(std::ostream& out) const = 0;

  void findExpr(fnExprCallback cb);
  bool lookup(Symbol* sym) const { return base ? base->lookup(sym) : false; }

  TypeQual getQualifiers(void) const { return base->getQualifiers(); }
  BaseType *getBase(void) { return base; }

  BaseType       *base;
  ExprVector     dims;
  bool isIterator;

protected:
  /*
   * All constructors are protected because no one should be instantiating
   * these directly.  Use the children.
   */
  BrtStreamType(const ArrayType *t);
  BrtStreamType(const BaseType *_base, const ExprVector _dims); /* For dup0 */
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class BrtStreamParamType : public BrtStreamType
{
public:
  BrtStreamParamType(const ArrayType *t) : BrtStreamType(t) {};

  Type* dup0() const { return new BrtStreamParamType(base, dims); }

  void printType(std::ostream& out, Symbol *name,
                 bool showBase, int level, bool raw=false ) const;
  void printBefore(std::ostream& out, Symbol *name, int level) const {
  }
  void printAfter(std::ostream& out ) const {
  }

  void printForm(std::ostream& out) const;
  void printInitializer(std::ostream& out) const;
  void printBase(std::ostream& out, int level) const;
protected:
  BrtStreamParamType(const BaseType *_base, const ExprVector _dims)
     : BrtStreamType(_base, _dims) {} /* Only for dup0() */
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class BrtInitializedStreamType : public BrtStreamType
{
public:
  BrtInitializedStreamType(const ArrayType *t) : BrtStreamType(t) {};

  Type* dup0() const { return new BrtInitializedStreamType(base, dims); }

  void printType(std::ostream& out, Symbol *name,
                 bool showBase, int level, bool raw=false ) const;
  void printForm(std::ostream& out) const;
  void printInitializer(std::ostream& out) const;

protected:
  BrtInitializedStreamType(const BaseType *_base, const ExprVector _dims)
     : BrtStreamType(_base, _dims) {} /* Only for dup0() */
};


class BrtIterType : public BrtStreamType
{
public:
  BrtIterType(const ArrayType *stream, const FunctionCall *f);
  ~BrtIterType();

  Type* dup0() const { return new BrtIterType(base, dims, args); }

  void printType( std::ostream& out, Symbol *name,
		  bool showBase, int level, bool raw=false ) const;
  void printForm(std::ostream& out) const;
  void printInitializer(std::ostream& out) const;

  void findExpr(fnExprCallback cb);

  ExprVector     args;

protected:
  BrtIterType(const BaseType *_base, const ExprVector _dims,
              const ExprVector _args); /* Only for dup0() */
};

class CPUGatherType{
  mutable bool raw;
   // used instead of changing argument signature of modified printBefore
public:
  Type * at;
  Type * subtype;
  bool copy_on_write;
  unsigned int dimension;
  CPUGatherType(ArrayType &t,bool copy_on_write);
  Type * dup0()const;
  virtual Type ** getSubType() {return &subtype;}
   void printType(std::ostream & out, Symbol *name, bool showBase, int level, bool raw=false) const;
  void printBefore(std::ostream & out, Symbol *name, int level) const;
  void printAfter(std::ostream &out)const;
  void printSubtype(std::ostream &out,Symbol *name, bool showBase,int level)const;
};

class BrtKernelType : public FunctionType
{
public:
  BrtKernelType(FunctionType *functionType);
  ~BrtKernelType();

  Type* dup0() const { return new BrtKernelType(_functionType); }

private:
  Type* convertArgumentType(Type*);

  FunctionType* _functionType;
};

#endif  /* BRTDECL_H */

