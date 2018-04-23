/*
 * brtdecl.cpp --
 * Stream types and gather types are actually printed and dealt with here.
 */

#include <cassert>
#include <cstring>

#include "decl.h"
#include "express.h"
#include "stemnt.h"

#include "token.h"
#include "gram.h"
#include "project.h"

#include "brtdecl.h"

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BrtStreamType::BrtStreamType(const ArrayType *t)
  : Type(TT_BrtStream)
{
  const ArrayType *p;

  // First find the base type of the array;
  for (p = t;
       p->subType && p->subType->isArray(); p = (ArrayType *)p->subType) {

     //assert(p->size);
     if (p->size)
         dims.push_back(p->size->dup0());
  }

  /*
   * p->size is NULL when parsing s<> (i.e. a stream function argument or
   * some other case where no length information is given).  This is fine in
   * some cases, but not in others, so the super class tolerates it and lets
   * the children individually decide if that's okay for them.
   */

  if (p->size) {
     dims.push_back(p->size->dup0());
  }

  assert (p->subType);
  assert (p->subType->isBaseType());

  base = (BaseType *) p->subType->dup0();

  isIterator = (t->getQualifiers() & TQ_Iter) != 0;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BrtStreamType::BrtStreamType(const BaseType *_base, const ExprVector _dims)
  : Type(TT_BrtStream)
{
  isIterator = false;

   ExprVector::const_iterator i;

   base = (BaseType *) _base->dup();
   for (i = _dims.begin(); i != _dims.end(); i++) {
      dims.push_back((*i)->dup());
   }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BrtStreamType::~BrtStreamType()
{
    // Handled by deleting the global type list
    // delete subType;
    for (unsigned int i=0; i<dims.size(); i++)
      delete dims[i];
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtStreamType::findExpr( fnExprCallback cb )
{
    if (base)
        base->findExpr(cb);

    for (unsigned int i=0; i<dims.size(); i++) {
       dims[i] = (cb)(dims[i]);
       dims[i]->findExpr(cb);
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtStreamParamType::printType(std::ostream& out, Symbol *name,
                              bool showBase, int level, bool raw) const
{
  if( isIterator )
    out << "::brook::iter ";
  else
    out << "::brook::stream ";

  if (name)
    out << *name;
}
void 
BrtStreamParamType::printBase(std::ostream &out, int level) const 
{
  if( isIterator )
    out << "::brook::iter ";
  else
    out << "::brook::stream ";
}
void
BrtStreamParamType::printForm(std::ostream& out) const
{
    out << "-BrtStreamParam Type ";
    if (base)
        base->printBase(out, 0);
}

void
BrtStreamParamType::printInitializer(std::ostream& out) const
{
   /* Nothing.  There are no initializers for parameters to functions */
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtInitializedStreamType::printType(std::ostream& out, Symbol *name,
                                    bool showBase, int level, bool raw) const
{
  out << "::brook::stream ";

  if (name)
    out << *name;

  // TIM: add initializer as constructor
  out << "(::brook::getStreamType(( ";

  base->printType( out, NULL, true, 0 ,raw);

  out << " *)0), ";
  /*
  if (base->getBase()->typemask&BT_Float4) {
    out << "__BRTFLOAT4";
  }else if (base->getBase()->typemask&BT_Float3) {
    out << "__BRTFLOAT3";
  }else if (base->getBase()->typemask&BT_Float2) {
    out << "__BRTFLOAT2";
  }else if (base->getBase()->typemask&BT_Float) {
    out << "__BRTFLOAT";
  }else {
    std::cerr << "Warning: Unsupported stream type ";
    base->printBase(std::cerr,0);
    std::cerr << std::endl;
    out << "__BRTFLOAT";
  }*/
  for (unsigned int i=0; i<dims.size(); i++) {
    dims[i]->print(out);
    out << ",";
  }
  out << "-1)";
}

void
BrtInitializedStreamType::printForm(std::ostream& out) const
{
    out << "-BrtInitializedStream Type ";
    if (base)
        base->printBase(out, 0);
}

void
BrtInitializedStreamType::printInitializer(std::ostream& out) const
{
  /*
   * I think this is bogus.  I'm pretty sure BrtInitializedStreams don't
   * have initializers per se, instead they use constructor syntax. --Jeremy.
   */
  out << "__BRTCreateStream(\"";
  base->printBase(out, 0);
  out << "\",";
  for (unsigned int i=0; i<dims.size(); i++) {
    dims[i]->print(out);
    out << ",";
  }
  out << "-1)";
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BrtIterType::BrtIterType(const ArrayType *stream, const FunctionCall *f)
  : BrtStreamType(stream)
{
  ExprVector::const_iterator i;

  assert(f->function->etype == ET_Variable);
  assert(strcmp(((Variable *) f->function)->name->name.c_str(),"iter") == 0);

  /*
   * Now fish the min / max out of 'f'.
   *
   * We impose the following constraints on dimensions
   *   - We support float, float2, float3, or float4 1-D streams
   *   - We float2 2-D streams
   */
  assert(f->args.size() == 2);
  assert(f->args[0]->type);
  assert(f->args[0]->type->getBase()->typemask == base->typemask);
  assert(f->args[1]->type);
  assert(f->args[1]->type->getBase()->typemask == base->typemask);

  assert(dims.size() == 1 ||
         (dims.size() == 2 && base->typemask == BT_Float2));
  for (i = f->args.begin(); i != f->args.end(); i++) {
     args.push_back((*i)->dup0());
  }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BrtIterType::BrtIterType(const BaseType *_base, const ExprVector _dims,
                         const ExprVector _args)
  : BrtStreamType(_base, _dims)
{
   ExprVector::const_iterator i;

   for (i = _args.begin(); i != _args.end(); i++) {
      args.push_back((*i)->dup());
   }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BrtIterType::~BrtIterType()
{
   ExprVector::iterator i;

   for (i = args.begin(); i != args.end(); i++) {
      delete *i;
   }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtIterType::printType( std::ostream& out, Symbol *name,
                        bool showBase, int level, bool raw ) const
{
  ExprVector::const_iterator i;

  out << "::brook::iter ";

  if (name)
    out << *name;

  // TIM: add initializer as constructor
  out << "(::brook::";

  if (base->getBase()->typemask&BT_Float4) {
    out << "__BRTFLOAT4, ";
  }else if (base->getBase()->typemask&BT_Float3) {
    out << "__BRTFLOAT3, ";
  }else if (base->getBase()->typemask&BT_Float2) {
    out << "__BRTFLOAT2, ";
  }else if (base->getBase()->typemask&BT_Float) {
    out << "__BRTFLOAT, ";
  }else {
    std::cerr << "Warning: Unsupported iterator type ";
    base->printBase(std::cerr,0);
    std::cerr << std::endl;
    out << "__BRTFLOAT, ";
  }

  /* Now print the dimensions */
  for (i = dims.begin(); i != dims.end(); i++) {
    (*i)->print(out);
    out << ",";
  }
  out << " -1, ";

  /* Now print the min / max */
  for (i = args.begin(); i != args.end(); i++) {
     if (((*i)->type->type == TT_Base) &&
         ((BaseType *) (*i)->type)->typemask == BT_Float) {
        (*i)->print(out);
        out << ", ";
     } else {
        static const char xyzw[] = { 'x', 'y', 'z', 'w' };
        for (int j = 0; j < FloatDimension(base->typemask); j++) {
           out << '(';
           (*i)->print(out);
           out << ")." << xyzw[j] << ", ";
        }
     }
  }
  out << "-1)";
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtIterType::printForm(std::ostream& out) const
{
    out << "-BrtIter Type ";
    if (base)
        base->printBase(out, 0);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtIterType::printInitializer(std::ostream& out) const
{
  out << "__BRTCreateIter(\"";
  base->printBase(out, 0);
  out << "\",";
  for (unsigned int i=0; i<dims.size(); i++) {
    dims[i]->print(out);
    out << ",";
  }
  out << "-1)";
}


// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtIterType::findExpr( fnExprCallback cb )
{
   ExprVector::iterator i;

   BrtStreamType::findExpr(cb);

   for (i = args.begin(); i != args.end(); i++) {
       *i = (cb)(*i);
       (*i)->findExpr(cb);
   }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
CPUGatherType::CPUGatherType(ArrayType &t,bool copy_on_write) {
  dimension=0;
  raw=false;
  at = &t;
  this->copy_on_write=copy_on_write;
  subtype = at;
  while (subtype->type==TT_Array) {
    dimension +=1;
    subtype = static_cast<const ArrayType*>(subtype)->subType;
  }
}

void CPUGatherType::printSubtype(std::ostream&out,Symbol *name,bool showBase,int level) const {
   subtype->printType(out,name,showBase,level,this->raw); 
}

void CPUGatherType::printType(std::ostream &out, Symbol * name, bool showBase, int level, bool raw) const
{
  this->raw=raw;
  printBefore(out,name,level);
  printAfter(out);	 
}

void CPUGatherType::printBefore(std::ostream & out, Symbol *name, int level) const {
  if (!copy_on_write) {
    Symbol nothing;nothing.name="";

    out << "__BrtArray<";
      
    printSubtype(out,&nothing,true,level);
   
  }else {
    out << "Array"<<dimension<<"d<";
    
    at->printBase(out,level);
        
    const Type * t = at;
    for (unsigned int i=0;i<dimension&&i<3;i++) {
      if (i!=0)
        out <<"	 ";
      out <<", ";
      const ArrayType *a =static_cast<const ArrayType *>(t);
      a->size->print(out);
      t = a->subType;			
    }
  }
  out << "> "; 

  if (name)
    out << *name;
}

void CPUGatherType::printAfter(std::ostream &out) const{
  //nothing happens
  //you fail to obtain anything
  //...
}

BrtKernelType::BrtKernelType(FunctionType *functionType)
  : _functionType(functionType)
{
  unsigned int i;
  extend((*_functionType->getSubType())->dup0());
  for (i=0;i<_functionType->nArgs;i++)
    addArg(_functionType->args[i]->dup0());

  for (i=0;i<nArgs;i++)
  {
    TypeQual q = args[i]->form->getQualifiers();
    if( (q & TQ_Out) == 0 )
    {
      BaseType* b = args[i]->form->getBase();
      b->qualifier |= TQ_Const;
    }
  }
}

BrtKernelType::~BrtKernelType() {
}


