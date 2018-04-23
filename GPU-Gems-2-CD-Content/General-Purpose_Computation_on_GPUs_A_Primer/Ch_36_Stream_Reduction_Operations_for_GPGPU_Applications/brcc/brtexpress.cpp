/*
 * brtexpress.cpp -- 
 *  the actual code to convert gathers streams and indexof exprs
 */ 
#ifdef _WIN32
#pragma warning(disable:4786)
//debug symbol warning
#endif

#include "brtexpress.h"
#include "splitting/splitting.h"
#include "main.h"
#include <stdio.h>

BrtGatherExpr::BrtGatherExpr(const IndexExpr *expr) 
   : Expression (ET_BrtGatherExpr, expr->location)
{
  const IndexExpr *p;
  std::vector<Expression *> t;
  int i;

  t.push_back(expr->_subscript->dup0());

  // IAB:  Note that we have to reorder the arguments
  for (p = (IndexExpr *) expr->array; 
       p && p->etype == ET_IndexExpr;
       p = (IndexExpr *) p->array)
     t.push_back(p->_subscript->dup0());

  for (i=t.size()-1; i>=0; i--)
     dims.push_back(t[i]);

  base = (Expression *) p->dup0();
  
  assert(base->etype == ET_Variable);
  Variable *v = (Variable *) base;

  assert(v->name);
  assert(v->name->entry);
  assert(v->name->entry->uVarDecl);
  assert(v->name->entry->uVarDecl->form);
  assert(v->name->entry->uVarDecl->form->isArray());
  ArrayType *a = (ArrayType *) v->name->entry->uVarDecl->form;

  const ArrayType *pp;
  ndims = 1;
  for (pp = a; 
       pp->subType && pp->subType->isArray(); 
       pp = (ArrayType *)pp->subType)
     ndims++;

}


void
BrtGatherExpr::print (std::ostream& out) const
{
   // we need the type of the gather variable
   assert(base->etype == ET_Variable);
   Variable *v = (Variable *) base;   
   assert(v->name);
   assert(v->name->entry);
   assert(v->name->entry->IsParamDecl());
   assert(v->name->entry->uVarDecl);
   assert(v->name->entry->uVarDecl->form);
   assert(v->name->entry->uVarDecl->form->isArray());
   ArrayType *a = (ArrayType *) v->name->entry->uVarDecl->form;
   BaseType *b = a->getBase();

   out << "__gather_";
   b->printBase( out, 0 );
   out << "(";
   base->print(out);

   out << ",__gatherindex";
   out << ndims << "((";
   
   if (dims.size() == 1) 
     dims[0]->print(out);
   else if (dims.size() == 2) {
     out << "float2(";
     dims[1]->print(out);
     out << ",";
     dims[0]->print(out);
     out << ")";
   } else if(dims.size() == 3) {
     out << "float3(";
     dims[2]->print(out);
     out << ",";
     dims[1]->print(out);
     out << ",";
     dims[0]->print(out);
     out << ")";
   } else if(dims.size() == 4) {
     out << "float4(";
     dims[3]->print(out);
     out << ",";
     dims[2]->print(out);
     out << ",";
     dims[1]->print(out);
     out << ",";
     dims[0]->print(out);
     out << ")";
   }
   else {
     std::cerr << location
               << "GPU runtimes can't handle gathers greater than 4D.\n";
   }

   out << ")";

   if( ndims == 1 )
   {
     out << ".x";
   }
   else if( ndims == 2 )
   {
     out << ".xy";
   }
   else if( ndims == 3 )
   {
     out << ".xyz";
   }
   else if( ndims == 4 )
   {
     out << ".xyzw";
   }
   else
   {
     // TODO: handle the larger cases
     std::cerr << location
               << "GPU runtimes can't handle gathers greater than 4D.\n";
   }

   if( globals.enableGPUAddressTranslation )
   {
     out << ", __gatherdomainmin_";
     base->print(out);
     out << ", __gatherlinearize_";
     base->print(out);
     out << ", __gathertexshape_";
     base->print(out);
     out << "))";
   }
   else
   {
    out << ", _const_";
    base->print(out);
    out << "_scalebias))";
   }
}

// TIM: adding DAG-building for kernel splitting support
SplitNode* BrtGatherExpr::buildSplitTree( SplitTreeBuilder& ioBuilder )
{
  SplitNode* stream = base->buildSplitTree( ioBuilder );
  std::vector<SplitNode*> indices;
  for( ExprVector::iterator i = dims.begin(); i != dims.end(); i++ )
  {
    indices.push_back( (*i)->buildSplitTree(ioBuilder) );
  }

  return ioBuilder.addGather( stream, indices );
}

BrtStreamInitializer::BrtStreamInitializer(const BrtStreamType *type,
					 const Location& loc )
  : Expression( ET_BrtStreamInitializer, loc), l(loc)
{
  t = (BrtStreamType *) type->dup0();
}

BrtStreamInitializer::~BrtStreamInitializer() {
  // Handled by global type list
  //   delete t;
}

Expression *
BrtStreamInitializer::dup0() const {
  return new BrtStreamInitializer(t, l);
}

void
BrtStreamInitializer::print(std::ostream& out) const {
  t->printInitializer(out);
}



// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BrtIndexofExpr::BrtIndexofExpr( Variable *operand, const Location& l )
         : Expression( ET_BrtIndexofExpr, l )
{
    expr = operand;
    type = new BaseType(BT_Float4);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BrtIndexofExpr::~BrtIndexofExpr()
{
    // delete sizeofType;
    delete expr;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Expression*
BrtIndexofExpr::dup0() const
{
    BrtIndexofExpr *ret ;
    ret = new BrtIndexofExpr(static_cast<Variable*>(expr->dup()), location);
    ret->type = type;
    return ret;
}

// Ian:  I'm so sorry about this...
extern bool horrible_horrible_indexof_hack;

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtIndexofExpr::print(std::ostream& out) const
{

  // Ian:  I'm so sorry about this...
  if (horrible_horrible_indexof_hack) {
    std::string bak = expr->name->name;
    out << "indexof(";
    expr->print(out);
    out << ")";
  } else {
    std::string bak = expr->name->name;
    expr->name->name="__indexof_"+bak;
    expr->print(out);
    expr->name->name=bak;
  }

#ifdef    SHOW_TYPES
    if (type != NULL)
    {
        out << "/* ";
        type->printType(out,NULL,true,0);
        out << " */";
    }
#endif
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BrtIndexofExpr::findExpr( fnExprCallback cb )
{
   if (expr != NULL) {
     Expression* e=(cb)(expr);
     assert (e->etype==ET_Variable);
     expr = (Variable *) e;
     expr->findExpr(cb);
   }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
// TIM: adding DAG-building for kernel splitting support
SplitNode* BrtIndexofExpr::buildSplitTree( SplitTreeBuilder& ioBuilder )
{
  std::string variableName = expr->name->name;
  return ioBuilder.addIndexof( variableName );
}
