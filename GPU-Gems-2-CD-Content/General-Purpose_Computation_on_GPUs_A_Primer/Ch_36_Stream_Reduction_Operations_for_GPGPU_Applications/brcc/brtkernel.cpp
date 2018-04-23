/*
 * brtkernel.cpp
 *
 *      Classes reflecting the body of Brook kernels for the different
 *      backends.  Each one knows how to build itself from a function
 *      definition and then how to emit C++ for itself.
 */
#ifdef _WIN32
#pragma warning(disable:4786)
//the above warning disables visual studio's annoying habit of warning when using the standard set lib
#endif
#include <stdio.h>
//someone removed the above header...it is needed for linux
#include <cstring>
#include <cassert>
#include <sstream>

#include "brtkernel.h"
#include "brtexpress.h"
#include "codegen.h"
#include "main.h"
#include "splitting/splitting.h"

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
std::string whiteout (std::string s) {
   for (unsigned int i=0;i<s.length();++i) {
      s[i]=' ';
   }
   return s;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
// checks if a function is a gather stream
extern bool recursiveIsGather(Type*);

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
std::ostream&
operator<< (std::ostream& o, const BRTKernelCode& k) {
   if (k.standAloneKernel())
      k.printCode(o);
   else
      k.onlyPrintInner(o);
   return o;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool BRTKernelCode::standAloneKernel()const {
   return fDef->returnsVoid();
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BRTGPUKernelCode::BRTGPUKernelCode(const FunctionDef& _fDef)
   : BRTKernelCode(_fDef) {
  fDef->findExpr(ConvertGathers);
}

static Variable * NewGatherArg (Variable * v) {
  Symbol * s = new Symbol;
  s->name = "_const_"+v->name->name+"_scalebias";
  return new Variable(s,v->location);
}

static Variable * NewAddressTransArg (Variable * v, const char* prefix) {
  Symbol * s = new Symbol;
  s->name = prefix+v->name->name;
  return new Variable(s,v->location);
}

// TIM: HACK
int getGatherStructureSamplerCount( Type* form );

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
// This function prints the code of an internally callable kernel
// from within another kernel.
void BRTGPUKernelCode::printInnerCode (std::ostream&out) const {
  unsigned int i;
  std::string myvoid("void  ");
  FunctionType * ft = static_cast<FunctionType *>(fDef->decl->form);

  ft->printBase(out,0);
  out << fDef->decl->name->name<< " (";
  std::string blank (whiteout(myvoid + fDef->decl->name->name +" ("));

  for (i=0;i<ft->nArgs;++i) {
    if (i!=0) {
      out << ","<<std::endl<< blank;
    }
    
    Symbol * nam = ft->args[i]->name;
    Type * t = ft->args[i]->form;
    if (recursiveIsGather(t)) {
      out << "_stype "<<nam->name <<"[";
      // TIM: HACK:
      out << getGatherStructureSamplerCount(t);
      out << "],"<<std::endl;
      
      if( globals.enableGPUAddressTranslation ) {
        out << "float4 __gatherlinearize_" << nam->name;
        out << ", float4 __gathertexshape_" << nam->name;
        out << ", float4 __gatherdomainmin_" << nam->name;
      } else {
        out << blank << "float4 _const_"<<nam->name<<"_scalebias";
      }
    } else {
      if ((ft->args[i]->form->getQualifiers() & TQ_Reduce) != 0) {
        out << "inout ";
      }
      if (ft->args[i]->isStream()) {
        t = static_cast<ArrayType *>(t)->subType;
      }
      t->printType(out,nam, true,0);
    }
  }
  std::set<unsigned int>::iterator iter=
    FunctionProp[ fDef->decl->name->name].begin();
  std::set<unsigned int>::iterator iterend =
    FunctionProp[ fDef->decl->name->name].end();
  for (;iter!=iterend;++iter,++i) {
    if (i!=0)
      out << ","<<std::endl<<blank;
    out << "float4 __indexof_"<<ft->args[*iter]->name->name;
  }
  out << ")"<<std::endl;
  fDef->Block::print(out,0);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
// This function converts gathers into scale and bias expressions
// This function converts function calls' gathers into two args.
// This function adds the indexof items to function calls requriing indexof
Expression *
BRTGPUKernelCode::ConvertGathers (Expression *expr) {
  BrtGatherExpr *gather;

  /* Check function calls inside of kernels */
  // TIM: we don't convert the argumenst to a call when
  // we use the DAG-building kernel-splitting codegen
  if (expr->etype == ET_FunctionCall && !globals.enableKernelSplitting) {

     //now we have to convert gathers that are passed into functions
     FunctionCall * fc = static_cast<FunctionCall*>(expr);
     if (fc->function->etype==ET_Variable) {
        Variable * function = static_cast<Variable*>(fc->function);
        if (function->name->entry && function->name->entry->uVarDecl) {
           if (function->name->entry->uVarDecl->isKernel() &&
               !function->name->entry->uVarDecl->isReduce()) {

              std::set<unsigned int>::iterator iter=
                 FunctionProp[function->name->name].begin();
              std::set<unsigned int>::iterator iterend =
                 FunctionProp[function->name->name].end();

              for ( ; iter!=iterend; ++iter) {
                 if (fc->args[*iter]->etype != ET_Variable) {
                    std::cerr<<"Error: ";
                    fc->args[*iter]->location.printLocation(std::cerr);
                    std::cerr<< "Argument "<<*iter+1<<" not a stream where";
                    std::cerr<< "indexof used in subfunction";
                 } else {
                    Variable * v = static_cast<Variable*>(fc->args[*iter]);
                    if (v->name->entry &&
                        v->name->entry->uVarDecl){
                       if (v->name->entry->uVarDecl->isStream()) {
                          Decl * indexofDecl
                             = new Decl(new BaseType(BT_Float4));

                          Symbol * indexofS = new Symbol;
                          indexofS->name = "__indexof_"+v->name->name;
                          indexofS->entry = mk_vardecl(indexofS->name,
                                                       indexofDecl);
                          fc->addArg(new Variable(indexofS,v->location));
                       } else {
                          std::cerr<< "Error: ";
                          v->location.printLocation(std::cerr);
                          std::cerr<<" Argument "<<*iter+1<<" not a stream";
                          std::cerr<< "where indexof used in subfunction";
                       }
                    }
                 }
              }

              int i;
              for (i=0;i<fc->nArgs();++i) {
                 if (fc->args[i]->etype==ET_Variable){
                    Variable * v = static_cast<Variable*>(fc->args[i]);
                    if (v->name->entry&&v->name->entry->uVarDecl) {
                       if(recursiveIsGather(v->name->entry->uVarDecl->form)) {
                          if( globals.enableGPUAddressTranslation )
                          {
                            ++i;
                            fc->args.insert(fc->args.begin()+i,NewAddressTransArg(v,"__gatherlinearize_"));
                            ++i;
                            fc->args.insert(fc->args.begin()+i,NewAddressTransArg(v,"__gathertexshape_"));
                            ++i;
                            fc->args.insert(fc->args.begin()+i,NewAddressTransArg(v,"__gatherdomainmin_"));
                          }
                          else
                          {
                            ++i;
                           fc->args.insert(fc->args.begin()+i,NewGatherArg(v));
                          }
                       }
                    }
                 }
              }
           }
        }
     }
     return expr;
  }


  /* Convert gather expressions: a[i][j] */
  if (expr->etype == ET_IndexExpr) {

    if (globals.verbose) {
      std::cerr << "Found Index Expr: " << expr << std::endl;
    }

    // Check to see if the expression is from a gather stream
    IndexExpr *p = (IndexExpr *) expr;
    for (p = (IndexExpr *) p->array;
         p && p->etype == ET_IndexExpr;
         p = (IndexExpr *) p->array);

    // If things have gone horribly wrong
    if (!p) return expr;
    if (p->etype != ET_Variable) return expr;

    Variable *v = (Variable *) p;
    assert(v->name->entry);

    if (v->name->entry->type != ParamDeclEntry)
      return expr;

    // XXX Daniel: BrtGatherExpr asserts that it is
    //             indeed an array, not a TT_Stream
    if (v->name->entry->uVarDecl)
      if (v->name->entry->uVarDecl->form)
        if (v->name->entry->uVarDecl->form->type!=TT_Array)
          return expr;
    gather = new BrtGatherExpr((IndexExpr *) expr);

    // IAB: XXX For some reason I can't delete expr!!!
    //delete expr;
    return gather;
  }
  return expr;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTFP30KernelCode::printCode(std::ostream& out) const
{
   printCodeForType(out, CODEGEN_FP30);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTFP40KernelCode::printCode(std::ostream& out) const
{
   printCodeForType(out, CODEGEN_FP40);
}
// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTARBKernelCode::printCode(std::ostream& out) const
{
   printCodeForType(out, CODEGEN_ARB);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTPS20KernelCode::printCode(std::ostream& out) const
{
   printCodeForType(out, CODEGEN_PS20);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTPS2BKernelCode::printCode(std::ostream& out) const
{
   printCodeForType(out, CODEGEN_PS2B);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTPS2AKernelCode::printCode(std::ostream& out) const
{
   printCodeForType(out, CODEGEN_PS2A);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTPS30KernelCode::printCode(std::ostream& out) const
{
   printCodeForType(out, CODEGEN_PS30);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BRTGPUKernelCode::printCodeForType(std::ostream& out,
                                   CodeGenTarget target) const
{
   FunctionType *fType;
   std::ostringstream wrapOut;
   char *fpcode;

   if( globals.enableKernelSplitting )
   {
    // TIM: insert attempt to build a split tree
    CodeGen_SplitAndEmitCode( fDef, target, out );
   }
   else
   {
    fDef->Block::print(wrapOut, 0);
    if (globals.verbose) {
        std::cerr << "***Wrapping***\n";
        fDef->decl->print(std::cerr, true);
        std::cerr << std::endl << wrapOut.str() << "\n**********\n";
    }

    assert (fDef->decl->form->type == TT_Function);
    fType = (FunctionType *) fDef->decl->form;
    fpcode = CodeGen_GenerateCode(fType->subType,
                                  fDef->FunctionName()->name.c_str(),
                                  fType->args, fType->nArgs,
                                  wrapOut.str().c_str(), target);
    out << fpcode;
    free(fpcode);
   }
}

#if 0
// These functions no longer seem to be used

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
// This function prints out the type of a variable from a stream passed in
// it may optionally add indirection.
static void printType (std::ostream & out,
                       Type * t,
                       bool addIndirection,
                       std::string name ="") {
  Symbol sym;
  sym.name=name;
  if (addIndirection)
    sym.name=std::string("*")+sym.name;
  t->printBase(out,0);
  t->printBefore(out,&sym,0);
  t->printAfter(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
static std::string tostring(unsigned int i) {
  char c[1024];
  c[1023]=0;
  sprintf(c,"%d",i);
  return std::string(c);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
static Symbol getSymbol(std::string in) {
  Symbol name;
  name.name = in;
  return name;
}
#endif


// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool recursiveIsArrayType(Type * form) {
  if ((form->getQualifiers()&TQ_Reduce)!=0) {
    return form->type==TT_Array;
  }
  return form->type==TT_Stream
    && (static_cast<ArrayType*>(form)->subType->type==TT_Array);
}


// Ian:  I'm so sorry about this...
bool horrible_horrible_indexof_hack;

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
// This function prints the function that is called from the inner loop
// of the CPU. This function may also be called from within other kernels.
void BRTCPUKernelCode::printInnerCode (std::ostream & out) const {

  Symbol enhanced_name;
  std::string myvoid("void  ");
  unsigned int i;
  
  FunctionDef *fDef = (FunctionDef *) (this->fDef->dup());
 
  Brook2Cpp_ConvertKernel(fDef);

  FunctionType *ft = static_cast<FunctionType *> (fDef->decl->form);

  enhanced_name.name = "__"+fDef->decl->name->name+"_cpu_inner";

  ft->printBase(out,0);
  out << " ";
  ft->printBefore(out,&enhanced_name,0);
  out << "(";

  std::string white = whiteout ("void "+enhanced_name.name+"(");

  // Print the function arguments
  for (i=0;i<ft->nArgs;++i) {
    if (i!=0) {
      out << "," << std::endl << white;
    }
    Type * t = ft->args[i]->form;

    Symbol * nam = ft->args[i]->name;
    if (0) {// Uses addressable to get copy-semantics between kernels
    if (ft->args[i]->form->isArray()||((ft->args[i]->form->getQualifiers())
                                       & (TQ_Reduce | TQ_Out))!=0) {
      nam->name = "&" + nam->name;
      if (ft->args[i]->form->isArray())
        out << "const ";
    }
    
    // if it is a stream, loose the "<>" 
    if (ft->args[i]->isStream()) {
      nam->name="> "+nam->name;
      t = static_cast<ArrayType *>(t)->subType;
      out << "Addressable <";
    }
    }else {
    nam->name = "&" + nam->name;
    
    // if it is a stream, loose the "<>" 
    if (ft->args[i]->isStream()) {
      t = static_cast<ArrayType *>(t)->subType;
    }

    // if it is not an out/reduce arg, make it const
    if ((ft->args[i]->form->getQualifiers() 
         & (TQ_Reduce | TQ_Out)) == 0) {
      out << "const ";
    }

    }
    // if it is not an out/reduce arg, make it const NO LONGER 
    /*
    if ((ft->args[i]->form->getQualifiers() 
         & (TQ_Reduce | TQ_Out)) == 0) {
      out << "const ";
      }*/
     
    // if it is a gather
    if (recursiveIsGather(t)) {
      CPUGatherType gt(*(ArrayType *)t, false);
      gt.printType(out, nam, true, 0);
      continue;
    }

    // print the argument
    t->printType(out, nam, true,0);    
  }

  out << ")"<<std::endl;

  // Ian:  I'm so sorry about this...
  horrible_horrible_indexof_hack = true;  

  // Print the body
  fDef->Block::print(out,0);

  horrible_horrible_indexof_hack = false;
}

void BRTCPUKernelCode::onlyPrintInner(std::ostream& out) const {
  printInnerCode (out);
}
static void printFetchElement(std::ostream &out, Decl * decl) {
   Type * t = decl->form;
   t = static_cast<ArrayType *>(t)->subType;
   out << "(";
   t->printType(out, NULL, true, 0,true);
   out << "*) __k->FetchElem("
       << decl->name->name
       << ")";
}
// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void BRTCPUKernelCode::printCode(std::ostream& out) const
{

  Symbol enhanced_name;
  enhanced_name.name = "__"+fDef->decl->name->name + "_cpu";
  std::string myvoid("void  ");
  unsigned int i;
  std::string indent("  ");

  // Print the core function
  printInnerCode(out);

  // Print the function decl
  out << myvoid;
  fDef->decl->form->printBefore(out,&enhanced_name,0);
  out << "(::brook::Kernel *__k, const std::vector<void *>&args)" 
      << std::endl
      << "{" 
      << std::endl << indent;

  FunctionDef *fDef = (FunctionDef *) (this->fDef->dup());
 
  Brook2Cpp_ConvertKernel(fDef);

  FunctionType *ft = static_cast<FunctionType *> (fDef->decl->form);

  // Print the function arguments
  for (i=0;i<ft->nArgs;++i) {
    Symbol * nam = ft->args[i]->name;
    Type * t = ft->args[i]->form;
    
    nam->name = "arg_" + nam->name;

    // Print the arg declaration
    if (ft->args[i]->isStream()) {
      out << "::brook::StreamInterface ";
    } else if (recursiveIsGather(t)) {
      CPUGatherType gt(*(ArrayType *)t, false);
      gt.printType(out, NULL, true, 0);    
    } else {
      t->printType(out, NULL, true, 0);    
    }
    
    out << "*" << nam->name << " = (";

    // Print the arg initialization
    if (ft->args[i]->isStream()) {
      out << "::brook::StreamInterface ";
    } else if (recursiveIsGather(t)) {
      CPUGatherType gt(*(ArrayType *)t, false);
      gt.printType(out, NULL, true, 0);    
    } else {
      t->printType(out, NULL, true, 0);    
    }

    out << "*) args[" << i << "];" << std::endl << indent;
  }
  out << std::endl << indent;

  // Print the do while loop
  out << "do {" << std::endl;
  for (i=0;i<ft->nArgs;++i) {
     if (ft->args[i]->isStream()) {
       if ((ft->args[i]->form->getQualifiers()&(TQ_Reduce|TQ_Out))!=0){
           Type * t = static_cast<ArrayType *>(ft->args[i]->form)->subType;
           out << indent << indent<< "Addressable <";
           Symbol sym;sym.name="> __out_"+ft->args[i]->name->name;
           t->printType(out, &sym, true, 0,false);
           out << "(";
           printFetchElement(out,ft->args[i]);
           out <<");"<<std::endl; 
        }
     }
  }
  std::string white = whiteout (enhanced_name.name + "_inner (");
  
  out << indent << indent <<enhanced_name.name 
      << "_inner (";
  
  for (i=0;i<ft->nArgs;++i) {
    if (i!=0) out << "," 
                  << std::endl 
                  << indent 
                  << indent 
                  << white;

    if (ft->args[i]->isStream()) {
      if ((ft->args[i]->form->getQualifiers()&(TQ_Reduce|TQ_Out))!=0){
          out << "__out_"+ft->args[i]->name->name;
       }else{
          out << "Addressable <";
          static_cast<ArrayType*>(ft->args[i]->form)->subType->printType(out, NULL, true, 0,false);
          out<<">(";
          printFetchElement(out,ft->args[i]);
          out << ")";
       }
    } else
      out << "*"<<ft->args[i]->name->name;
  }
  out << ");"
      << std::endl;
  for (i=0;i<ft->nArgs;++i) {
     if (ft->args[i]->isStream()) {
       if ((ft->args[i]->form->getQualifiers()&(TQ_Reduce|TQ_Out))!=0){
           out << indent << indent<<"*reinterpret_cast<";
           static_cast<ArrayType*>(ft->args[i]->form)->subType->printType(out,NULL,true,0,true);
           out <<"*>(__out_"<<ft->args[i]->name->name<<".address)"<< " = __out_"<<ft->args[i]->name->name<<".castToArg(*reinterpret_cast<";
           static_cast<ArrayType*>(ft->args[i]->form)->subType->printType(out,NULL,true,0,true);
           out <<"*>(__out_"<<ft->args[i]->name->name<<".address));"<< std::endl;
        }
     }
  }  
  out << indent<< "} while (__k->Continue());"
      << std::endl 
      << "}" 
      << std::endl;


}
