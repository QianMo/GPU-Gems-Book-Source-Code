/*
 * b2creduce.cpp
 * utility function for making base and combine code for reduction variables
 * and functions
 */
#ifdef _WIN32
#pragma warning(disable:4786)
//the above warning disables visual studio's annoying habit of warning when using the standard set lib
#endif
#include "ctool.h"
#include <map>
#include <set>
#include <string>
#include <vector>
 
// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool reduceNeeded (const FunctionDef * fd) {
   
   if (fd->decl->isReduce()) {
      return false;
   }
   
   bool ret =false;
   Type * form = fd->decl->form;
   assert (form->isFunction());
   FunctionType* func = static_cast<FunctionType *>(form);
   for (unsigned int i=0;i<func->nArgs;++i) {
      if (func->args[i]->isReduce())
         return true;
   }
   return ret;
}

std::map<std::string,Decl *> reductionBools;
static std::set <Expression*> searchedexpressions;
typedef std::map<std::string,Expression*>::value_type reducenameval;
static std::string functionmodifier;
static Expression * (*ModifyAssignExpr)(AssignExpr*ae)=0;
static Expression * (*ModifyFunctionCall)(FunctionCall *, 
                                          unsigned int, 
                                          unsigned int)=0;
static Expression* (*ModifyFutureReduceOperator)(Expression*)=0;

static Expression* DoNothing(Expression * fc) {
   return fc;
}
static Expression* ConvertToNop(Expression*fc) {
   Location l (fc->location);
   delete fc;
   return new UIntConstant (0,l);
}
const std::string dual_reduction_arg="__partial_";

static Expression *  DemoteAssignExpr (AssignExpr * ae) {
   ae->aOp=AO_Equal;
   return ae;   
}
Expression *ChangeVariable(Expression * lv) {
   if (lv->etype==ET_Variable) {
      Variable * v = static_cast<Variable*>(lv);
      v->name->name=dual_reduction_arg+v->name->name;
   }
   return lv;
}
static Expression *  DuplicateLVal(AssignExpr * ae) {
   Expression * rval = ae->rValue();
   ae->_rightExpr=ChangeVariable(ae->lValue()->dup());
   delete rval;
   return ae;
}
Variable * ReduceVar (Decl *reducebool,const Location &l) {
   Variable * ret = new Variable(reducebool->name->dup(),l);
   ret->name->entry=new SymEntry(VarDeclEntry,ret->name->name,reducebool);
   ret->name->entry->uVarDecl = reducebool->dup();
   return ret;
}
Expression * FirstQuestionColon (std::string reducename,
                                 Expression * ifFirst,
                                 Expression * ifFuture) {
   searchedexpressions.insert(ifFirst);
   searchedexpressions.insert(ifFuture);
   Location l (ifFirst->location);
   Decl * reducebool =reductionBools[reducename];
   return 
      new TrinaryExpr(ReduceVar(reducebool,l),
                      new BinaryExpr(BO_Comma,
                                     ifFirst,
                                     new AssignExpr(AO_Equal,
                                                    ReduceVar(reducebool,l),
                                                    new IntConstant(0,l),
                                                    l),
                                     l),
                      new BinaryExpr(BO_Comma,
                                     ifFuture,
                                     new IntConstant(0,l),
                                     l),
                      l);
}

static Expression* CombineReduceStream(FunctionCall *func,
                                       unsigned int reduce,
                                       unsigned int stream) {
   Expression * tmp=func->args[stream];
   func->args[stream]=ChangeVariable(func->args[reduce]->dup());
   delete tmp;
   return func;
}
static Expression * ArrayAssign (Expression * lval, Expression* rval, const Location &l, Type * t) {
   //now we have the job of making a comma separated expression that assigns all the values in type t
   //prepare ship for ludicrous speed.
   std::vector <Expression*>bounds;
   Type * orig=t;
#if 0
   while (t->type==TT_Array) {
      ArrayType * at= static_cast<ArrayType*>(t);
      if (!at->size) {
         std::cerr << "error: ";
         l.printLocation(std::cerr);
         std::cerr << " All bounds must be specified.\n";
         exit(1);
      }
      bounds.push_back(at->size);
      t=at->subType;
   };
   Expression * cur = NULL;
   while (!bounds.empty()) {
      Expression * size = bounds.back();bounds.pop_back();
      if (!cur) cur =size;
      else {
         cur = new BinaryExpr(BO_Mult,size->dup(),cur->dup(),l);
      }
   }
#endif
   Symbol * mymemcpy = new Symbol();mymemcpy->name="memcpy";
   mymemcpy->entry = new SymEntry(FctDeclEntry);
   FunctionCall * fc =  new FunctionCall(new Variable(mymemcpy,l),l);
   fc->addArg(lval);
   fc->addArg(rval);
   fc->addArg(new SizeofExpr(orig,l));
   return fc;
}
static Expression* FunctionCallToAssign(FunctionCall *func,
                                        unsigned int reduce,
                                        unsigned int stream) {
      Location l (func->location);
      Expression * mreduce=func->args[reduce]->dup();
      Expression * mstream=func->args[stream]->dup();
      delete func;
      if (mreduce->etype==ET_Variable) {
         Variable * vreduce = static_cast<Variable*>(mreduce);
         if (vreduce->name->entry){
            if (vreduce->name->entry->uVarDecl) {
               Type * t = vreduce->name->entry->uVarDecl->form;
               if (t) {
                  if (t->type==TT_Array) {
                     return ArrayAssign (mreduce,
                                         mstream,
                                         l,
                                         static_cast<ArrayType*>(t));
                  }
               }
            }

         }
      }
      return new AssignExpr(AO_Equal,
                            mreduce,
                            mstream,
                            l);   
}

static Expression * ConvertPlusTimesGets(Expression * e) {
   if (e->etype==ET_BinaryExpr
       &&static_cast<BinaryExpr*>(e)->bOp==BO_Assign
       &&searchedexpressions.find(e)==searchedexpressions.end()) {
      
      AssignExpr * ae = static_cast<AssignExpr*>(e);
      if (ae->lValue()->etype==ET_Variable) {
         Variable *v= static_cast<Variable*>(ae->lValue());
         SymEntry * s = v->name->entry;
         if (s&&s->uVarDecl) {
            if (s->uVarDecl->form){ 
               if (s->uVarDecl->isReduce()) {
                  AssignExpr * ab = static_cast<AssignExpr*>(ae->dup());
                  std::string assignname(v->name->name);
                  return FirstQuestionColon(assignname,
                                            (*ModifyAssignExpr)(ab),
                                            (*ModifyFutureReduceOperator)(ae));
                 
               }
            }
         }
      }
   }
   return e;
}
static Expression * ConvertReduceToGets(FunctionCall* func, FunctionType * type) {

   Expression * mreduce=NULL;
   Expression * mstream=NULL;
   unsigned int reduceloc=0,streamloc=0;
   std::string reducename;
   for (unsigned int i=0;i<type->nArgs;++i) {
      if (type->args[i]->isReduce()) {
         if (func->args[i]->etype==ET_Variable) {
            Variable * v = static_cast<Variable*>(func->args[i]);
            reducename=v->name->name;
            mreduce = func->args[i];
            reduceloc=i;

         }
      }
      if (type->args[i]->isStream()) {
         mstream = func->args[i];         
         streamloc=i;
      }
   }
   if (mreduce&&mstream
       &&searchedexpressions.find(func)==searchedexpressions.end()) {
      FunctionCall * fcdup=static_cast<FunctionCall*>(func->dup());
      return FirstQuestionColon(reducename,
                                (*ModifyFunctionCall)(func,
                                                      reduceloc,
                                                      streamloc),
                                (*ModifyFutureReduceOperator) (fcdup));
   }
   return func;
}


Expression *ChangeFirstReduceFunction (Expression * e) {
   if (e->etype==ET_FunctionCall) {
      FunctionCall * fc = static_cast<FunctionCall*>(e);
      Expression * k =fc->function;
      if (k->etype==ET_Variable) {
         Variable * callname =static_cast<Variable*>(k);
         Symbol * sym = callname->name;
         SymEntry * s= sym->entry;
         if (s&&s->uVarDecl) {
            if (s->uVarDecl->form){ 
               if (s->uVarDecl->isReduce()) {
                  if (s->uVarDecl->form->type==TT_Function) {
                     FunctionType* func = 
                        static_cast<FunctionType*>(s->uVarDecl->form);
                     return ConvertReduceToGets(fc,func);
                  
                  }
               }else if (sym->name.find(functionmodifier)==std::string::npos){
                  std::string tmp = sym->name;
                  unsigned int nOrigArgs=static_cast<FunctionType*>(s->uVarDecl->form)->nArgs;                                 
                  for (unsigned int i=0;i<nOrigArgs;++i) {
                     if (fc->args[i]->etype==ET_Variable) {
                        Variable * v = static_cast<Variable*>(fc->args[i]);                        
                        if (v->name->entry&&v->name->entry->uVarDecl) {
                           if (v->name->entry->uVarDecl->isReduce()) {
                              unsigned int inner=sym->name.find("_cpu_inner");
                              if (sym->name.find(functionmodifier)==std::string::npos) {
                                 sym->name = tmp.substr(0,inner)+functionmodifier+tmp.substr(inner);
                              }
                              if (functionmodifier.find("combine")!=std::string::npos) {
                                 Symbol * partial =v->name->dup();
                                 partial->name=std::string("__partial_")+partial->name;
                                 
                                 fc->args.insert(fc->args.begin()+nOrigArgs,new Variable(partial,fc->location));
                              }
                              // change kernels taking in a reduction
                              // to be the appropriate combine function
                              // other kernels might not even have such a 
                              // construct as they may not be reduce funcs
                           }
                        }
                     }
                  }

               }
            }
         }
      }
   }
   return e;
}


void FindFirstReduceFunctionCall (Statement * s) {

   s->findExpr(&ChangeFirstReduceFunction);
   s->findExpr(&ConvertPlusTimesGets);
}
void addReductionBools (FunctionDef *fDef) {
   FunctionType * t = static_cast<FunctionType*>(fDef->decl->form);
   for (unsigned int i=0;i<t->nArgs;++i) {
      if (t->args[i]->isReduce()) {
         DeclStemnt * ds = new DeclStemnt(fDef->location);
         ds->next = fDef->head;
         fDef->head=ds;
         Decl * newchar = new Decl(new BaseType(BT_Char));
         Symbol * s = new Symbol;
         s->name = "__first_reduce_"+t->args[i]->name->name;
         newchar->name=s;
         newchar->initializer = new IntConstant(1,fDef->location);
         ds->addDecl(newchar);
         reductionBools[t->args[i]->name->name]=newchar;
      }
   }
}
void BrookReduce_ConvertKernel(FunctionDef *fDef) {
   reductionBools.clear();
   searchedexpressions.clear();
   addReductionBools(fDef);
   functionmodifier="__base";
   ModifyAssignExpr= &DemoteAssignExpr;
   ModifyFunctionCall=&FunctionCallToAssign;
   ModifyFutureReduceOperator=&DoNothing;
   fDef->findStemnt(&FindFirstReduceFunctionCall);
   //   fDef->decl->name = fDef->decl->name->dup();
   fDef->decl->name->name+=functionmodifier;
}

void BrookCombine_ConvertKernel(FunctionDef *fDef) { 
   reductionBools.clear();
   searchedexpressions.clear();
   addReductionBools(fDef);
   functionmodifier="__combine";
   ModifyAssignExpr= &DuplicateLVal;
   ModifyFunctionCall=&CombineReduceStream;
   ModifyFutureReduceOperator=&ConvertToNop;
   fDef->findStemnt(&FindFirstReduceFunctionCall);
   //   fDef->decl->name = fDef->decl->name->dup();
   fDef->decl->name->name+=functionmodifier;
   if (1) {
      FunctionType*  func=static_cast<FunctionType*>(fDef->decl->form);
      std::vector <Decl *>AdditionalDecl;
      for (unsigned int i=0;i<func->nArgs;++i) {
         if (func->args[i]->isReduce()) {
            AdditionalDecl.push_back(func->args[i]->dup());
            AdditionalDecl.back()->name->name=
               dual_reduction_arg+
               AdditionalDecl.back()->name->name;
         }
      }
      for (unsigned int j=0;j<AdditionalDecl.size();++j) {
         func->addArg(AdditionalDecl[j]);
      }
   }
}



