#include "brtvout.h"

#include "ctool.h"
/*
const double SENTINEL_VALUE = -18446321861244485632.0;
std::string DoubleToString(double a) {
   char str[1024];
   sprintf(str,"%lf",a);
   return str;
}
std::string SENTINEL_STRING= DoubleToString(SENTINEL_VALUE);
*/
VoutFunctionType voutFunctions;

FunctionDef *IdentifyVoutFunc(FunctionDef * fd){
   FunctionType * ft = static_cast<FunctionType* >(fd->decl->form);
   bool foundvout=false;
   std::set<unsigned int> vouts;
   for (unsigned int i=0;i<ft->nArgs;++i) {
      if ((ft->args[i]->form->getQualifiers()&TQ_Vout)!=0) {
         foundvout=true;
         vouts.insert(i);
      }
   }
   if (foundvout)
      voutFunctions.insert(VoutFunctionType::
                           value_type(fd->FunctionName()->name,vouts));
   
   return NULL;
}
FunctionDef * TransformHeader (FunctionDef * fd) {
   VoutFunctionType::iterator func
      =voutFunctions.find(fd->FunctionName()->name);
   if (func==voutFunctions.end())
      return NULL;
   FunctionType * ft = static_cast<FunctionType*>(fd->decl->form);
   std::set<unsigned int>::iterator iter =func->second.begin();
   
   for (;iter!=func->second.end();++iter) {
      DeclStemnt * ds = new DeclStemnt (fd->location);
      Decl * tmpVout =ft->args[*iter];
      ft->args[*iter]=ft->args[*iter]->dup();
      ft->args[*iter]->name->name=
         "__"+ft->args[*iter]->name->name+"_stream";
      if (tmpVout->form->type==TT_Stream) {
         tmpVout->form = static_cast<ArrayType*>(tmpVout->form)->subType;
      }
      tmpVout->next=NULL;
      ds->addDecl(tmpVout);
      ds->next = fd->head;
      fd->head = ds;
   }
   Symbol * voutCounter = new Symbol;
   voutCounter->name = "__vout_counter";
   Decl * VoutCounter =  new Decl (voutCounter);
   VoutCounter->form = new BaseType (BT_Float);
   ft->addArg(VoutCounter);
   Symbol * Inf = new Symbol;
   Inf->name = "__inf";
   Decl * InfDecl =  new Decl (Inf);
#ifdef INF_SENTINEL
   ArrayType * streamtype = new ArrayType(TT_Array,NULL);
   streamtype->extend(new BaseType(BT_Float));
   InfDecl->form=streamtype;
#else
   InfDecl->form = new BaseType(BT_Float);
#endif
   ft->addArg(InfDecl);
   return NULL;
}
FunctionDef * TransformVoutToOut (FunctionDef * fd) {
   VoutFunctionType::iterator func
      =voutFunctions.find(fd->FunctionName()->name);
   if (func==voutFunctions.end())
      return NULL;
   FunctionType * ft = static_cast<FunctionType*>(fd->decl->form);
   std::set<unsigned int>::iterator iter =func->second.begin();
   
   for (;iter!=func->second.end();++iter) {
      BaseType * bat = ft->args[*iter]->form->getBase();

      
      bat->qualifier&=(~TQ_Vout);
      bat->qualifier|=TQ_Out;
   }   
   return NULL;
}

Symbol * findSentinel (FunctionType * ft) {
   Symbol * ret = new Symbol;
   ret->name="__inf";
   for (unsigned int i=0;i<ft->nArgs;++i) {
      if (ft->args[i]->name->name=="__inf"){
         ret->entry = mk_paramdecl("__inf",ft->args[i]);
         return ret;
      }
   }
   return ret;
}
Decl * findVoutCounter (FunctionType * ft, bool  replacement=true) {
  if (replacement) {
    static std::map<FunctionType*,Decl*> replacements;
    std::map<FunctionType*,Decl*>::iterator i =replacements.find(ft);
    if (i!=replacements.end()) {
      return (*i).second;
    }else {
      Symbol * voutCounterRW = new Symbol;
      voutCounterRW->name = "__vout_counter_rw";
      Decl * VoutCounter =  new Decl (voutCounterRW);
      replacements[ft]=VoutCounter;
      return VoutCounter;
    }
  }
  for (unsigned int i=0;i<ft->nArgs;++i) {
    if (ft->args[i]->name->name=="__vout_counter")
      return ft->args[i];
  }
  return ft->args[ft->nArgs-2];
}

Decl * MakeVoutCounter(std::string fname,                                 
                            FunctionType * ft,
                            const Location & location) {
  //return new ExpressionStemnt (new FloatConstant(31337,location),location);
  //DeclStemnt * ds = new DeclStemnt (location);
  Decl * VoutCounter = findVoutCounter(ft,true);
  VoutCounter->form = new BaseType (BT_Float);
  VoutCounter->initializer = new Variable(findVoutCounter(ft,false)->name->dup(),location);  
  return VoutCounter;
}
Statement * InitialInfSet (std::string fname,                                 
                           FunctionType * ft,
                           const Location & location) {

   VoutFunctionType::iterator func
      =voutFunctions.find(fname);
   if (func==voutFunctions.end())
      return NULL;
   std::set<unsigned int>::iterator iter =func->second.begin();
   Expression* expression;
   if (iter!=func->second.end()) {
      Symbol * vout_sym=  ft->args[*iter]->name->dup();     
      expression = 
         new AssignExpr (AO_Equal,
                          new Variable(vout_sym,location),
#ifdef INF_SENTINEL
                          new IndexExpr (new Variable(findSentinel(ft),
                                                      location),
                                         new FloatConstant(0.0,location),
                                         location),
#else
                          new Variable(findSentinel(ft),location),
#endif
          location);
      ++iter;
   }else return NULL;
   for (;iter!=func->second.end();++iter) {
   Symbol * vout_sym=  ft->args[*iter]->name->dup();
   Symbol * Why = new Symbol;Why->name = "y";
   expression 
      = new BinaryExpr(BO_Comma,
                       expression,
                       new AssignExpr (AO_Equal,
                                       new Variable(vout_sym,location),
#ifdef INF_SENTINEL
                                      new IndexExpr (new Variable
                                                     (findSentinel(ft),
                                                      location),
                                                     new FloatConstant
                                                      (0.0,
                                                      location),
                                                     location),
#else
                                       new Variable(findSentinel(ft),
                                                         location),
#endif
                                       location),  
                       location);

   }
   return new ExpressionStemnt(expression,location);
}

static FunctionType * pushFunctionType=NULL;
static Statement * PushToIfStatement(Statement * ste) {
   Statement * newstemnt=NULL;   
   FunctionType * ft = pushFunctionType;
   Decl * vout_counter = findVoutCounter(ft);
      if (ste->type==ST_ExpressionStemnt) {
         ExpressionStemnt * es = static_cast<ExpressionStemnt*>(ste);
         if (es->expression->etype==ET_FunctionCall) {
            FunctionCall* fc=static_cast<FunctionCall*>(es->expression);
            if (fc->args.size()==1
                &&fc->function->etype==ET_Variable
                &&static_cast<Variable*>(fc->function)->name->name=="push") {
               if (fc->args[0]->etype!=ET_Variable) {
                  std::cerr<<"Error: ";
                  fc->args[0]->location.printLocation(std::cerr);
                  std::cerr<< " Push called without specific vout stream.\n";
                  return NULL;
               }
               std::string voutname=static_cast<Variable*>
                  (fc->args[0])->name->name;
               Decl * streamDecl=NULL;
               for (unsigned int i=0;i<(unsigned int)ft->nArgs;++i) {
                  if (ft->args[i]->name->name
                      ==std::string("__")+voutname+"_stream") {
                     streamDecl=ft->args[i];
                     break;
                  }
               }
               if (streamDecl==NULL) {
                  std::cerr<<"Error: ";
                  fc->args[0]->location.printLocation(std::cerr);
                  std::cerr<<" Push called on var that is not a vout arg.\n";
                  return NULL;
               }
               Symbol * Eks = new Symbol; Eks->name="x";
               Symbol * counter=vout_counter->name->dup();
               counter->name=vout_counter->name->name;
               counter->entry=mk_paramdecl(counter->name,vout_counter);
               Symbol * stream=new Symbol;
               stream->name="__"+voutname+"_stream";
               stream->entry=mk_paramdecl(stream->name,streamDecl);
               Block * AssignStream = new Block(fc->location);
               AssignStream->add
                  (new ExpressionStemnt
                   (new AssignExpr(AO_Equal,
                                   new Variable(stream,fc->location),
                                   fc->args[0]->dup(),
                                   fc->location),
                    fc->location));
               
               newstemnt=new IfStemnt
                  (new RelExpr(RO_Equal,
                               new FloatConstant(-1,fc->location),
                               new AssignExpr(AO_MinusEql,
                                               new Variable (counter,
                                                             fc->location),
                                              new IntConstant(1,fc->location),
                                              fc->location),
                               fc->location),
                   AssignStream,
                   fc->location);
               
            }
         }
      }
      return newstemnt;
}
bool isFilter (Expression * vout) {
           if (vout&&vout->etype==ET_Constant) {
             Constant * cons = static_cast<Constant*>(vout);
             if (cons->ctype==CT_UInt) {
               if (static_cast<UIntConstant*>(cons)->ulng==1) {
                 return true;
               }
             }
             if (cons->ctype==CT_Int) {
               if (static_cast<IntConstant*>(cons)->lng==1) {
                 return true;
               }
             }
             if (cons->ctype==CT_Char) {
               if (static_cast<CharConstant*>(cons)->ch==1) {
                 return true;
               }
             }
           }
           
           return false;

}
static Expression * TransformExprVoutPush (Expression * expression) {
   FunctionType * ft = pushFunctionType;
   Decl * vout_counter = findVoutCounter(ft);
   if (expression->etype==ET_FunctionCall) {
      FunctionCall* fc=static_cast<FunctionCall*>(expression);
      if (fc->args.size()==1
          &&fc->function->etype==ET_Variable
          &&static_cast<Variable*>(fc->function)->name->name=="push") {
         if (fc->args[0]->etype!=ET_Variable) {
            std::cerr<<"Error: ";
            fc->args[0]->location.printLocation(std::cerr);
            std::cerr<< " Push called without specific vout stream.\n";
            return expression;
         }
         std::string voutname=static_cast<Variable*>
            (fc->args[0])->name->name;
         Decl * streamDecl=NULL;
         for (unsigned int i=0;i<(unsigned int)ft->nArgs;++i) {
            if (ft->args[i]->name->name
                ==std::string("__")+voutname+"_stream") {
               streamDecl=ft->args[i];
               break;
            }
         }
         if (streamDecl==NULL) {
            std::cerr<<"Error: ";
            fc->args[0]->location.printLocation(std::cerr);
            std::cerr<<" Push called on var that is not a vout arg.\n";
            return expression;
         }
         Symbol * Eks = new Symbol; Eks->name="x";
         Symbol * counter=vout_counter->name->dup();
         counter->name=vout_counter->name->name;
         counter->entry=mk_paramdecl(counter->name,vout_counter);
         Symbol * stream=new Symbol;
         stream->name="__"+voutname+"_stream";
         stream->entry=mk_paramdecl(stream->name,streamDecl);
         bool filter=isFilter(streamDecl->form->getQualifiers().vout);
         if (filter) {
           for (unsigned int i=0;i<ft->nArgs;++i) {
             if ((ft->args[i]->form->getQualifiers()&TQ_Vout)!=0||(ft->args[i]->form->getQualifiers()&TQ_Out)!=0){
               if (!isFilter(ft->args[i]->form->getQualifiers().vout)) {
                 filter=false;
               }
             }
           }
         }
         if (filter) {
           expression = new AssignExpr 
             (AO_Equal,
              new Variable(stream,fc->location),
              fc->args[0]->dup(),
              fc->location);                                        
         }else {
           expression = new AssignExpr
             (AO_Equal,
              new Variable(stream,fc->location),
              new TrinaryExpr(new RelExpr(RO_Equal,
                                          new FloatConstant(-1,fc->location),
                                          new AssignExpr(AO_MinusEql,
                                                         new Variable (counter,
                                                                       fc->location),
                                                         new IntConstant(1,
                                                                         fc->location),
                                                         fc->location),
                                          fc->location),
                              fc->args[0]->dup(),
                              new Variable(stream->dup(),fc->location),
                              fc->location),
              fc->location);         
         }
      }
   }
   return expression;
}
FunctionDef* TransformVoutPush(FunctionDef*fd) {
   VoutFunctionType::iterator func
      =voutFunctions.find(fd->FunctionName()->name);
   if (func==voutFunctions.end())
      return NULL;
   FunctionType * ft = static_cast<FunctionType*>(fd->decl->form);
   pushFunctionType = ft;
   fd->findExpr(&TransformExprVoutPush);
   if (0) TransformStemnt(fd,&PushToIfStatement);
   Block * MainFunction= new Block(fd->location);
   Statement * tmp=MainFunction->head;
   MainFunction->head=fd->head;
   fd->head=tmp;
   tmp =MainFunction->tail;
   MainFunction->tail=fd->tail;
   fd->tail=tmp;
   fd->addDecls(MakeVoutCounter(fd->FunctionName()->name,
                                ft,
                                fd->location));
   fd->add(InitialInfSet(fd->FunctionName()->name,
                         ft,
                         fd->location));
   fd->add(MainFunction);
   return NULL;   
}

void transform_vout (TransUnit * tu) {
   tu->findFunctionDef (IdentifyVoutFunc);
   tu->findFunctionDef (TransformHeader);
   tu->findFunctionDef(TransformVoutToOut);
   tu->findFunctionDef(TransformVoutPush);
   //transform push calls
   //transform function calls
   

}




