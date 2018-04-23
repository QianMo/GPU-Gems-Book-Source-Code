#ifdef _WIN32
#pragma warning(disable:4786)
#endif
#include <assert.h>
#include "ps2arb_intermediate.h"

using namespace ps2arb;

Register::Register (string reg, string mask, string negate)
  : string(reg), swizzle(mask), negate(negate)
{}

DefineConstantRegister::DefineConstantRegister (const Register & reg,
                                                float a, float b, 
                                                float c, float d)
  :  Declaration(reg) 
{
	Symbol s;

	x=a; y=b; z=c; w=d;
	
	s.Set(s.type,s.registerindex,x,y,z,w);
	iLanguage->SpecifySymbol(reg,s);
}	

void IntermediateLanguage::SpecifySymbol(std::string name, const Symbol &sym) {
	std::map<string,Symbol>::iterator i = symbolTable.find(name);
	if (i==symbolTable.end()) {
		symbolTable[name]=sym;
	}else {
		(*i).second.Specify(sym);
	}
}
Symbol IntermediateLanguage::findSymbol(std::string name) {
	std::map<string,Symbol>::iterator i = symbolTable.find(name);
	if (i==symbolTable.end()) {
		return Symbol();
	}else {
		return (*i).second;
	}
}

bool IntermediateLanguage::makeTemporaries(std::vector <Symbol> &ret,
                                           int num,int maxtemporaries) {
   std::vector<bool>temporaries;
   temporaries.insert(temporaries.begin(),maxtemporaries,false);
   std::map<string,Symbol>::iterator i = symbolTable.begin();
   for (;i!=symbolTable.end();++i) {
      if ((*i).second.type==Symbol::TEMP) {
         assert ((*i).second.registerindex < (int) temporaries.size());
         temporaries[(*i).second.registerindex] = true;
      }
   }
   assert (temporaries.size()== (unsigned int) maxtemporaries);
   for (int j=0;j<maxtemporaries&&num>0;++j) {
      if (temporaries[j]==false) {
         Symbol s;
         s.Set(Symbol::TEMP,j);
         ret.push_back(s);
         num--;
      }
   }
   return num>0;
}

void Symbol::Specify (const Symbol &sym) {
	if (sym.type!=UNKNOWN) {
		if (type==UNKNOWN) {
			type= sym.type;
			registerindex=sym.registerindex;
		}else {
			assert (type==sym.type);
			assert (registerindex==sym.registerindex);
		}
	}
	if (sym.properties.texturetarget!=UNSPECIFIED) {
		if (properties.texturetarget==UNSPECIFIED) {
			properties.texturetarget=sym.properties.texturetarget;
		}else {
			assert (properties.texturetarget==sym.properties.texturetarget);
		}
	}
	if (sym.properties.validfloats) {
		properties.x=sym.properties.x;		properties.y=sym.properties.y;		properties.z=sym.properties.z;		properties.w=sym.properties.w;
		properties.validfloats=true;
	}
	if (sym.lineno!=-1&&(lineno==-1||sym.lineno<lineno)) {
		lineno=sym.lineno;//get the first occurance;
	}
}
Symbol::Symbol () {type=UNKNOWN;properties.texturetarget=UNSPECIFIED;lineno=-1;properties.x=properties.y=properties.z=properties.w=-999;}
void Symbol::Set (TYPE type, unsigned int index) {
	properties.validfloats=false;
	registerindex=index;
	properties.texturetarget=UNSPECIFIED;
	this->type=type;
	lineno=iLanguage->nextInstructionNumber();
}
void Symbol::Set(TYPE type, unsigned int index,float x,float y, float z, float w) {
	properties.validfloats=true;
	properties.x=x;	properties.y=y;	properties.z=z;	properties.w=w;
	this->type=type;
	lineno=iLanguage->nextInstructionNumber();
}
void Symbol::Set (TYPE type,unsigned int index,TEXTARGET tt) {
	this->lineno=lineno;
	registerindex=index;
	properties.texturetarget=tt;
	this->type=type;
	lineno=iLanguage->nextInstructionNumber();
}
