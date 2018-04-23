#ifndef _WIN32
#include <ctype.h>
#else
#pragma warning(disable:4786)
#endif
#include "ps2arb_intermediate.h"


using std::string;
using std::map;
using namespace std; // otherwise VC6 does not understand map::pair
using namespace ps2arb;

static string strtoupper (string input) {
	for (string::iterator i= input.begin();i!=input.end();++i) {
		*i=toupper(*i);
	}
	return input;
}
static string arbdelim=", ";
static string arbendl=";\n";
const float ARBMAXTEMPS=30;
extern string inttostr(int);
std::ostream &  InstructionFlags::print_arbfp (std::ostream & s)const {
	if (saturate)
	    s<<"_SAT";
	if (acc!=FULL) {
		//too bad!
	}
	return s;
}
std::ostream &  Newline::print_arbfp (std::ostream & s)const {
	if (0)
	for(unsigned int i=0;i<(unsigned int) numnewlines;++i) {
		s << "\n";
	}
	return s;
}
std::ostream &  Comment::print_arbfp (std::ostream & s)const {
	s<< "#"<<message;
	return s;
}
std::ostream &  OpCode::print_arbfp (std::ostream & s)const {
	s << strtoupper(opcode);
	flags.print_arbfp(s);
	s << " ";
	return s;
}
std::ostream &  Register::print_arbfp (std::ostream & s)const {
	s << negate << *this;
	if (swizzle.length())
		s << "." << swizzle;
	return s;
}

std::ostream &  DeclareRegisterUsage::print_arbfp (std::ostream & s)const {
	return s;
}

std::ostream &  DeclareSampleRegister::print_arbfp (std::ostream & s)const {
	return s;
}

std::ostream &  DefineConstantRegister::print_arbfp (std::ostream & s)const {
//	s << "PARAM "<<reg.print_arbfp(s)<<" = { " << x <<", "<<y << ", "<< z << ", "<< w <<" }"<< arbendl; 
	return s;
}

std::ostream &  VoidOp::print_arbfp (std::ostream & s)const {
	op.print_arbfp (s);
	src0.print_arbfp(s);
	s<<arbendl;
	return s;
}

std::ostream &  UnaOp::print_arbfp (std::ostream & s)const {
	op.print_arbfp(s);
	dst.print_arbfp(s);
	s<<arbdelim;
	src0.print_arbfp(s);
	s<<arbendl;
	return s;
}
std::ostream &NrmOp::print_arbfp(std::ostream &s)const {
  char inst[]="DP3";
  //if (src0.swizzle.length()==3)
  //inst[2]='3';
  OpCode OP(inst,InstructionFlags(InstructionFlags::FULL,false));  
  OP.print_arbfp(s);
  Register dstalias(dst);
  if (dstalias.swizzle.length()==0) {
    dstalias.swizzle="x";
  }else {
    dstalias.swizzle=dstalias.swizzle.substr(0,1);
  }
  dstalias.negate="";
  dstalias.print_arbfp(s);
  s<<arbdelim;
  src0.print_arbfp(s);
  s<<arbdelim;
  src0.print_arbfp(s);
  s<<arbendl;
  OP.opcode="RSQ";
  OP.print_arbfp(s);
  dstalias.print_arbfp(s);
  s<<arbdelim;
  dstalias.print_arbfp(s);
  s<<arbendl;
  OpCode FinalScale(op);
  FinalScale.opcode="MUL";//with saturation/precision issues
  FinalScale.print_arbfp(s);
  dst.print_arbfp(s);
  s<<arbdelim;
  //dstalias.swizzle+=dstalias.swizzle;
  //dstalias.swizzle+=dstalias.swizzle;//replicate 4x
  src0.print_arbfp(s);
  s<<arbdelim;
  dstalias.print_arbfp(s);
  s<<arbendl;
  return s;
}
std::ostream &  BinOp::print_arbfp (std::ostream & s)const {
	op.print_arbfp(s);
	dst.print_arbfp(s);
	s<<arbdelim;
	src0.print_arbfp(s);
	s<<arbdelim;
	src1.print_arbfp(s);
	s<<arbendl;
	return s;
}

std::ostream &  TriOp::print_arbfp (std::ostream & s)const {
	op.print_arbfp(s);
	dst.print_arbfp(s)<<arbdelim;
	src0.print_arbfp(s)<<arbdelim;
	src1.print_arbfp(s)<<arbdelim;
	src2.print_arbfp(s)<<arbendl;
	return s;
}

static std::ostream& printTexldX(std::ostream &s, 
                                 OpCode op,
                                 const Register & dst, 
                                 const Register & src0, 
                                 const Register & src1,
                                 const char *default_tex = "2D") {
	op.print_arbfp(s);
	dst.print_arbfp(s)<<arbdelim;
	src0.print_arbfp(s)<<arbdelim;
	Symbol sym(iLanguage->findSymbol(src1));
	s << "texture["<<sym.registerindex<<"]"<<arbdelim;
	switch (sym.properties.texturetarget) {
	case Symbol::TEX1D:
		s<< "1D";
		break;
	case Symbol::TEX3D:
		s<<"3D";
		break;
	case Symbol::CUBE:
		s<<"CUBE";
		break;
	case Symbol::RECT:
		s<<"RECT";
		break;
	case Symbol::TEX2D:
	default:
           s << "RECT";  //DANGER WILL ROBINSON! ALL 2D textures are now RECT
		break;
	}
	s<<arbendl;
	return s;
}

std::ostream & TexldOp::print_arbfp(std::ostream & s)const {
	OpCode texld (op);
	texld.opcode="TEX";
	return printTexldX(s,texld,dst,src0,src1);
}
std::ostream & TexldpOp::print_arbfp(std::ostream & s)const {
	OpCode texld (op);
	texld.opcode="TXP";
	return printTexldX(s,texld,dst,src0,src1);
}
std::ostream & TexldbOp::print_arbfp(std::ostream & s)const {
	OpCode texld (op);
	texld.opcode="TXB";
	return printTexldX(s,texld,dst,src0,src1);
}

std::ostream & PowOp::print_arbfp(std::ostream &s) const{
	OpCode exp (op);exp.opcode= ExpOp_str;
	OpCode mul (op);mul.opcode= MulOp_str;
	OpCode log (op);log.opcode=LogOp_str;
	log.print_arbfp(s);
	dst.print_arbfp(s)<<arbdelim;
	src0.print_arbfp(s)<<arbendl;
	mul.print_arbfp(s);
	dst.print_arbfp(s)<<arbdelim;
	src1.print_arbfp(s)<<arbdelim;
	dst.print_arbfp(s)<<arbendl;
	exp.print_arbfp(s);
	dst.print_arbfp(s)<<arbdelim;
	dst.print_arbfp(s)<<arbendl;
	return s;								 
}
string getArbTemp(std::ostream &s) {
	static int initialized=0;
	if (!initialized) {
		s<<"TEMP prm_tmp"<<arbendl;
		initialized=1;
	}
	return "prm_tmp";
}
std::ostream & Dp2addOp::print_arbfp(std::ostream &s) const{
	return print_alternative(s);
}

std::ostream & Dp2addOp::print_alternative(std::ostream &s) const{
	string adelim=arbdelim;
	string aendl=arbendl;
	OpCode mad =op;
	mad.opcode=MadOp_str;
	OpCode madnosat = mad;
	madnosat.flags.saturate=0;  //don't want to saturate for the first round of this thing
	Register src0x(src0);src0x.swizzle = "x";
	Register src1x(src1);src1x.swizzle = "x";
	//FIXME you may need to check to make sure that destination register is readable and writeable... in ps20 though it has to be...so perhaps this is a nonissue since this instr doesn't exist in arbfp
	Register dstZ(dst);
	dstZ.swizzle=(src2.swizzle=="w"||src2.swizzle=="a")?string("z"):string("w");//this way we don't overwrite values from x or y in src0 or src1...and we don't clobber src2.
	string temps;
	bool usetemporaries=false;
	if ((src0==dstZ&&src0.swizzle.find(dstZ.swizzle)!=string::npos) ||
		(src1==dstZ&&src1.swizzle.find(dstZ.swizzle)!=string::npos)) {
		usetemporaries=true;
                temps=getArbTemp(s);
	}
        madnosat.print_arbfp(s);
	
	if (usetemporaries) {
		s<<temps;
	}else {           
           dstZ.print_arbfp(s);
	}
	s<<adelim;
        src0x.print_arbfp(s);
	s<<adelim;
        src1x.print_arbfp(s);
	s<<adelim;
        src2.print_arbfp(s);
	s<<aendl;
	Register src0y(src0);src0y.swizzle = "y";	
	Register src1y(src1);src1y.swizzle = "y";
        mad.print_arbfp(s);
        dst.print_arbfp(s);
	s<< adelim;
        src0y.print_arbfp(s);
	s<<adelim;
        src1y.print_arbfp(s);	
	s<<adelim;
	if (usetemporaries) {
		s<<temps;
	}else {
           dstZ.print_arbfp(s);
	}
	s<<aendl;
	return s;
}

std::ostream & IntermediateLanguage::print_arbfp (std::ostream & s) {
   for (unsigned int i=0;i<comments.size();++i) {
      //hack for now to get some important comments before the !!
      comments[i]->print_arbfp(s);
   }
   s<<"!!ARBfp1.0\n";
   if (1){
      bool multiple_output=false;
      if (1) {			   
         for (map<string,Symbol>::iterator i = symbolTable.begin();i!=symbolTable.end();++i) {
            Symbol * sym= &(*i).second;
            if (sym->type==Symbol::OUTPUTCOLOR) {
               if (sym->registerindex>0) {
                  multiple_output=true;
                  s<<"OPTION ATI_draw_buffers"<<arbendl;
                  break;
               }
            }
         }
      }
      for (map<string,Symbol>::iterator i = symbolTable.begin();i!=symbolTable.end();++i) {
         string nam = (*i).first;
         Symbol * sym = &(*i).second;
         switch (sym->type) {
         case Symbol::OUTPUTCOLOR:
            s << "OUTPUT "<<nam<<" = result.color";
            if (multiple_output)
               s << "["<<sym->registerindex<<"]";
            s<<arbendl;
            break;
         case Symbol::OUTPUTDEPTH:
            s << "OUTPUT "<<nam<<" = result.depth"<<arbendl;					
            break;
         case Symbol::TEMP:
            s << "TEMP " << nam << arbendl;										
            break;
         case Symbol::CONST:
            if (sym->properties.validfloats) {
               s << "PARAM " << nam << " = {"<<sym->properties.x <<", "<<sym->properties.y <<", "<<sym->properties.z <<", "<<sym->properties.w <<"}"<<arbendl;
            }else {
               s << "PARAM " << nam << " = program.local[" <<sym->registerindex<<"]"<<arbendl;
            }
            break;
         case Symbol::SAMPLE:
            //we need this?
            break;
         case Symbol::TEXCOORD:
            s << "ATTRIB "<<nam<<" = fragment.texcoord["<<sym->registerindex<<"]"<<arbendl;
            break;
         case Symbol::COLOR:
            s << "ATTRIB "<<nam<<" = fragment.color["<<sym->registerindex<<"]"<<arbendl;					
            break;
         case Symbol::FOGCOORD:
            s << "ATTRIB "<<nam<<" = fragment.fogcoord"<<arbendl;					
            break;
         case Symbol::POSITION:
            s << "ATTRIB "<<nam<<" = fragment.position"<<arbendl;  
            break;
         default:
            s << "UNKNOWN" <<arbendl;
            break;
         }
      }
   }
   {for (unsigned int i=0;i<stmt.size();++i) {
      stmt[i]->print_arbfp(s);
   }}
   s << "END\n";
   return s;
}
