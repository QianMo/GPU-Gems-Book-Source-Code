%{

//#define YYDEBUG 1
#ifdef WIN32
#pragma warning(disable:4786)
#pragma warning(disable:4065)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <sstream>
#include "ps2arb.h"
#include "ps2arb_intermediate.h"

using std::map;
using std::string;
using namespace std;//otherwise VC6 dies
extern int ps_lineno;

using namespace ps2arb;

BinOp * MXxYFactory (BinaryFactory * ThreeOrFour, 
                     const InstructionFlags & iflag,
                     const Register & dst, 
                     const Register & src0, 
                     const Register & src1,
                     int y);

static BinOp * M4x4Factory  (const InstructionFlags & iflag, const Register & dst, const Register & src0, const Register & src1) {
	return MXxYFactory(&Dp4Op::factory,iflag,dst,src0,src1,4);
}

static BinOp * M3x4Factory  (const InstructionFlags & iflag, const Register & dst, const Register & src0, const Register & src1) {
	return MXxYFactory(&Dp3Op::factory,iflag,dst,src0,src1,4);
}

static BinOp * M4x3Factory  (const InstructionFlags & iflag, const Register & dst, const Register & src0, const Register & src1) {
	return MXxYFactory(&Dp4Op::factory,iflag,dst,src0,src1,3);
}

static BinOp * M3x3Factory  (const InstructionFlags & iflag, const Register & dst, const Register & src0, const Register & src1) {
	return MXxYFactory(&Dp3Op::factory,iflag,dst,src0,src1,3);
}

static BinOp * M3x2Factory  (const InstructionFlags & iflag, const Register & dst, const Register & src0, const Register & src1) {
	return MXxYFactory(&Dp3Op::factory,iflag,dst,src0,src1,2);
}

static TriOp * ReverseCmpFactory  (const InstructionFlags & iflag, const Register & d, const Register & a, const Register & b, const Register & c) {
	return CmpOp::factory(iflag,d,a,c,b);
}


static map<string,VoidFactory*> createVoidFactory() {
	map<string,VoidFactory*> ret;
	ret.insert(map<string,VoidFactory*>::value_type("texkill",&KilOp::factory));
	return ret;
}
static map<string,UnaryFactory*> createUnaryFactory() {
	map<string,UnaryFactory*> ret;
	ret.insert(map<string,UnaryFactory*>::value_type("abs",&AbsOp::factory));
	ret.insert(map<string,UnaryFactory*>::value_type("exp",&ExpOp::factory));
	ret.insert(map<string,UnaryFactory*>::value_type("log",&LogOp::factory));
	ret.insert(map<string,UnaryFactory*>::value_type("frc",&FrcOp::factory));
	ret.insert(map<string,UnaryFactory*>::value_type("rcp",&RcpOp::factory));
	ret.insert(map<string,UnaryFactory*>::value_type("rsq",&RsqOp::factory));
	ret.insert(map<string,UnaryFactory*>::value_type("nrm",&NrmOp::factory));
	ret.insert(map<string,UnaryFactory*>::value_type("mov",&MovOp::factory));
	ret.insert(map<string,UnaryFactory*>::value_type("sincos",&ScsOp::factory));
	return ret;
}
static map<string,BinaryFactory*> createBinaryFactory() {
	map<string,BinaryFactory*> ret;
	ret.insert(map<string,BinaryFactory*>::value_type("add",&AddOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("sub",&SubOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("mul",&MulOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("crs",&XpdOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("dp3",&Dp3Op::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("dp4",&Dp4Op::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("pow",&PowOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("min",&MinOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("max",&MaxOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("m3x2",&M3x2Factory));
	ret.insert(map<string,BinaryFactory*>::value_type("m3x3",&M3x3Factory));
	ret.insert(map<string,BinaryFactory*>::value_type("m3x4",&M3x4Factory));
	ret.insert(map<string,BinaryFactory*>::value_type("m4x3",&M4x3Factory));
	ret.insert(map<string,BinaryFactory*>::value_type("m4x4",&M4x4Factory));
	ret.insert(map<string,BinaryFactory*>::value_type("texld",&TexldOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("texldp",&TexldpOp::factory));
	ret.insert(map<string,BinaryFactory*>::value_type("texldb",&TexldbOp::factory));
	return ret;
}
static map<string,TrinaryFactory*> createTrinaryFactory() {
	map<string,TrinaryFactory*> ret;
	ret.insert(map<string,TrinaryFactory*>::value_type("cmp",&ReverseCmpFactory));
	ret.insert(map<string,TrinaryFactory*>::value_type("lrp",&LrpOp::factory));
	ret.insert(map<string,TrinaryFactory*>::value_type("mad",&MadOp::factory));
	ret.insert(map<string,TrinaryFactory*>::value_type("dp2add",&Dp2addOp::factory));
	return ret;
}

static std::map<string,VoidFactory*> void_factory=createVoidFactory();
static std::map<string,UnaryFactory*> unary_factory=createUnaryFactory();
static std::map<string,BinaryFactory*> binary_factory=createBinaryFactory();
static std::map<string,TrinaryFactory*> trinary_factory=createTrinaryFactory();

#ifdef WIN32
#pragma warning( disable : 4102 ) 
#endif

extern int yylex(void);
static void yyerror (char *s) {
  fprintf (stderr, "Error Line %d: %s\n", ps_lineno,s);
}

%}

%union {
  char *s;
  float f;
  struct Register{
    char * swizzlemask;
    char * negate;
    char * reg;
  } reg;
}
%{

/* so about this function...it's nice cus it's centrally called
** --however there might be some argument that we're redoing the parsing
** phase of PS2 and that we should put this in the individual sections
** that know that ps2registers are being used...  static Register
*/
static Register
createPS2Register (const YYSTYPE::Register &r) {
	if (r.reg[0]!=0) {
		Symbol s;
		unsigned int index = atoi (r.reg+1);
		switch (r.reg[0]) {
		case 't':
			s.Set(Symbol::TEXCOORD,index);
			break;
		case 's':
			s.Set(Symbol::SAMPLE,index);
			break;
		case 'v':
			s.Set(Symbol::COLOR,index);
			break;
		case 'o':
			switch ( r.reg[1]) {
			case 'D':
			case 'd':
				s.Set (Symbol::OUTPUTDEPTH,0);
				break;
			case 'c':
			default:
				index = atoi(r.reg+2);
				s.Set(Symbol::OUTPUTCOLOR,index);
				break;
			}
			//output
			break;
		case 'c':
			s.Set(Symbol::CONST,index);
			break;
		case 'r':
		default:
			s.Set(Symbol::TEMP,index);
			break;			
		}
		iLanguage->SpecifySymbol (r.reg,s);
	}else{
		fprintf (stderr,"register %s not properly specified:",r.reg);
	}
	return Register(r.reg,r.swizzlemask,r.negate);
}

char * incRegister (string s) {
	if (s.length()) {
		char c=*s.begin();
		s = s.substr(1);
		int which=1+strtol(s.c_str(),NULL,10);
		char out [1024];
		sprintf(out,"%c%d",c,which);
		return strdup(out);
	}	
	return strdup("");
}

BinOp * MXxYFactory  (BinaryFactory * ThreeOrFour, const InstructionFlags & iflag, const Register & dst, const Register & src0, const Register & src1, int y) {
	YYSTYPE cinc;
	char * tmp=NULL;
	Register destination(dst);
	cinc.reg.swizzlemask="";
	cinc.reg.negate="";
	cinc.reg.reg = incRegister(src1);
	destination.swizzle="x";
	iLanguage->AddInst((*ThreeOrFour)(iflag,destination,src0,src1));
	destination.swizzle="y";
	BinOp * ret =(*ThreeOrFour)(iflag,destination,src0,createPS2Register(cinc.reg));
	if (y>2) {
		iLanguage->AddInst(ret);
		cinc.reg.reg = incRegister(tmp = cinc.reg.reg);
		free(tmp);	
		destination.swizzle="z";
		ret = (*ThreeOrFour)(iflag,destination,src0,createPS2Register(cinc.reg));
		if (y>3) {
			iLanguage->AddInst(ret);
			cinc.reg.reg = incRegister(tmp = cinc.reg.reg);
			free(tmp);
			destination.swizzle="w";
			ret= (*ThreeOrFour)(iflag,destination,src0,createPS2Register(cinc.reg));
		}
	}
	free(cinc.reg.reg);
	return ret;
}

static string strtoupper (string s) {
	for (string::iterator i =s.begin();i!=s.end();++i) {
		*i= toupper(*i);
	}
	return s;
}

InstructionFlags AdjustInstructionFlags (std::string flags) {
	flags = strtoupper(flags);
	bool pp = flags.find("_PP")!=string::npos;	
	bool sat = flags.find("_SAT")!=string::npos;
	return InstructionFlags(pp?InstructionFlags::PP:InstructionFlags::FULL,sat);
}

#define DEFINERETVAL YYSTYPE ret

%}
%token  <s>              PS_NOP PS_NEGATE PS_SWIZZLEMASK PS_COLORREG PS_TEMPREG PS_TEXCOORDREG PS_OUTPUTREG PS_SAMPLEREG PS_CONSTREG PS_TEXKILL PS_SINCOS PS_UNARY_OP PS_BINARY_OP PS_TRINARY_OP PS_OP_FLAGS PS_DCLTEX PS_DCL PS_DEF PS_COMMA PS_MOV PS_COMMENT PS_ENDLESS_COMMENT
%token <f>               PS_FLOAT PS_NEWLINE PS_PSHEADER
%type	<f>	program
%type <f>       newlines
%type <s>       colorortexreg
%type <reg> declreg;
%type <reg> srcreg;
%type <reg> dstreg;
%type <reg> movreg;
%type <s>        optionalwritemask;
%type <s>        readablereg;
%type <s>        optionalnegate;
%type <s>        optional_flags;
%type <f>	 optnewlines
%%
program: optnewlines PS_PSHEADER newlines declarations instructions optendlesscomment
;
declarations: declaration newlines declarations
                |
;
instructions: instruction newlines instructions
		| instruction
		|
;

declaration: PS_DCL declreg 
{	
	iLanguage->AddDecl(new DeclareRegisterUsage(createPS2Register($2)));
}
           | PS_DCLTEX PS_SAMPLEREG
{
	Symbol::TEXTARGET tt=Symbol::UNSPECIFIED;
	switch ($1[4]) {
           case '2':
              tt=Symbol::TEX2D;
	   break;
	   case '1':
              tt=Symbol::TEX1D;
	   break;
	   case 'c':
              tt=Symbol::CUBE;
	   break;
	   case '3':
	   case 'v':
	   default:
              tt=Symbol::TEX3D;	
	   break;
        }

	Symbol s;
	s.Set(Symbol::SAMPLE,atoi($2+1),tt);
	iLanguage->SpecifySymbol($2,s);
	YYSTYPE tmp;
        tmp.reg.reg=$2; 
        tmp.reg.swizzlemask="";
        tmp.reg.negate="";
	iLanguage->AddDecl(
            new DeclareSampleRegister(
               createPS2Register(tmp.reg)));
}
      | PS_DEF PS_CONSTREG PS_COMMA PS_FLOAT PS_COMMA PS_FLOAT PS_COMMA PS_FLOAT PS_COMMA PS_FLOAT
{       
        YYSTYPE tmp;
        tmp.reg.reg=$2; 
        tmp.reg.swizzlemask="";
        tmp.reg.negate="";
	iLanguage->AddDecl(
           new DefineConstantRegister(
             createPS2Register(tmp.reg),
		$4, $6, $8, $10));
};

instruction:  unary_op | binary_op | trinary_op | sincos | texkill | mov
{};

trinary_op: PS_TRINARY_OP optional_flags dstreg PS_COMMA srcreg PS_COMMA srcreg PS_COMMA srcreg
{
        iLanguage->AddInst((*trinary_factory[$1])(
	     AdjustInstructionFlags($2),
             createPS2Register($3),
             createPS2Register($5),
             createPS2Register($7),
             createPS2Register($9)));
};

binary_op: PS_BINARY_OP optional_flags dstreg PS_COMMA srcreg PS_COMMA srcreg
{
        iLanguage->AddInst((*binary_factory[$1])(
             AdjustInstructionFlags($2),
             createPS2Register($3),
             createPS2Register($5),
             createPS2Register($7)));
};

unary_op: PS_UNARY_OP optional_flags dstreg PS_COMMA srcreg
{
        iLanguage->AddInst((*unary_factory[$1])(
             AdjustInstructionFlags($2),
             createPS2Register($3),
             createPS2Register($5)));
};

optional_dualreg:  PS_COMMA srcreg PS_COMMA srcreg 
{}
                |  /*empty*/
{};

sincos: PS_SINCOS optional_flags dstreg PS_COMMA srcreg optional_dualreg
{
        iLanguage->AddInst((*unary_factory[$1])(
             AdjustInstructionFlags($2),
             createPS2Register($3),
             createPS2Register($5)));
};

texkill: PS_TEXKILL optional_flags srcreg
{
        iLanguage->AddInst((*void_factory[$1])(
             AdjustInstructionFlags($2),
             createPS2Register($3)));
};

mov: PS_MOV optional_flags movreg PS_COMMA  srcreg
{
        iLanguage->AddInst((*unary_factory[$1])(
             AdjustInstructionFlags($2),
             createPS2Register($3),
             createPS2Register($5)));
};

optional_flags: PS_OP_FLAGS
{
        $$=$1;
}
              | /*empty*/
{
        $$="";
};

optionalwritemask: PS_SWIZZLEMASK
{
        $$=$1;
}
                 | /*empty*/
{
        $$="";
};

colorortexreg: PS_COLORREG
             | PS_TEXCOORDREG
;

declreg: colorortexreg optionalwritemask
{
        DEFINERETVAL; 
        ret.reg.reg=$1;
        ret.reg.swizzlemask=$2;
        ret.reg.negate="";
        $$=ret.reg;
};

movreg: dstreg
{
        DEFINERETVAL;
        ret.reg = $1;
        $$ = ret.reg;
}
      | PS_OUTPUTREG
{
        DEFINERETVAL;
        ret.reg.reg = $1;
        ret.reg.swizzlemask = "";
        ret.reg.negate = "";
        $$ = ret.reg;
}; 

dstreg: PS_TEMPREG optionalwritemask
{
        DEFINERETVAL;
        ret.reg.reg=$1;
        ret.reg.swizzlemask = $2;
        ret.reg.negate="";
        $$=ret.reg;
};

optionalnegate: PS_NEGATE
{
        $$=$1;
}
              | /* empty */
{
        $$="";
};

srcreg: optionalnegate readablereg optionalwritemask
{
        DEFINERETVAL;
        ret.reg.negate = $1;
        ret.reg.reg = $2;
        ret.reg.swizzlemask = $3;
        $$=ret.reg
};

readablereg: PS_COLORREG | PS_TEXCOORDREG | PS_SAMPLEREG | PS_CONSTREG | PS_TEMPREG
;

newlines: PS_NEWLINE newlines 
{
        iLanguage->AddCommentOrNewline(
           new Newline((int)$1));
        $$ = $1 +$2;
}
        | PS_NEWLINE
{
        iLanguage->AddCommentOrNewline(
           new Newline((int)$1));
        $$ = $1;
}
        | PS_COMMENT newlines
{
        iLanguage->AddCommentOrNewline(
           new Comment($1));
        $$ = 1 + $2;
}
        | PS_COMMENT
{
        iLanguage->AddCommentOrNewline(
          new Comment($1));
        $$ = 1; 
};

optendlesscomment: PS_ENDLESS_COMMENT
{
        iLanguage->AddCommentOrNewline(new Comment($1));
}
                 | /* empty */
;

optnewlines: newlines 
{
        $$ = $1;
}
           | /*empty*/
{
        $$ = 0;
};

%%




