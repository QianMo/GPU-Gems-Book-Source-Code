#include <string>
#include <vector>
#include <fstream>

#ifdef _WIN32
#include <ios>
#pragma warning(disable:4786)
//that above pragma disables the warnings about #include <map>
#endif

#include <map>
#include "autocloner.h"

#define CLONE_F(T,S) virtual S * clone (S *)const {return new T (*this);}

namespace ps2arb {

//FIXME has to be PROTOTYPES not functions
#define PRINT_PROTOTYPES \
   virtual std::ostream & print_ps2  (std::ostream & s) const {return s;} \
   virtual std::ostream & print_arbfp(std::ostream & s) const;            \

#define PRINT_FUNCTIONS \
   virtual std::ostream & print_ps2  (std::ostream & s) const {return s;} \
   virtual std::ostream & print_arbfp(std::ostream & s) const {return s;}

using std::string;

class Statement {
public:
	CLONE_F(Statement,Statement);
	virtual ~Statement(){}
	PRINT_FUNCTIONS;
};

class Newline: public Statement {
	int numnewlines;
public:
	Newline (int num):numnewlines(num){}
	CLONE_F(Newline, Statement);
	PRINT_PROTOTYPES;
};
class Comment: public Newline {
	std::string message;
public:
	Comment (std::string s):Newline(1),message(s+string("\n")){}
	CLONE_F(Comment,Statement);
	PRINT_PROTOTYPES;
};

class Register : public string{
public:
	string swizzle;//or mask;
	string negate;
	Register (string reg, string mask, string negate);
	PRINT_PROTOTYPES;
	virtual ~Register () {}
};

class Declaration: public Statement {
public:
	Register reg;
	CLONE_F(Declaration, Statement);
	CLONE_F(Declaration, Declaration);
	PRINT_FUNCTIONS;
	Declaration ( const Register & r):reg (r){}
};

class DeclareRegisterUsage: public Declaration {
public:
	CLONE_F (DeclareRegisterUsage,Statement);
	CLONE_F (DeclareRegisterUsage,Declaration);
	PRINT_PROTOTYPES;
	DeclareRegisterUsage (const Register & reg):Declaration(reg){}
};
class DeclareSampleRegister : public Declaration {
public:
	CLONE_F (DeclareSampleRegister,Statement);
	CLONE_F (DeclareSampleRegister,Declaration);
	PRINT_PROTOTYPES;
	DeclareSampleRegister (const Register & reg):Declaration(reg){}
};

class DefineConstantRegister : public Declaration {
public:
	float x,y,z,w;
	CLONE_F (DefineConstantRegister,Statement);
	CLONE_F (DefineConstantRegister,Declaration);
	PRINT_PROTOTYPES;
	DefineConstantRegister (const Register & reg,float a,float b, float c, float d);
};
class InstructionFlags {
public:
	enum ACCURACY {FULL, H, X,PP} acc;
	unsigned saturate:1;
	InstructionFlags (ACCURACY a, bool sat) {
		saturate=  sat?1:0;
		acc=a;
	}
	PRINT_PROTOTYPES;
};
class OpCode{
public:
  InstructionFlags flags;
  string opcode;
  OpCode (string opcode, const InstructionFlags &flags):
    flags(flags), opcode(opcode) {}
  virtual ~OpCode () {}
  //OpCode (string opcode):flags(InstructionFlags::FULL,0),opcode(opcode){}
	PRINT_PROTOTYPES;
};

class Instruction:public Statement {
public:
  CLONE_F(Instruction,Statement);
  CLONE_F(Instruction,Instruction);
  Instruction (const OpCode &oc):op(oc){}
  void SetLowPrecision(InstructionFlags::ACCURACY a) {op.flags.acc=a;}
  void SetSaturated(bool sat) {op.flags.saturate=sat?1:0;}
  PRINT_FUNCTIONS;
  void AdjustOpcodeFlags(const InstructionFlags & f) {
    op.flags = f;
  }
protected:
  OpCode op;
};

class VoidOp:public Instruction {
public:
	CLONE_F(VoidOp,Statement);
	CLONE_F(VoidOp,Instruction);
	VoidOp (const OpCode & instruction, const Register &src0):Instruction(instruction),src0(src0){}
	PRINT_PROTOTYPES;
protected:
	Register src0;
};
class UnaOp:public Instruction {
public:
	CLONE_F (UnaOp,Statement);
	CLONE_F (UnaOp,Instruction);
	UnaOp (const OpCode & instruction, const Register& dst,const Register & src0):Instruction(instruction),dst(dst),src0(src0){}
	PRINT_PROTOTYPES;
protected:	
	Register dst;
	Register src0;
};
class BinOp:public Instruction {
public:
	CLONE_F (BinOp,Statement);
	CLONE_F (BinOp,Instruction);
	BinOp (const OpCode & instruction, const Register& dst,const Register & src0,const Register & src1):Instruction(instruction),dst(dst),src0(src0),src1(src1){}
	PRINT_PROTOTYPES;
protected:
	Register dst;
	Register src0,src1;
};
class TriOp:public Instruction{
public:
	CLONE_F (TriOp,Statement);
	CLONE_F (TriOp,Instruction);
	TriOp (const OpCode & instruction, const Register& dst,const Register & src0,const Register & src1, const Register & src2):Instruction(instruction),dst(dst),src0(src0),src1(src1),src2(src2){}
	PRINT_PROTOTYPES;
protected:	
	Register dst;
	Register src0,src1,src2;
};
typedef UnaOp * (UnaryFactory) (const InstructionFlags & iflag, const Register &dst,const Register &src0);
typedef BinOp * (BinaryFactory) (const InstructionFlags & iflag,const Register &dst,const Register &src0,const Register &src1);
typedef TriOp * (TrinaryFactory) (const InstructionFlags & iflag,const Register &dst,const Register &src0,const Register &src1,const Register &src2);
typedef VoidOp * (VoidFactory) (const InstructionFlags & iflag,const Register &src);
#define VOIDOPDEF(Name) CLONE_F(Name,Statement);CLONE_F(Name,Instruction); static VoidOp *factory(const InstructionFlags & iflag,const Register & src0){return new Name(iflag,src0);}	Name(const InstructionFlags & iflag,const Register & src0):VoidOp(OpCode(Name##_str,iflag),src0){}
#define UNAOPDEF(Name) CLONE_F(Name,Statement);CLONE_F(Name,Instruction); static UnaOp *factory(const InstructionFlags & iflag,const Register & dst, const Register & src0){return new Name(iflag,dst,src0);}  Name(const InstructionFlags & iflag,const Register & dst, const Register & src0):UnaOp(OpCode(Name##_str,iflag),dst,src0){}
#define BINOPDEF(Name) CLONE_F(Name,Statement);CLONE_F(Name,Instruction); static BinOp *factory(const InstructionFlags & iflag,const Register & dst, const Register & src0, const Register & src1){return new Name(iflag,dst,src0,src1);}	Name(const InstructionFlags & iflag,const Register & dst, const Register & src0, const Register & src1):BinOp(OpCode(Name##_str,iflag),dst,src0,src1){}
#define TRIOPDEF(Name) CLONE_F(Name,Statement);CLONE_F(Name,Instruction); static TriOp *factory(const InstructionFlags & iflag,const Register & dst, const Register & src0, const Register & src1,const Register& src2){return new Name(iflag,dst,src0,src1,src2);}	Name(const InstructionFlags & iflag,const Register & dst, const Register & src0, const Register & src1, const Register & src2):TriOp(OpCode(Name##_str,iflag),dst,src0,src1,src2){}

static const char * const KilOp_str="kil";
class KilOp:public VoidOp {
public:
	VOIDOPDEF(KilOp);
};
static const char *const  LitOp_str="lit";	
class LitOp:public UnaOp {
public:
	UNAOPDEF(LitOp);
};
static const char *const AbsOp_str="abs";
class AbsOp:public UnaOp {
public:
	UNAOPDEF(AbsOp);
};
static const char *const ExpOp_str="ex2";
class ExpOp:public UnaOp{
public:
	UNAOPDEF(ExpOp);
};
static const char *const ScsOp_str="scs";
class ScsOp:public UnaOp{
public:
	UNAOPDEF(ScsOp);
};
static const char * const DphOp_str="dph";
class DphOp:public BinOp {
	BINOPDEF(DphOp);
};
static const char *const LogOp_str="lg2";
class LogOp:public UnaOp{
public:
	UNAOPDEF(LogOp);
};
static const char *const FrcOp_str="frc";
class FrcOp:public UnaOp{
public:
	UNAOPDEF(FrcOp);
};
static const char *const RcpOp_str="rcp";
class RcpOp:public UnaOp{
public:
	UNAOPDEF(RcpOp);
};
static const char *const RsqOp_str="rsq";
class RsqOp:public UnaOp{
public:
	UNAOPDEF(RsqOp);
};
static const char *const NrmOp_str="nrm";
class NrmOp:public UnaOp{
public:
	UNAOPDEF(NrmOp);
	virtual std::ostream& print_arbfp(std::ostream &s)const;
};
static const char *const MovOp_str="mov";
class MovOp:public UnaOp{
public:
	UNAOPDEF(MovOp);
};
static const char *const  AddOp_str="add";	
class AddOp:public BinOp {
public:
	BINOPDEF(AddOp);
};
static const char *const  SubOp_str="sub";	
class SubOp:public BinOp {
public:
	BINOPDEF(SubOp);
};
static const char *const  MulOp_str="mul";	
class MulOp:public BinOp {
public:
	BINOPDEF(MulOp);
};
static const char *const  XpdOp_str="xpd";	
class XpdOp:public BinOp {
public:
	BINOPDEF(XpdOp);
};
static const char *const  Dp3Op_str="dp3";	
class Dp3Op:public BinOp {
public:
	BINOPDEF(Dp3Op);
};
static const char *const  Dp4Op_str="dp4";	
class Dp4Op:public BinOp {
public:
	BINOPDEF(Dp4Op);
};
static const char *const  MinOp_str="min";	
class MinOp:public BinOp {
public:
	BINOPDEF(MinOp);
};
static const char *const  MaxOp_str="max";	
class MaxOp:public BinOp {
public:
	BINOPDEF(MaxOp);
};

#define XXX(a, b)                                         \
static const char *const  a##Op_str=#b;                   \
class a##Op:public BinOp {                                \
public:                                                   \
	BINOPDEF(a##Op);                                  \
	std::ostream &print_arbfp(std::ostream &s)const;  \
};

   XXX(Pow, pow);
   XXX(Texld, tex);
   XXX(Texldp, texldp);
   XXX(Texldb, texldb);

#undef XXX

static const char *const  CmpOp_str="cmp";	
class CmpOp:public TriOp {
public:
	TRIOPDEF(CmpOp);
};
static const char *const  Dp2addOp_str="dp2add";	
class Dp2addOp:public TriOp {
public:
	virtual std::ostream& print_alternative(std::ostream &s)const;
	virtual std::ostream& print_arbfp(std::ostream &s)const;
	TRIOPDEF(Dp2addOp);
};
static const char *const  LrpOp_str="lrp";	
class LrpOp:public TriOp {
public:
	TRIOPDEF(LrpOp);
};
static const char *const  MadOp_str="mad";	
class MadOp:public TriOp {
public:
	TRIOPDEF(MadOp);
};


class Symbol {
public:
	enum TYPE {UNKNOWN,
			   OUTPUTCOLOR,
			   OUTPUTDEPTH,
			   TEMP,/*r0,r1...*/
			   CONST,/*c0,c1,...GL state*/
			   SAMPLE,/*s0,s1,...*/
			   //below we have items in the per fragment.* section
			   TEXCOORD,/*t0,t1,...*/
			   COLOR,/*v0,v1 colors*/
			   FOGCOORD,
			   POSITION
			   }type;
	int registerindex;//especially important for texcoord,color, sample, outputcolor registers
	bool Valid () const{return type!=UNKNOWN;}
	bool inValid () const {return type==UNKNOWN;}
	enum TEXTARGET {UNSPECIFIED,TEX1D,TEX2D,TEX3D,CUBE,RECT};	
	struct Properties {
		TEXTARGET texturetarget;
		bool validfloats;
		float x,y,z,w;
		
	} properties;
	int lineno;
	bool operator ==(const Symbol & a)const {
		return type==a.type&&registerindex==a.registerindex&&properties.texturetarget==a.properties.texturetarget;
	}
	Symbol ();
	void Set (TYPE type, unsigned int index);
	void Set (TYPE type,unsigned int index,TEXTARGET tt);
	void Set (TYPE type,unsigned int index,float x,float y, float z, float w);
	void Specify (const Symbol &s);
};
						
class IntermediateLanguage {
	std::vector<AutoCloner<Statement> > comments;	
	std::vector<AutoCloner<Declaration> > decl;
	std::vector<AutoCloner<Statement> > stmt;
	std::map<string,Symbol> symbolTable;		
public:
	bool makeTemporaries(std::vector <Symbol> &ret, int num, int maxTemporaries);
	void SpecifySymbol(std::string name, const Symbol &sym);
	Symbol findSymbol(string s);
	unsigned int nextInstructionNumber() {
		return stmt.size();
	}
	void AddCommentOrNewline (Statement * s) {
		AutoCloner<Statement> cmt(s);
		if (stmt.empty()) {
			comments.push_back(cmt);
		}else {
			stmt.push_back (cmt);
		}
	}
	void AddInst (Statement * s) {
		stmt.push_back (AutoCloner<Statement> (s));
	}
	void AddDecl (Declaration * s) {
		decl.push_back(AutoCloner<Declaration> (s->clone(s)));
		stmt.push_back(AutoCloner<Statement> (s));
	}
	std::ostream & print_arbfp (std::ostream & s);
};

extern IntermediateLanguage *iLanguage;

}
