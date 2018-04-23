/* A Bison parser, made by GNU Bison 1.875b.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Written by Richard Stallman by simplifying the original so called
   ``semantic'' parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0

/* If NAME_PREFIX is specified substitute the variables and functions
   names.  */
#define yyparse ps2arb_parse
#define yylex   ps2arb_lex
#define yyerror ps2arb_error
#define yylval  ps2arb_lval
#define yychar  ps2arb_char
#define yydebug ps2arb_debug
#define yynerrs ps2arb_nerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     PS_NOP = 258,
     PS_NEGATE = 259,
     PS_SWIZZLEMASK = 260,
     PS_COLORREG = 261,
     PS_TEMPREG = 262,
     PS_TEXCOORDREG = 263,
     PS_OUTPUTREG = 264,
     PS_SAMPLEREG = 265,
     PS_CONSTREG = 266,
     PS_TEXKILL = 267,
     PS_SINCOS = 268,
     PS_UNARY_OP = 269,
     PS_BINARY_OP = 270,
     PS_TRINARY_OP = 271,
     PS_OP_FLAGS = 272,
     PS_DCLTEX = 273,
     PS_DCL = 274,
     PS_DEF = 275,
     PS_COMMA = 276,
     PS_MOV = 277,
     PS_COMMENT = 278,
     PS_ENDLESS_COMMENT = 279,
     PS_FLOAT = 280,
     PS_NEWLINE = 281,
     PS_PSHEADER = 282
   };
#endif
#define PS_NOP 258
#define PS_NEGATE 259
#define PS_SWIZZLEMASK 260
#define PS_COLORREG 261
#define PS_TEMPREG 262
#define PS_TEXCOORDREG 263
#define PS_OUTPUTREG 264
#define PS_SAMPLEREG 265
#define PS_CONSTREG 266
#define PS_TEXKILL 267
#define PS_SINCOS 268
#define PS_UNARY_OP 269
#define PS_BINARY_OP 270
#define PS_TRINARY_OP 271
#define PS_OP_FLAGS 272
#define PS_DCLTEX 273
#define PS_DCL 274
#define PS_DEF 275
#define PS_COMMA 276
#define PS_MOV 277
#define PS_COMMENT 278
#define PS_ENDLESS_COMMENT 279
#define PS_FLOAT 280
#define PS_NEWLINE 281
#define PS_PSHEADER 282




/* Copy the first part of user declarations.  */
#line 1 "ps2arb_gram.y"


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



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 120 "ps2arb_gram.y"
typedef union YYSTYPE {
  char *s;
  float f;
  struct Register{
    char * swizzlemask;
    char * negate;
    char * reg;
  } reg;
} YYSTYPE;
/* Line 191 of yacc.c.  */
#line 268 "ps2arb_gram.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */
#line 129 "ps2arb_gram.y"


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



/* Line 214 of yacc.c.  */
#line 389 "ps2arb_gram.tab.c"

#if ! defined (yyoverflow) || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# if YYSTACK_USE_ALLOCA
#  define YYSTACK_ALLOC alloca
# else
#  ifndef YYSTACK_USE_ALLOCA
#   if defined (alloca) || defined (_ALLOCA_H)
#    define YYSTACK_ALLOC alloca
#   else
#    ifdef __GNUC__
#     define YYSTACK_ALLOC __builtin_alloca
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
# else
#  if defined (__STDC__) || defined (__cplusplus)
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   define YYSIZE_T size_t
#  endif
#  define YYSTACK_ALLOC malloc
#  define YYSTACK_FREE free
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  short yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short) + sizeof (YYSTYPE))				\
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  register YYSIZE_T yyi;		\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (0)
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (0)

#endif

#if defined (__STDC__) || defined (__cplusplus)
   typedef signed char yysigned_char;
#else
   typedef short yysigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  8
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   80

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  28
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  25
/* YYNRULES -- Number of rules. */
#define YYNRULES  50
/* YYNRULES -- Number of states. */
#define YYNSTATES  99

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   282

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned char yyprhs[] =
{
       0,     0,     3,    10,    14,    15,    19,    21,    22,    25,
      28,    39,    41,    43,    45,    47,    49,    51,    61,    69,
      75,    80,    81,    88,    92,    98,   100,   101,   103,   104,
     106,   108,   111,   113,   115,   118,   120,   121,   125,   127,
     129,   131,   133,   135,   138,   140,   143,   145,   147,   148,
     150
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const yysigned_char yyrhs[] =
{
      29,     0,    -1,    52,    27,    50,    30,    31,    51,    -1,
      32,    50,    30,    -1,    -1,    33,    50,    31,    -1,    33,
      -1,    -1,    19,    44,    -1,    18,    10,    -1,    20,    11,
      21,    25,    21,    25,    21,    25,    21,    25,    -1,    36,
      -1,    35,    -1,    34,    -1,    38,    -1,    39,    -1,    40,
      -1,    16,    41,    46,    21,    48,    21,    48,    21,    48,
      -1,    15,    41,    46,    21,    48,    21,    48,    -1,    14,
      41,    46,    21,    48,    -1,    21,    48,    21,    48,    -1,
      -1,    13,    41,    46,    21,    48,    37,    -1,    12,    41,
      48,    -1,    22,    41,    45,    21,    48,    -1,    17,    -1,
      -1,     5,    -1,    -1,     6,    -1,     8,    -1,    43,    42,
      -1,    46,    -1,     9,    -1,     7,    42,    -1,     4,    -1,
      -1,    47,    49,    42,    -1,     6,    -1,     8,    -1,    10,
      -1,    11,    -1,     7,    -1,    26,    50,    -1,    26,    -1,
      23,    50,    -1,    23,    -1,    24,    -1,    -1,    50,    -1,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   253,   253,   255,   256,   258,   259,   260,   263,   267,
     298,   310,   310,   310,   310,   310,   310,   313,   323,   332,
     340,   343,   345,   353,   360,   368,   373,   377,   382,   386,
     387,   390,   399,   405,   414,   423,   428,   432,   441,   441,
     441,   441,   441,   444,   450,   456,   462,   469,   473,   476,
     481
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "PS_NOP", "PS_NEGATE", "PS_SWIZZLEMASK", 
  "PS_COLORREG", "PS_TEMPREG", "PS_TEXCOORDREG", "PS_OUTPUTREG", 
  "PS_SAMPLEREG", "PS_CONSTREG", "PS_TEXKILL", "PS_SINCOS", "PS_UNARY_OP", 
  "PS_BINARY_OP", "PS_TRINARY_OP", "PS_OP_FLAGS", "PS_DCLTEX", "PS_DCL", 
  "PS_DEF", "PS_COMMA", "PS_MOV", "PS_COMMENT", "PS_ENDLESS_COMMENT", 
  "PS_FLOAT", "PS_NEWLINE", "PS_PSHEADER", "$accept", "program", 
  "declarations", "instructions", "declaration", "instruction", 
  "trinary_op", "binary_op", "unary_op", "optional_dualreg", "sincos", 
  "texkill", "mov", "optional_flags", "optionalwritemask", 
  "colorortexreg", "declreg", "movreg", "dstreg", "optionalnegate", 
  "srcreg", "readablereg", "newlines", "optendlesscomment", "optnewlines", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    28,    29,    30,    30,    31,    31,    31,    32,    32,
      32,    33,    33,    33,    33,    33,    33,    34,    35,    36,
      37,    37,    38,    39,    40,    41,    41,    42,    42,    43,
      43,    44,    45,    45,    46,    47,    47,    48,    49,    49,
      49,    49,    49,    50,    50,    50,    50,    51,    51,    52,
      52
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     6,     3,     0,     3,     1,     0,     2,     2,
      10,     1,     1,     1,     1,     1,     1,     9,     7,     5,
       4,     0,     6,     3,     5,     1,     0,     1,     0,     1,
       1,     2,     1,     1,     2,     1,     0,     3,     1,     1,
       1,     1,     1,     2,     1,     2,     1,     1,     0,     1,
       0
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
      50,    46,    44,     0,    49,     0,    45,    43,     1,     0,
       4,     0,     0,     0,     7,     0,     9,    29,    30,    28,
       8,     0,    26,    26,    26,    26,    26,    26,    48,     6,
      13,    12,    11,    14,    15,    16,     4,    27,    31,     0,
      25,    36,     0,     0,     0,     0,     0,    47,     2,     7,
       3,     0,    35,     0,    23,    28,     0,     0,     0,     0,
      33,     0,    32,     5,     0,    38,    42,    39,    40,    41,
      28,    34,    36,    36,    36,    36,    36,     0,    37,    21,
      19,     0,     0,    24,     0,    36,    22,    36,    36,     0,
       0,    18,     0,     0,    36,    36,    10,    20,    17
};

/* YYDEFGOTO[NTERM-NUM]. */
static const yysigned_char yydefgoto[] =
{
      -1,     3,    14,    28,    15,    29,    30,    31,    32,    86,
      33,    34,    35,    41,    38,    19,    20,    61,    56,    53,
      54,    70,     4,    48,     5
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -71
static const yysigned_char yypact[] =
{
       0,     0,     0,    16,   -71,    -7,   -71,   -71,   -71,     0,
      27,    11,    42,    21,    -3,     0,   -71,   -71,   -71,    22,
     -71,    19,    32,    32,    32,    32,    32,    32,    28,     0,
     -71,   -71,   -71,   -71,   -71,   -71,    27,   -71,   -71,    29,
     -71,    51,    49,    49,    49,    49,    44,   -71,   -71,    -3,
     -71,    36,   -71,    23,   -71,    22,    37,    38,    39,    40,
     -71,    41,   -71,   -71,    43,   -71,   -71,   -71,   -71,   -71,
      22,   -71,    51,    51,    51,    51,    51,    45,   -71,    46,
     -71,    48,    50,   -71,    47,    51,   -71,    51,    51,    52,
      53,   -71,    54,    55,    51,    51,   -71,   -71,   -71
};

/* YYPGOTO[NTERM-NUM].  */
static const yysigned_char yypgoto[] =
{
     -71,   -71,    34,    14,   -71,   -71,   -71,   -71,   -71,   -71,
     -71,   -71,   -71,    12,   -48,   -71,   -71,   -71,    -2,   -71,
     -70,   -71,    -1,   -71,   -71
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const unsigned char yytable[] =
{
       6,     7,    79,    80,    81,    82,    83,    71,    10,    22,
      23,    24,    25,    26,    36,    90,     8,    91,    92,    27,
       9,    16,    78,     1,    97,    98,     2,    37,    49,    65,
      66,    67,    21,    68,    69,    42,    43,    44,    45,    46,
      39,    57,    58,    59,    62,    11,    12,    13,    17,    40,
      18,    55,    47,    60,    51,    52,    55,    64,    72,    73,
      74,    75,    76,    63,     0,     0,    84,    85,    77,    87,
      50,    88,    89,    93,    94,    95,     0,     0,     0,     0,
      96
};

static const yysigned_char yycheck[] =
{
       1,     2,    72,    73,    74,    75,    76,    55,     9,    12,
      13,    14,    15,    16,    15,    85,     0,    87,    88,    22,
      27,    10,    70,    23,    94,    95,    26,     5,    29,     6,
       7,     8,    11,    10,    11,    23,    24,    25,    26,    27,
      21,    43,    44,    45,    46,    18,    19,    20,     6,    17,
       8,     7,    24,     9,    25,     4,     7,    21,    21,    21,
      21,    21,    21,    49,    -1,    -1,    21,    21,    25,    21,
      36,    21,    25,    21,    21,    21,    -1,    -1,    -1,    -1,
      25
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,    23,    26,    29,    50,    52,    50,    50,     0,    27,
      50,    18,    19,    20,    30,    32,    10,     6,     8,    43,
      44,    11,    12,    13,    14,    15,    16,    22,    31,    33,
      34,    35,    36,    38,    39,    40,    50,     5,    42,    21,
      17,    41,    41,    41,    41,    41,    41,    24,    51,    50,
      30,    25,     4,    47,    48,     7,    46,    46,    46,    46,
       9,    45,    46,    31,    21,     6,     7,     8,    10,    11,
      49,    42,    21,    21,    21,    21,    21,    25,    42,    48,
      48,    48,    48,    48,    21,    21,    37,    21,    21,    25,
      48,    48,    48,    21,    21,    21,    25,    48,    48
};

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# if defined (__STDC__) || defined (__cplusplus)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# endif
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrlab1


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    { 								\
      yyerror ("syntax error: cannot back up");\
      YYERROR;							\
    }								\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

/* YYLLOC_DEFAULT -- Compute the default location (before the actions
   are run).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)         \
  Current.first_line   = Rhs[1].first_line;      \
  Current.first_column = Rhs[1].first_column;    \
  Current.last_line    = Rhs[N].last_line;       \
  Current.last_column  = Rhs[N].last_column;
#endif

/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (0)

# define YYDSYMPRINT(Args)			\
do {						\
  if (yydebug)					\
    yysymprint Args;				\
} while (0)

# define YYDSYMPRINTF(Title, Token, Value, Location)		\
do {								\
  if (yydebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      yysymprint (stderr, 					\
                  Token, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (cinluded).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_stack_print (short *bottom, short *top)
#else
static void
yy_stack_print (bottom, top)
    short *bottom;
    short *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (/* Nothing. */; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_reduce_print (int yyrule)
#else
static void
yy_reduce_print (yyrule)
    int yyrule;
#endif
{
  int yyi;
  unsigned int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %u), ",
             yyrule - 1, yylno);
  /* Print the symbols being reduced, and their result.  */
  for (yyi = yyprhs[yyrule]; 0 <= yyrhs[yyi]; yyi++)
    YYFPRINTF (stderr, "%s ", yytname [yyrhs[yyi]]);
  YYFPRINTF (stderr, "-> %s\n", yytname [yyr1[yyrule]]);
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (Rule);		\
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YYDSYMPRINT(Args)
# define YYDSYMPRINTF(Title, Token, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   SIZE_MAX < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#if YYMAXDEPTH == 0
# undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined (__GLIBC__) && defined (_STRING_H)
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
#   if defined (__STDC__) || defined (__cplusplus)
yystrlen (const char *yystr)
#   else
yystrlen (yystr)
     const char *yystr;
#   endif
{
  register const char *yys = yystr;

  while (*yys++ != '\0')
    continue;

  return yys - yystr - 1;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined (__GLIBC__) && defined (_STRING_H) && defined (_GNU_SOURCE)
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
#   if defined (__STDC__) || defined (__cplusplus)
yystpcpy (char *yydest, const char *yysrc)
#   else
yystpcpy (yydest, yysrc)
     char *yydest;
     const char *yysrc;
#   endif
{
  register char *yyd = yydest;
  register const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

#endif /* !YYERROR_VERBOSE */



#if YYDEBUG
/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yysymprint (FILE *yyoutput, int yytype, YYSTYPE *yyvaluep)
#else
static void
yysymprint (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  if (yytype < YYNTOKENS)
    {
      YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
# ifdef YYPRINT
      YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
    }
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  switch (yytype)
    {
      default:
        break;
    }
  YYFPRINTF (yyoutput, ")");
}

#endif /* ! YYDEBUG */
/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yydestruct (int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yytype, yyvaluep)
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  switch (yytype)
    {

      default:
        break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM);
# else
int yyparse ();
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM)
# else
int yyparse (YYPARSE_PARAM)
  void *YYPARSE_PARAM;
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  register int yystate;
  register int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short	yyssa[YYINITDEPTH];
  short *yyss = yyssa;
  register short *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  register YYSTYPE *yyvsp;



#define YYPOPSTACK   (yyvsp--, yyssp--)

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* When reducing, the number of symbols on the RHS of the reduced
     rule.  */
  int yylen;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed. so pushing a state here evens the stacks.
     */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack. Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	short *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow ("parser stack overflow",
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyoverflowlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyoverflowlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	short *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyoverflowlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YYDSYMPRINTF ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */
  YYDPRINTF ((stderr, "Shifting token %s, ", yytname[yytoken]));

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;


  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  yystate = yyn;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 8:
#line 264 "ps2arb_gram.y"
    {	
	iLanguage->AddDecl(new DeclareRegisterUsage(createPS2Register(yyvsp[0].reg)));
;}
    break;

  case 9:
#line 268 "ps2arb_gram.y"
    {
	Symbol::TEXTARGET tt=Symbol::UNSPECIFIED;
	switch (yyvsp[-1].s[4]) {
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
	s.Set(Symbol::SAMPLE,atoi(yyvsp[0].s+1),tt);
	iLanguage->SpecifySymbol(yyvsp[0].s,s);
	YYSTYPE tmp;
        tmp.reg.reg=yyvsp[0].s; 
        tmp.reg.swizzlemask="";
        tmp.reg.negate="";
	iLanguage->AddDecl(
            new DeclareSampleRegister(
               createPS2Register(tmp.reg)));
;}
    break;

  case 10:
#line 299 "ps2arb_gram.y"
    {       
        YYSTYPE tmp;
        tmp.reg.reg=yyvsp[-8].s; 
        tmp.reg.swizzlemask="";
        tmp.reg.negate="";
	iLanguage->AddDecl(
           new DefineConstantRegister(
             createPS2Register(tmp.reg),
		yyvsp[-6].f, yyvsp[-4].f, yyvsp[-2].f, yyvsp[0].f));
;}
    break;

  case 16:
#line 311 "ps2arb_gram.y"
    {;}
    break;

  case 17:
#line 314 "ps2arb_gram.y"
    {
        iLanguage->AddInst((*trinary_factory[yyvsp[-8].s])(
	     AdjustInstructionFlags(yyvsp[-7].s),
             createPS2Register(yyvsp[-6].reg),
             createPS2Register(yyvsp[-4].reg),
             createPS2Register(yyvsp[-2].reg),
             createPS2Register(yyvsp[0].reg)));
;}
    break;

  case 18:
#line 324 "ps2arb_gram.y"
    {
        iLanguage->AddInst((*binary_factory[yyvsp[-6].s])(
             AdjustInstructionFlags(yyvsp[-5].s),
             createPS2Register(yyvsp[-4].reg),
             createPS2Register(yyvsp[-2].reg),
             createPS2Register(yyvsp[0].reg)));
;}
    break;

  case 19:
#line 333 "ps2arb_gram.y"
    {
        iLanguage->AddInst((*unary_factory[yyvsp[-4].s])(
             AdjustInstructionFlags(yyvsp[-3].s),
             createPS2Register(yyvsp[-2].reg),
             createPS2Register(yyvsp[0].reg)));
;}
    break;

  case 20:
#line 341 "ps2arb_gram.y"
    {;}
    break;

  case 21:
#line 343 "ps2arb_gram.y"
    {;}
    break;

  case 22:
#line 346 "ps2arb_gram.y"
    {
        iLanguage->AddInst((*unary_factory[yyvsp[-5].s])(
             AdjustInstructionFlags(yyvsp[-4].s),
             createPS2Register(yyvsp[-3].reg),
             createPS2Register(yyvsp[-1].reg)));
;}
    break;

  case 23:
#line 354 "ps2arb_gram.y"
    {
        iLanguage->AddInst((*void_factory[yyvsp[-2].s])(
             AdjustInstructionFlags(yyvsp[-1].s),
             createPS2Register(yyvsp[0].reg)));
;}
    break;

  case 24:
#line 361 "ps2arb_gram.y"
    {
        iLanguage->AddInst((*unary_factory[yyvsp[-4].s])(
             AdjustInstructionFlags(yyvsp[-3].s),
             createPS2Register(yyvsp[-2].reg),
             createPS2Register(yyvsp[0].reg)));
;}
    break;

  case 25:
#line 369 "ps2arb_gram.y"
    {
        yyval.s=yyvsp[0].s;
;}
    break;

  case 26:
#line 373 "ps2arb_gram.y"
    {
        yyval.s="";
;}
    break;

  case 27:
#line 378 "ps2arb_gram.y"
    {
        yyval.s=yyvsp[0].s;
;}
    break;

  case 28:
#line 382 "ps2arb_gram.y"
    {
        yyval.s="";
;}
    break;

  case 31:
#line 391 "ps2arb_gram.y"
    {
        DEFINERETVAL; 
        ret.reg.reg=yyvsp[-1].s;
        ret.reg.swizzlemask=yyvsp[0].s;
        ret.reg.negate="";
        yyval.reg=ret.reg;
;}
    break;

  case 32:
#line 400 "ps2arb_gram.y"
    {
        DEFINERETVAL;
        ret.reg = yyvsp[0].reg;
        yyval.reg = ret.reg;
;}
    break;

  case 33:
#line 406 "ps2arb_gram.y"
    {
        DEFINERETVAL;
        ret.reg.reg = yyvsp[0].s;
        ret.reg.swizzlemask = "";
        ret.reg.negate = "";
        yyval.reg = ret.reg;
;}
    break;

  case 34:
#line 415 "ps2arb_gram.y"
    {
        DEFINERETVAL;
        ret.reg.reg=yyvsp[-1].s;
        ret.reg.swizzlemask = yyvsp[0].s;
        ret.reg.negate="";
        yyval.reg=ret.reg;
;}
    break;

  case 35:
#line 424 "ps2arb_gram.y"
    {
        yyval.s=yyvsp[0].s;
;}
    break;

  case 36:
#line 428 "ps2arb_gram.y"
    {
        yyval.s="";
;}
    break;

  case 37:
#line 433 "ps2arb_gram.y"
    {
        DEFINERETVAL;
        ret.reg.negate = yyvsp[-2].s;
        ret.reg.reg = yyvsp[-1].s;
        ret.reg.swizzlemask = yyvsp[0].s;
        yyval.reg=ret.reg
;}
    break;

  case 43:
#line 445 "ps2arb_gram.y"
    {
        iLanguage->AddCommentOrNewline(
           new Newline((int)yyvsp[-1].f));
        yyval.f = yyvsp[-1].f +yyvsp[0].f;
;}
    break;

  case 44:
#line 451 "ps2arb_gram.y"
    {
        iLanguage->AddCommentOrNewline(
           new Newline((int)yyvsp[0].f));
        yyval.f = yyvsp[0].f;
;}
    break;

  case 45:
#line 457 "ps2arb_gram.y"
    {
        iLanguage->AddCommentOrNewline(
           new Comment(yyvsp[-1].s));
        yyval.f = 1 + yyvsp[0].f;
;}
    break;

  case 46:
#line 463 "ps2arb_gram.y"
    {
        iLanguage->AddCommentOrNewline(
          new Comment(yyvsp[0].s));
        yyval.f = 1; 
;}
    break;

  case 47:
#line 470 "ps2arb_gram.y"
    {
        iLanguage->AddCommentOrNewline(new Comment(yyvsp[0].s));
;}
    break;

  case 49:
#line 477 "ps2arb_gram.y"
    {
        yyval.f = yyvsp[0].f;
;}
    break;

  case 50:
#line 481 "ps2arb_gram.y"
    {
        yyval.f = 0;
;}
    break;


    }

/* Line 999 of yacc.c.  */
#line 1630 "ps2arb_gram.tab.c"

  yyvsp -= yylen;
  yyssp -= yylen;


  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (YYPACT_NINF < yyn && yyn < YYLAST)
	{
	  YYSIZE_T yysize = 0;
	  int yytype = YYTRANSLATE (yychar);
	  const char* yyprefix;
	  char *yymsg;
	  int yyx;

	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  int yyxbegin = yyn < 0 ? -yyn : 0;

	  /* Stay within bounds of both yycheck and yytname.  */
	  int yychecklim = YYLAST - yyn;
	  int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
	  int yycount = 0;

	  yyprefix = ", expecting ";
	  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	      {
		yysize += yystrlen (yyprefix) + yystrlen (yytname [yyx]);
		yycount += 1;
		if (yycount == 5)
		  {
		    yysize = 0;
		    break;
		  }
	      }
	  yysize += (sizeof ("syntax error, unexpected ")
		     + yystrlen (yytname[yytype]));
	  yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg != 0)
	    {
	      char *yyp = yystpcpy (yymsg, "syntax error, unexpected ");
	      yyp = yystpcpy (yyp, yytname[yytype]);

	      if (yycount < 5)
		{
		  yyprefix = ", expecting ";
		  for (yyx = yyxbegin; yyx < yyxend; ++yyx)
		    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
		      {
			yyp = yystpcpy (yyp, yyprefix);
			yyp = yystpcpy (yyp, yytname[yyx]);
			yyprefix = " or ";
		      }
		}
	      yyerror (yymsg);
	      YYSTACK_FREE (yymsg);
	    }
	  else
	    yyerror ("syntax error; also virtual memory exhausted");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror ("syntax error");
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      /* Return failure if at end of input.  */
      if (yychar == YYEOF)
        {
	  /* Pop the error token.  */
          YYPOPSTACK;
	  /* Pop the rest of the stack.  */
	  while (yyss < yyssp)
	    {
	      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
	      yydestruct (yystos[*yyssp], yyvsp);
	      YYPOPSTACK;
	    }
	  YYABORT;
        }

      YYDSYMPRINTF ("Error: discarding", yytoken, &yylval, &yylloc);
      yydestruct (yytoken, &yylval);
      yychar = YYEMPTY;

    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*----------------------------------------------------.
| yyerrlab1 -- error raised explicitly by an action.  |
`----------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;

      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
      yydestruct (yystos[yystate], yyvsp);
      yyvsp--;
      yystate = *--yyssp;

      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  YYDPRINTF ((stderr, "Shifting error token, "));

  *++yyvsp = yylval;


  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*----------------------------------------------.
| yyoverflowlab -- parser overflow comes here.  |
`----------------------------------------------*/
yyoverflowlab:
  yyerror ("parser stack overflow");
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  return yyresult;
}


#line 485 "ps2arb_gram.y"






