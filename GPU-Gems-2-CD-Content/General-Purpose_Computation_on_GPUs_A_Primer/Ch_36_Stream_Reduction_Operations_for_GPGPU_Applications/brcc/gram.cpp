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
#define YYPURE 1

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     IDENT = 258,
     TAG_NAME = 259,
     LABEL_NAME = 260,
     TYPEDEF_NAME = 261,
     STRING = 262,
     LSTRING = 263,
     CHAR_CONST = 264,
     LCHAR_CONST = 265,
     INUM = 266,
     RNUM = 267,
     PP_DIR = 268,
     PP_LINE = 269,
     INVALID = 270,
     CONST = 271,
     VOLATILE = 272,
     OUT = 273,
     REDUCE = 274,
     VOUT = 275,
     ITER = 276,
     KERNEL = 277,
     AUTO = 278,
     EXTRN = 279,
     REGISTR = 280,
     STATIC = 281,
     TYPEDEF = 282,
     VOID = 283,
     CHAR = 284,
     SHORT = 285,
     INT = 286,
     LONG = 287,
     SGNED = 288,
     UNSGNED = 289,
     FLOAT = 290,
     FLOAT2 = 291,
     FLOAT3 = 292,
     FLOAT4 = 293,
     FIXED = 294,
     FIXED2 = 295,
     FIXED3 = 296,
     FIXED4 = 297,
     HALF = 298,
     HALF2 = 299,
     HALF3 = 300,
     HALF4 = 301,
     DOUBLE = 302,
     DOUBLE2 = 303,
     ENUM = 304,
     STRUCT = 305,
     UNION = 306,
     BREAK = 307,
     CASE = 308,
     CONT = 309,
     DEFLT = 310,
     DO = 311,
     ELSE = 312,
     IF = 313,
     FOR = 314,
     GOTO = 315,
     RETURN = 316,
     SWITCH = 317,
     WHILE = 318,
     PLUS_EQ = 319,
     MINUS_EQ = 320,
     STAR_EQ = 321,
     DIV_EQ = 322,
     MOD_EQ = 323,
     B_AND_EQ = 324,
     B_OR_EQ = 325,
     B_XOR_EQ = 326,
     L_SHIFT_EQ = 327,
     R_SHIFT_EQ = 328,
     EQUAL = 329,
     LESS_EQ = 330,
     GRTR_EQ = 331,
     NOT_EQ = 332,
     RPAREN = 333,
     RBRCKT = 334,
     LBRACE = 335,
     RBRACE = 336,
     SEMICOLON = 337,
     COMMA = 338,
     ELLIPSIS = 339,
     LB_SIGN = 340,
     DOUB_LB_SIGN = 341,
     BACKQUOTE = 342,
     AT = 343,
     ATTRIBUTE = 344,
     ALIGNED = 345,
     PACKED = 346,
     CDECL = 347,
     MODE = 348,
     FORMAT = 349,
     NORETURN = 350,
     COMMA_OP = 351,
     ASSIGN = 352,
     EQ = 353,
     COMMA_SEP = 354,
     COLON = 355,
     QUESTMARK = 356,
     OR = 357,
     AND = 358,
     B_OR = 359,
     B_XOR = 360,
     B_AND = 361,
     COMP_EQ = 362,
     GRTR = 363,
     LESS = 364,
     COMP_GRTR = 365,
     COMP_LESS = 366,
     COMP_ARITH = 367,
     R_SHIFT = 368,
     L_SHIFT = 369,
     MINUS = 370,
     PLUS = 371,
     MOD = 372,
     DIV = 373,
     STAR = 374,
     CAST = 375,
     DECR = 376,
     INCR = 377,
     INDEXOF = 378,
     SIZEOF = 379,
     B_NOT = 380,
     NOT = 381,
     UNARY = 382,
     HYPERUNARY = 383,
     LBRCKT = 384,
     LPAREN = 385,
     DOT = 386,
     ARROW = 387
   };
#endif
#define IDENT 258
#define TAG_NAME 259
#define LABEL_NAME 260
#define TYPEDEF_NAME 261
#define STRING 262
#define LSTRING 263
#define CHAR_CONST 264
#define LCHAR_CONST 265
#define INUM 266
#define RNUM 267
#define PP_DIR 268
#define PP_LINE 269
#define INVALID 270
#define CONST 271
#define VOLATILE 272
#define OUT 273
#define REDUCE 274
#define VOUT 275
#define ITER 276
#define KERNEL 277
#define AUTO 278
#define EXTRN 279
#define REGISTR 280
#define STATIC 281
#define TYPEDEF 282
#define VOID 283
#define CHAR 284
#define SHORT 285
#define INT 286
#define LONG 287
#define SGNED 288
#define UNSGNED 289
#define FLOAT 290
#define FLOAT2 291
#define FLOAT3 292
#define FLOAT4 293
#define FIXED 294
#define FIXED2 295
#define FIXED3 296
#define FIXED4 297
#define HALF 298
#define HALF2 299
#define HALF3 300
#define HALF4 301
#define DOUBLE 302
#define DOUBLE2 303
#define ENUM 304
#define STRUCT 305
#define UNION 306
#define BREAK 307
#define CASE 308
#define CONT 309
#define DEFLT 310
#define DO 311
#define ELSE 312
#define IF 313
#define FOR 314
#define GOTO 315
#define RETURN 316
#define SWITCH 317
#define WHILE 318
#define PLUS_EQ 319
#define MINUS_EQ 320
#define STAR_EQ 321
#define DIV_EQ 322
#define MOD_EQ 323
#define B_AND_EQ 324
#define B_OR_EQ 325
#define B_XOR_EQ 326
#define L_SHIFT_EQ 327
#define R_SHIFT_EQ 328
#define EQUAL 329
#define LESS_EQ 330
#define GRTR_EQ 331
#define NOT_EQ 332
#define RPAREN 333
#define RBRCKT 334
#define LBRACE 335
#define RBRACE 336
#define SEMICOLON 337
#define COMMA 338
#define ELLIPSIS 339
#define LB_SIGN 340
#define DOUB_LB_SIGN 341
#define BACKQUOTE 342
#define AT 343
#define ATTRIBUTE 344
#define ALIGNED 345
#define PACKED 346
#define CDECL 347
#define MODE 348
#define FORMAT 349
#define NORETURN 350
#define COMMA_OP 351
#define ASSIGN 352
#define EQ 353
#define COMMA_SEP 354
#define COLON 355
#define QUESTMARK 356
#define OR 357
#define AND 358
#define B_OR 359
#define B_XOR 360
#define B_AND 361
#define COMP_EQ 362
#define GRTR 363
#define LESS 364
#define COMP_GRTR 365
#define COMP_LESS 366
#define COMP_ARITH 367
#define R_SHIFT 368
#define L_SHIFT 369
#define MINUS 370
#define PLUS 371
#define MOD 372
#define DIV 373
#define STAR 374
#define CAST 375
#define DECR 376
#define INCR 377
#define INDEXOF 378
#define SIZEOF 379
#define B_NOT 380
#define NOT 381
#define UNARY 382
#define HYPERUNARY 383
#define LBRCKT 384
#define LPAREN 385
#define DOT 386
#define ARROW 387




/* Copy the first part of user declarations.  */
#line 1 "gram.y"

 /*
 ======================================================================

    CTool Library
    Copyright (C) 1995-2001	Shaun Flisakowski

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 1, or (at your option)
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

 ======================================================================
 */

/* grammar File for C - Shaun Flisakowski and Patrick Baudin */
/* Grammar was constructed with the assistance of:
    "C - A Reference Manual" (Fourth Edition),
    by Samuel P Harbison, and Guy L Steele Jr. */

#ifdef    NO_ALLOCA
#define   alloca  __builtin_alloca
#endif

#ifdef _WIN32
/* Don't complain about switch() statements that only have a 'default' */
#pragma warning( disable : 4065 )
#endif

#include <stdio.h>
#include <errno.h>
#include <setjmp.h>

#include "lexer.h"
#include "symbol.h"
#include "token.h"
#include "stemnt.h"
#include "location.h"
#include "project.h"
#include "brtexpress.h"
extern int err_cnt;
int yylex(YYSTYPE *lvalp);

extern int err_top_level;
/* Cause the `yydebug' variable to be defined.  */
#define YYDEBUG 1
void baseTypeFixup(BaseType * bt,Decl * decl) {
  BaseType * b = decl->form->getBase();
  while ((decl=decl->next)) {
    BaseType *nb = decl->form->getBase();
    if (memcmp(nb,b,sizeof(BaseType))!=0) {
      decl->form = decl->form->dup();
      *decl->form->getBase()=*b;
    }
  }

}
/*  int  yydebug = 1;  */

/* ###################################################### */
#line 232 "gram.y"

/* 1 if we explained undeclared var errors.  */
/*  static int undeclared_variable_notice = 0;  */


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
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 426 "gram.tab.c"

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
#define YYFINAL  80
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1649

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  133
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  202
/* YYNRULES -- Number of rules. */
#define YYNRULES  403
/* YYNRULES -- Number of states. */
#define YYNSTATES  598

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   387

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
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned short yyprhs[] =
{
       0,     0,     3,     4,     6,     8,    11,    15,    16,    18,
      20,    22,    24,    27,    31,    34,    38,    42,    43,    49,
      52,    53,    55,    58,    59,    62,    65,    67,    72,    75,
      76,    78,    81,    82,    85,    88,    90,    92,    94,    96,
      98,   100,   102,   104,   106,   108,   110,   112,   114,   117,
     120,   123,   127,   129,   131,   133,   135,   137,   143,   146,
     149,   153,   157,   159,   165,   173,   181,   187,   197,   199,
     201,   203,   205,   208,   213,   215,   217,   223,   225,   229,
     233,   234,   236,   238,   239,   241,   243,   245,   249,   251,
     255,   258,   260,   264,   266,   270,   272,   276,   279,   281,
     286,   288,   292,   294,   298,   300,   304,   306,   310,   312,
     316,   323,   332,   343,   348,   353,   358,   360,   362,   369,
     374,   376,   378,   380,   382,   384,   386,   388,   390,   392,
     394,   399,   402,   405,   410,   413,   416,   419,   422,   425,
     428,   430,   434,   436,   438,   440,   444,   448,   450,   452,
     454,   456,   458,   460,   462,   467,   469,   471,   474,   477,
     479,   483,   487,   492,   493,   495,   497,   499,   503,   507,
     509,   511,   513,   515,   517,   519,   521,   523,   525,   527,
     529,   531,   533,   535,   537,   539,   541,   543,   545,   546,
     548,   549,   550,   554,   555,   557,   558,   559,   563,   566,
     570,   573,   576,   579,   582,   583,   585,   587,   588,   591,
     593,   596,   598,   600,   602,   604,   606,   608,   610,   612,
     614,   615,   617,   620,   621,   625,   628,   629,   631,   632,
     636,   639,   641,   642,   646,   648,   652,   653,   655,   657,
     659,   663,   665,   667,   671,   673,   675,   677,   682,   683,
     685,   687,   689,   691,   693,   695,   697,   699,   704,   706,
     709,   710,   712,   714,   716,   718,   720,   722,   724,   726,
     728,   730,   732,   734,   736,   738,   740,   742,   744,   746,
     748,   750,   752,   754,   756,   758,   760,   762,   764,   766,
     768,   770,   772,   775,   778,   781,   784,   789,   794,   797,
     802,   807,   810,   815,   820,   821,   823,   824,   827,   828,
     830,   832,   834,   838,   840,   844,   846,   847,   848,   852,
     855,   858,   862,   865,   867,   869,   870,   874,   875,   881,
     883,   885,   887,   888,   893,   895,   896,   898,   900,   902,
     905,   907,   909,   911,   915,   917,   919,   924,   929,   933,
     938,   943,   945,   947,   951,   953,   954,   956,   960,   963,
     965,   968,   970,   971,   974,   976,   980,   982,   984,   986,
     988,   989,   990,   993,   994,   997,   999,  1003,  1005,  1009,
    1010,  1013,  1016,  1019,  1021,  1023,  1025,  1028,  1030,  1034,
    1038,  1043,  1047,  1052,  1053,  1055,  1062,  1063,  1065,  1067,
    1069,  1071,  1076,  1081
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
     134,     0,    -1,    -1,   135,    -1,     1,    -1,   137,   136,
      -1,   135,   137,   136,    -1,    -1,   233,    -1,   138,    -1,
      13,    -1,    14,    -1,     1,    82,    -1,     1,    81,   136,
      -1,   139,   140,    -1,   237,   303,   224,    -1,   236,   302,
     224,    -1,    -1,    80,   141,   228,   142,    81,    -1,     1,
      81,    -1,    -1,   143,    -1,   151,   144,    -1,    -1,   144,
     145,    -1,   144,    14,    -1,   150,    -1,    80,   228,   147,
      81,    -1,     1,    81,    -1,    -1,   148,    -1,   151,   149,
      -1,    -1,   149,   150,    -1,   149,    14,    -1,   151,    -1,
     152,    -1,   153,    -1,   154,    -1,   146,    -1,   155,    -1,
     156,    -1,   157,    -1,   158,    -1,   159,    -1,   160,    -1,
     161,    -1,   162,    -1,     1,    82,    -1,   191,    82,    -1,
     177,    82,    -1,   168,   100,   150,    -1,   163,    -1,   164,
      -1,   165,    -1,   166,    -1,   167,    -1,    62,   130,   177,
      78,   150,    -1,    52,    82,    -1,    54,    82,    -1,    61,
     176,    82,    -1,    60,     5,    82,    -1,    82,    -1,    58,
     130,   177,    78,   150,    -1,    58,   130,   177,    78,   150,
      57,   150,    -1,    56,   150,    63,   130,   177,    78,    82,
      -1,    63,   130,   177,    78,   150,    -1,    59,   130,   176,
      82,   176,    82,   176,    78,   150,    -1,   169,    -1,   170,
      -1,   171,    -1,   317,    -1,    53,   175,    -1,    53,   175,
      84,   175,    -1,    55,    -1,   178,    -1,   178,   101,   177,
     100,   172,    -1,   172,    -1,   194,   222,   173,    -1,   194,
     222,   191,    -1,    -1,   175,    -1,   177,    -1,    -1,   177,
      -1,   203,    -1,   179,    -1,   178,   102,   179,    -1,   181,
      -1,   179,   103,   181,    -1,   126,   185,    -1,   182,    -1,
     181,   104,   182,    -1,   183,    -1,   182,   105,   183,    -1,
     186,    -1,   183,   106,   186,    -1,   125,   185,    -1,   194,
      -1,   130,   241,    78,   185,    -1,   187,    -1,   186,   219,
     187,    -1,   188,    -1,   187,   220,   188,    -1,   189,    -1,
     188,   221,   189,    -1,   190,    -1,   189,   217,   190,    -1,
     185,    -1,   190,   218,   185,    -1,    36,   130,   173,    83,
     173,    78,    -1,    37,   130,   173,    83,   173,    83,   173,
      78,    -1,    38,   130,   173,    83,   173,    83,   173,    83,
     173,    78,    -1,    36,   130,     1,    78,    -1,    37,   130,
       1,    78,    -1,    38,   130,     1,    78,    -1,   173,    -1,
     191,    -1,    21,   130,   192,    83,   192,    78,    -1,    21,
     130,     1,    78,    -1,   206,    -1,   195,    -1,   197,    -1,
     198,    -1,   180,    -1,   184,    -1,   199,    -1,   200,    -1,
     201,    -1,   202,    -1,   124,   130,   239,    78,    -1,   124,
     194,    -1,   123,   317,    -1,   123,   130,   317,    78,    -1,
     115,   185,    -1,   116,   185,    -1,   106,   185,    -1,   119,
     185,    -1,   122,   194,    -1,   121,   194,    -1,   173,    -1,
     203,    83,   173,    -1,   317,    -1,   205,    -1,   223,    -1,
     130,   177,    78,    -1,   130,     1,    78,    -1,   204,    -1,
     207,    -1,   208,    -1,   214,    -1,   209,    -1,   210,    -1,
     196,    -1,   206,   129,   177,    79,    -1,   212,    -1,   213,
      -1,   206,   122,    -1,   206,   121,    -1,   319,    -1,   206,
     131,   211,    -1,   206,   132,   211,    -1,   206,   130,   215,
      78,    -1,    -1,   216,    -1,   173,    -1,   191,    -1,   216,
      83,   173,    -1,   216,    83,   191,    -1,   116,    -1,   115,
      -1,   119,    -1,   118,    -1,   117,    -1,   107,    -1,   112,
      -1,   111,    -1,   110,    -1,   114,    -1,   113,    -1,    98,
      -1,    97,    -1,    11,    -1,    12,    -1,     9,    -1,    10,
      -1,     7,    -1,     8,    -1,    -1,   225,    -1,    -1,    -1,
     226,   227,   232,    -1,    -1,   229,    -1,    -1,    -1,   230,
     231,   232,    -1,   235,    82,    -1,   235,    82,   232,    -1,
     234,    82,    -1,   235,    82,    -1,   236,   257,    -1,   237,
     257,    -1,    -1,   242,    -1,   329,    -1,    -1,   240,   241,
      -1,   242,    -1,   242,   238,    -1,   248,    -1,    24,    -1,
      26,    -1,    27,    -1,    23,    -1,    25,    -1,   243,    -1,
     244,    -1,   268,    -1,    -1,   248,    -1,   245,   247,    -1,
      -1,   246,   249,   247,    -1,   264,   247,    -1,    -1,   251,
      -1,    -1,   268,   252,   250,    -1,   264,   250,    -1,   251,
      -1,    -1,   302,   255,   332,    -1,   254,    -1,   254,    98,
     260,    -1,    -1,   258,    -1,   259,    -1,   256,    -1,   259,
      83,   256,    -1,   262,    -1,   262,    -1,   261,    83,   262,
      -1,   173,    -1,   191,    -1,   193,    -1,    80,   261,   263,
      81,    -1,    -1,    83,    -1,   265,    -1,    16,    -1,    17,
      -1,    18,    -1,    19,    -1,    21,    -1,    22,    -1,    20,
     129,   174,    79,    -1,   265,    -1,   266,   265,    -1,    -1,
     266,    -1,   279,    -1,   275,    -1,   277,    -1,   273,    -1,
     271,    -1,   272,    -1,   269,    -1,    28,    -1,    29,    -1,
      30,    -1,    31,    -1,    32,    -1,    35,    -1,    36,    -1,
      37,    -1,    38,    -1,    39,    -1,    40,    -1,    41,    -1,
      42,    -1,    43,    -1,    44,    -1,    45,    -1,    46,    -1,
      47,    -1,    48,    -1,    33,    -1,    34,    -1,     6,    -1,
       4,    -1,    50,   270,    -1,    51,   270,    -1,    49,   270,
      -1,    50,   270,    -1,    50,    80,   280,    81,    -1,   274,
      80,   280,    81,    -1,    51,   270,    -1,    51,    80,   280,
      81,    -1,   276,    80,   280,    81,    -1,    49,   270,    -1,
      49,    80,   281,    81,    -1,   278,    80,   281,    81,    -1,
      -1,   287,    -1,    -1,   283,   282,    -1,    -1,    83,    -1,
     284,    -1,   285,    -1,   283,    83,   285,    -1,   286,    -1,
     286,    98,   173,    -1,   319,    -1,    -1,    -1,   288,   289,
     290,    -1,   291,    82,    -1,   290,    82,    -1,   290,   291,
      82,    -1,   253,   292,    -1,   253,    -1,   293,    -1,    -1,
     294,   296,   332,    -1,    -1,   293,    83,   295,   296,   332,
      -1,   297,    -1,   298,    -1,   302,    -1,    -1,   301,   100,
     299,   300,    -1,   172,    -1,    -1,   302,    -1,   304,    -1,
     304,    -1,   313,   305,    -1,   305,    -1,   306,    -1,   317,
      -1,   130,   304,    78,    -1,   307,    -1,   308,    -1,   306,
     130,   322,    78,    -1,   306,   130,   314,    78,    -1,   306,
     130,    78,    -1,   306,   129,   174,    79,    -1,   306,   111,
     310,   110,    -1,   223,    -1,   214,    -1,   130,   177,    78,
      -1,   317,    -1,    -1,   309,    -1,   310,    83,   309,    -1,
     119,   267,    -1,   311,    -1,   312,   311,    -1,   312,    -1,
      -1,   315,   316,    -1,   317,    -1,   316,    83,   317,    -1,
       3,    -1,     6,    -1,   317,    -1,   318,    -1,    -1,    -1,
     321,   324,    -1,    -1,   323,   324,    -1,   325,    -1,   325,
      83,    84,    -1,   326,    -1,   325,    83,   326,    -1,    -1,
     327,   328,    -1,   242,   302,    -1,   242,   329,    -1,   242,
      -1,   313,    -1,   330,    -1,   313,   330,    -1,   331,    -1,
     130,   329,    78,    -1,   129,   174,    79,    -1,   331,   129,
     174,    79,    -1,   130,   320,    78,    -1,   331,   130,   320,
      78,    -1,    -1,   333,    -1,    89,   130,   130,   334,    78,
      78,    -1,    -1,    91,    -1,    92,    -1,    16,    -1,    95,
      -1,    90,   130,    11,    78,    -1,    93,   130,   317,    78,
      -1,    94,   130,   317,    83,    11,    83,    11,    78,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   244,   244,   251,   262,   270,   275,   282,   290,   294,
     298,   302,   306,   310,   316,   344,   373,   407,   406,   424,
     431,   434,   437,   456,   459,   468,   484,   487,   497,   504,
     507,   510,   529,   532,   541,   552,   556,   562,   563,   564,
     565,   566,   567,   568,   569,   570,   571,   572,   573,   596,
     603,   610,   624,   625,   628,   629,   630,   633,   642,   650,
     658,   666,   674,   681,   690,   700,   711,   720,   732,   733,
     734,   737,   745,   750,   758,   774,   775,   783,   784,   788,
     795,   798,   801,   805,   808,   811,   814,   815,   822,   823,
     830,   837,   838,   845,   846,   853,   854,   861,   868,   869,
     877,   878,   884,   885,   891,   892,   898,   899,   905,   906,
     912,   920,   930,   941,   945,   949,   955,   956,   959,   976,
     982,   983,   984,   985,   986,   987,   988,   989,   990,   991,
     994,  1001,  1008,  1012,  1032,  1038,  1045,  1052,  1058,  1065,
    1072,  1073,  1080,  1084,  1085,  1091,  1097,  1105,  1106,  1107,
    1108,  1109,  1110,  1111,  1115,  1123,  1124,  1127,  1134,  1141,
    1145,  1168,  1191,  1205,  1208,  1211,  1212,  1213,  1220,  1229,
    1230,  1233,  1234,  1235,  1238,  1241,  1242,  1243,  1246,  1247,
    1250,  1251,  1254,  1255,  1256,  1257,  1258,  1259,  1268,  1268,
    1279,  1284,  1279,  1300,  1300,  1310,  1315,  1310,  1323,  1328,
    1341,  1348,  1356,  1374,  1400,  1406,  1415,  1419,  1419,  1430,
    1439,  1462,  1468,  1469,  1470,  1473,  1474,  1477,  1478,  1488,
    1492,  1495,  1498,  1514,  1514,  1543,  1564,  1567,  1570,  1570,
    1580,  1595,  1605,  1604,  1615,  1616,  1624,  1627,  1630,  1633,
    1637,  1653,  1655,  1660,  1668,  1669,  1670,  1671,  1680,  1683,
    1695,  1698,  1699,  1700,  1701,  1702,  1703,  1704,  1712,  1713,
    1722,  1725,  1732,  1733,  1734,  1735,  1736,  1737,  1738,  1739,
    1740,  1741,  1742,  1743,  1744,  1745,  1746,  1747,  1748,  1749,
    1750,  1751,  1752,  1753,  1754,  1755,  1756,  1757,  1758,  1759,
    1762,  1769,  1777,  1787,  1797,  1807,  1817,  1824,  1848,  1858,
    1866,  1891,  1901,  1908,  1932,  1935,  1939,  1942,  1948,  1951,
    1961,  1964,  1969,  1977,  1989,  2003,  2010,  2015,  2010,  2029,
    2035,  2043,  2051,  2056,  2067,  2071,  2071,  2079,  2078,  2090,
    2108,  2119,  2123,  2122,  2142,  2146,  2149,  2156,  2162,  2168,
    2173,  2176,  2179,  2184,  2190,  2191,  2192,  2208,  2224,  2248,
    2262,  2273,  2277,  2281,  2285,  2292,  2295,  2299,  2309,  2315,
    2316,  2323,  2330,  2330,  2345,  2350,  2362,  2366,  2378,  2379,
    2388,  2392,  2392,  2401,  2401,  2414,  2415,  2424,  2425,  2433,
    2433,  2443,  2459,  2476,  2489,  2493,  2497,  2504,  2507,  2511,
    2515,  2525,  2529,  2546,  2549,  2552,  2563,  2567,  2571,  2575,
    2579,  2583,  2597,  2606
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "IDENT", "TAG_NAME", "LABEL_NAME", 
  "TYPEDEF_NAME", "STRING", "LSTRING", "CHAR_CONST", "LCHAR_CONST", 
  "INUM", "RNUM", "PP_DIR", "PP_LINE", "INVALID", "CONST", "VOLATILE", 
  "OUT", "REDUCE", "VOUT", "ITER", "KERNEL", "AUTO", "EXTRN", "REGISTR", 
  "STATIC", "TYPEDEF", "VOID", "CHAR", "SHORT", "INT", "LONG", "SGNED", 
  "UNSGNED", "FLOAT", "FLOAT2", "FLOAT3", "FLOAT4", "FIXED", "FIXED2", 
  "FIXED3", "FIXED4", "HALF", "HALF2", "HALF3", "HALF4", "DOUBLE", 
  "DOUBLE2", "ENUM", "STRUCT", "UNION", "BREAK", "CASE", "CONT", "DEFLT", 
  "DO", "ELSE", "IF", "FOR", "GOTO", "RETURN", "SWITCH", "WHILE", 
  "PLUS_EQ", "MINUS_EQ", "STAR_EQ", "DIV_EQ", "MOD_EQ", "B_AND_EQ", 
  "B_OR_EQ", "B_XOR_EQ", "L_SHIFT_EQ", "R_SHIFT_EQ", "EQUAL", "LESS_EQ", 
  "GRTR_EQ", "NOT_EQ", "RPAREN", "RBRCKT", "LBRACE", "RBRACE", 
  "SEMICOLON", "COMMA", "ELLIPSIS", "LB_SIGN", "DOUB_LB_SIGN", 
  "BACKQUOTE", "AT", "ATTRIBUTE", "ALIGNED", "PACKED", "CDECL", "MODE", 
  "FORMAT", "NORETURN", "COMMA_OP", "ASSIGN", "EQ", "COMMA_SEP", "COLON", 
  "QUESTMARK", "OR", "AND", "B_OR", "B_XOR", "B_AND", "COMP_EQ", "GRTR", 
  "LESS", "COMP_GRTR", "COMP_LESS", "COMP_ARITH", "R_SHIFT", "L_SHIFT", 
  "MINUS", "PLUS", "MOD", "DIV", "STAR", "CAST", "DECR", "INCR", 
  "INDEXOF", "SIZEOF", "B_NOT", "NOT", "UNARY", "HYPERUNARY", "LBRCKT", 
  "LPAREN", "DOT", "ARROW", "$accept", "program", "trans_unit", 
  "top_level_exit", "top_level_decl", "func_def", "func_spec", 
  "cmpnd_stemnt", "@1", "opt_stemnt_list", "stemnt_list", "stemnt_list2", 
  "stemnt", "cmpnd_stemnt_reentrance", "opt_stemnt_list_reentrance", 
  "stemnt_list_reentrance", "stemnt_list_reentrance2", 
  "stemnt_reentrance", "non_constructor_stemnt", "constructor_stemnt", 
  "expr_stemnt", "labeled_stemnt", "cond_stemnt", "iter_stemnt", 
  "switch_stemnt", "break_stemnt", "continue_stemnt", "return_stemnt", 
  "goto_stemnt", "null_stemnt", "if_stemnt", "if_else_stemnt", 
  "do_stemnt", "while_stemnt", "for_stemnt", "label", "named_label", 
  "case_label", "deflt_label", "cond_expr", "assign_expr", 
  "opt_const_expr", "const_expr", "opt_expr", "expr", "log_or_expr", 
  "log_and_expr", "log_neg_expr", "bitwise_or_expr", "bitwise_xor_expr", 
  "bitwise_and_expr", "bitwise_neg_expr", "cast_expr", "equality_expr", 
  "relational_expr", "shift_expr", "additive_expr", "mult_expr", 
  "constructor_expr", "iter_constructor_arg", "iter_constructor_expr", 
  "unary_expr", "sizeof_expr", "indexof_expr", "unary_minus_expr", 
  "unary_plus_expr", "addr_expr", "indirection_expr", "preinc_expr", 
  "predec_expr", "comma_expr", "prim_expr", "paren_expr", "postfix_expr", 
  "subscript_expr", "comp_select_expr", "postinc_expr", "postdec_expr", 
  "field_ident", "direct_comp_select", "indirect_comp_select", 
  "func_call", "opt_expr_list", "expr_list", "add_op", "mult_op", 
  "equality_op", "relation_op", "shift_op", "assign_op", "constant", 
  "opt_KnR_declaration_list", "@2", "@3", "@4", "opt_declaration_list", 
  "@5", "@6", "@7", "declaration_list", "decl_stemnt", 
  "old_style_declaration", "declaration", "no_decl_specs", "decl_specs", 
  "abs_decl", "type_name", "@8", "type_name_bis", 
  "decl_specs_reentrance_bis", "local_or_global_storage_class", 
  "local_storage_class", "storage_class", "type_spec", 
  "opt_decl_specs_reentrance", "decl_specs_reentrance", "@9", 
  "opt_comp_decl_specs", "comp_decl_specs_reentrance", "@10", 
  "comp_decl_specs", "decl", "@11", "init_decl", "opt_init_decl_list", 
  "init_decl_list", "init_decl_list_reentrance", "initializer", 
  "initializer_list", "initializer_reentrance", "opt_comma", "type_qual", 
  "type_qual_token", "type_qual_list", "opt_type_qual_list", 
  "type_spec_reentrance", "typedef_name", "tag_ref", "struct_tag_ref", 
  "union_tag_ref", "enum_tag_ref", "struct_tag_def", "struct_type_define", 
  "union_tag_def", "union_type_define", "enum_tag_def", 
  "enum_type_define", "struct_or_union_definition", "enum_definition", 
  "opt_trailing_comma", "enum_def_list", "enum_def_list_reentrance", 
  "enum_const_def", "enum_constant", "field_list", "@12", "@13", 
  "field_list_reentrance", "comp_decl", "comp_decl_list", 
  "comp_decl_list_reentrance", "@14", "@15", "comp_declarator", 
  "simple_comp", "bit_field", "@16", "width", "opt_declarator", 
  "declarator", "func_declarator", "declarator_reentrance_bis", 
  "direct_declarator_reentrance_bis", "direct_declarator_reentrance", 
  "array_decl", "stream_decl", "dimension_constraint", "comma_constants", 
  "pointer_start", "pointer_reentrance", "pointer", "ident_list", "@17", 
  "ident_list_reentrance", "ident", "typename_as_ident", "any_ident", 
  "opt_param_type_list", "@18", "param_type_list", "@19", 
  "param_type_list_bis", "param_list", "param_decl", "@20", 
  "param_decl_bis", "abs_decl_reentrance", 
  "direct_abs_decl_reentrance_bis", "direct_abs_decl_reentrance", 
  "opt_gcc_attrib", "gcc_attrib", "gcc_inner", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned short yyr1[] =
{
       0,   133,   134,   134,   134,   135,   135,   136,   137,   137,
     137,   137,   137,   137,   138,   139,   139,   141,   140,   140,
     142,   142,   143,   144,   144,   144,   145,   146,   146,   147,
     147,   148,   149,   149,   149,   150,   150,   151,   151,   151,
     151,   151,   151,   151,   151,   151,   151,   151,   151,   152,
     153,   154,   155,   155,   156,   156,   156,   157,   158,   159,
     160,   161,   162,   163,   164,   165,   166,   167,   168,   168,
     168,   169,   170,   170,   171,   172,   172,   173,   173,   173,
     174,   174,   175,   176,   176,   177,   178,   178,   179,   179,
     180,   181,   181,   182,   182,   183,   183,   184,   185,   185,
     186,   186,   187,   187,   188,   188,   189,   189,   190,   190,
     191,   191,   191,   191,   191,   191,   192,   192,   193,   193,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   194,
     195,   195,   196,   196,   197,   198,   199,   200,   201,   202,
     203,   203,   204,   204,   204,   205,   205,   206,   206,   206,
     206,   206,   206,   206,   207,   208,   208,   209,   210,   211,
     212,   213,   214,   215,   215,   216,   216,   216,   216,   217,
     217,   218,   218,   218,   219,   220,   220,   220,   221,   221,
     222,   222,   223,   223,   223,   223,   223,   223,   225,   224,
     226,   227,   224,   229,   228,   230,   231,   228,   232,   232,
     233,   233,   234,   235,   236,   237,   238,   240,   239,   241,
     241,   242,   243,   243,   243,   244,   244,   245,   245,   246,
     247,   247,   248,   249,   248,   248,   250,   250,   252,   251,
     251,   253,   255,   254,   256,   256,   257,   257,   258,   259,
     259,   260,   261,   261,   262,   262,   262,   262,   263,   263,
     264,   265,   265,   265,   265,   265,   265,   265,   266,   266,
     267,   267,   268,   268,   268,   268,   268,   268,   268,   268,
     268,   268,   268,   268,   268,   268,   268,   268,   268,   268,
     268,   268,   268,   268,   268,   268,   268,   268,   268,   268,
     269,   270,   271,   272,   273,   274,   275,   275,   276,   277,
     277,   278,   279,   279,   280,   280,   281,   281,   282,   282,
     283,   284,   284,   285,   285,   286,   288,   289,   287,   290,
     290,   290,   291,   291,   292,   294,   293,   295,   293,   296,
     296,   297,   299,   298,   300,   301,   301,   302,   303,   304,
     304,   305,   306,   306,   306,   306,   306,   306,   306,   307,
     308,   309,   309,   309,   309,   310,   310,   310,   311,   312,
     312,   313,   315,   314,   316,   316,   317,   318,   319,   319,
     320,   321,   320,   323,   322,   324,   324,   325,   325,   327,
     326,   328,   328,   328,   329,   329,   329,   330,   331,   331,
     331,   331,   331,   332,   332,   333,   334,   334,   334,   334,
     334,   334,   334,   334
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     0,     1,     1,     2,     3,     0,     1,     1,
       1,     1,     2,     3,     2,     3,     3,     0,     5,     2,
       0,     1,     2,     0,     2,     2,     1,     4,     2,     0,
       1,     2,     0,     2,     2,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     2,
       2,     3,     1,     1,     1,     1,     1,     5,     2,     2,
       3,     3,     1,     5,     7,     7,     5,     9,     1,     1,
       1,     1,     2,     4,     1,     1,     5,     1,     3,     3,
       0,     1,     1,     0,     1,     1,     1,     3,     1,     3,
       2,     1,     3,     1,     3,     1,     3,     2,     1,     4,
       1,     3,     1,     3,     1,     3,     1,     3,     1,     3,
       6,     8,    10,     4,     4,     4,     1,     1,     6,     4,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       4,     2,     2,     4,     2,     2,     2,     2,     2,     2,
       1,     3,     1,     1,     1,     3,     3,     1,     1,     1,
       1,     1,     1,     1,     4,     1,     1,     2,     2,     1,
       3,     3,     4,     0,     1,     1,     1,     3,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       0,     0,     3,     0,     1,     0,     0,     3,     2,     3,
       2,     2,     2,     2,     0,     1,     1,     0,     2,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       0,     1,     2,     0,     3,     2,     0,     1,     0,     3,
       2,     1,     0,     3,     1,     3,     0,     1,     1,     1,
       3,     1,     1,     3,     1,     1,     1,     4,     0,     1,
       1,     1,     1,     1,     1,     1,     1,     4,     1,     2,
       0,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     2,     2,     2,     4,     4,     2,     4,
       4,     2,     4,     4,     0,     1,     0,     2,     0,     1,
       1,     1,     3,     1,     3,     1,     0,     0,     3,     2,
       2,     3,     2,     1,     1,     0,     3,     0,     5,     1,
       1,     1,     0,     4,     1,     0,     1,     1,     1,     2,
       1,     1,     1,     3,     1,     1,     4,     4,     3,     4,
       4,     1,     1,     3,     1,     0,     1,     3,     2,     1,
       2,     1,     0,     2,     1,     3,     1,     1,     1,     1,
       0,     0,     2,     0,     2,     1,     3,     1,     3,     0,
       2,     2,     2,     1,     1,     1,     2,     1,     3,     3,
       4,     3,     4,     0,     1,     6,     0,     1,     1,     1,
       1,     4,     4,     8
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned short yydefact[] =
{
       0,     4,   290,    10,    11,   251,   252,   253,   254,     0,
     255,   256,   215,   212,   216,   213,   214,   269,   270,   271,
     272,   273,   288,   289,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,     0,     0,
       0,     0,     0,     7,     9,     0,     8,     0,     0,   236,
     236,   205,   217,   218,   220,   223,   211,   220,   250,   219,
     268,   266,   267,   265,     0,   263,     0,   264,     0,   262,
       7,    12,    80,   291,   306,   294,   316,   292,   316,   293,
       1,     0,     7,     5,     0,    17,    14,   200,   201,   366,
     260,     0,   234,   239,   202,   237,   238,   190,   337,   340,
     341,   344,   345,   359,   361,     0,   342,   203,   232,   190,
     338,   222,   221,   220,   225,   316,   316,   306,    13,   186,
     187,   184,   185,   182,   183,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    77,   140,     0,    81,
      82,    75,    86,   124,    88,    91,    93,   125,   108,    95,
     100,   102,   104,   106,    98,   121,   153,   122,   123,   126,
     127,   128,   129,    85,   147,   143,   120,   148,   149,   151,
     152,   155,   156,   150,   144,   142,   367,     0,   308,   310,
     311,   313,   368,   369,   315,     0,   305,   317,     0,     6,
      19,   195,   258,   261,   358,     0,     0,     0,    16,   189,
     191,   393,   355,    80,   373,   360,   339,    15,   224,     0,
       0,     0,   136,    98,   134,   135,   137,     0,   139,   138,
       0,   132,     0,   131,    97,    90,     0,     0,     0,   209,
     257,     0,     0,     0,     0,     0,     0,   174,     0,   177,
     176,   175,     0,   179,   178,     0,   170,   169,     0,   173,
     172,   171,     0,   181,   180,     0,     0,   158,   157,     0,
     163,     0,     0,   302,   309,   307,     0,   296,     0,   299,
       0,   194,   196,   259,   343,     0,     0,     0,     0,     0,
     244,   245,   246,   235,   241,   240,     0,     0,   233,   394,
       0,     0,   150,   144,   356,     0,   142,     0,   348,     0,
       0,     0,   379,   297,   300,   303,     0,     0,     0,   146,
     145,     0,    80,   371,   210,   384,   206,   385,   387,     0,
      87,    89,    92,    94,    96,   101,   103,   105,   107,   109,
      78,    79,   141,     0,   165,   166,     0,   164,   160,   159,
     161,   312,   314,   231,   325,   226,   228,   318,     0,     0,
       0,     0,     0,    74,     0,     0,     0,     0,    83,     0,
       0,   195,    62,     0,    21,    39,    23,    37,    38,    40,
      41,    42,    43,    44,    45,    46,    47,    52,    53,    54,
      55,    56,     0,    68,    69,    70,     0,   142,     0,     0,
       0,     0,     0,   248,   242,   192,     0,   236,     0,     0,
       0,   350,   349,   347,   363,   364,   346,   374,   375,   377,
       0,   133,   130,   208,    99,     0,     0,   379,     0,   386,
      80,   371,     0,   154,   162,     0,   322,   324,   335,   230,
     227,   226,   320,     0,   319,    28,    48,    58,    72,    59,
       0,    35,    36,     0,     0,    83,     0,     0,    84,     0,
       0,     0,    18,     0,     0,    50,   197,     0,   116,   117,
       0,     0,     0,     0,     0,     0,     0,   249,     0,   198,
     396,   145,   357,     0,   379,   383,   380,   389,   391,   372,
     388,     0,     0,    76,   167,   168,   327,   393,   329,   330,
       0,   331,   229,   321,     0,     0,    49,     0,     0,    61,
      60,     0,     0,     0,    30,    32,    25,    24,    26,    51,
     119,     0,   113,     0,   114,     0,   115,     0,   243,   247,
     199,   399,     0,   397,   398,     0,     0,   400,     0,   365,
     376,   378,   371,   381,   384,   382,   390,   392,   335,   326,
     332,    73,     0,     0,    83,     0,     0,    27,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   393,     0,     0,
      63,     0,    57,    66,    34,    33,   118,   110,     0,     0,
       0,     0,     0,   395,   328,   334,   333,     0,     0,    83,
       0,     0,   401,   402,     0,    65,    64,     0,   111,     0,
       0,     0,     0,     0,    67,   112,     0,   403
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short yydefgoto[] =
{
      -1,    41,    42,    83,    43,    44,    45,    86,   191,   363,
     364,   453,   507,   365,   503,   504,   548,   440,   441,   442,
     367,   368,   369,   370,   371,   372,   373,   374,   375,   376,
     377,   378,   379,   380,   381,   382,   383,   384,   385,   136,
     137,   138,   139,   447,   386,   141,   142,   143,   144,   145,
     146,   147,   148,   149,   150,   151,   152,   153,   443,   460,
     282,   154,   155,   156,   157,   158,   159,   160,   161,   162,
     163,   164,   165,   166,   167,   168,   169,   170,   338,   171,
     172,   173,   336,   337,   248,   252,   238,   242,   245,   255,
     174,   198,   199,   200,   286,   270,   271,   272,   388,   395,
      46,    47,   396,    49,   397,   314,   307,   308,   228,    51,
      52,    53,    54,    55,   111,    56,   113,   429,   343,   431,
     344,    92,   201,    93,   107,    95,    96,   283,   393,   284,
     468,    57,    58,   193,   194,    59,    60,    75,    61,    62,
      63,    64,    65,    66,    67,    68,    69,   185,   177,   265,
     178,   179,   180,   181,   186,   187,   268,   347,   348,   426,
     427,   428,   538,   487,   488,   489,   558,   576,   490,   108,
     109,    98,    99,   100,   101,   102,   294,   295,   103,   104,
     105,   299,   300,   404,   175,   183,   184,   416,   417,   301,
     302,   407,   408,   409,   410,   476,   418,   317,   318,   288,
     289,   528
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -441
static const short yypact[] =
{
     527,    65,  -441,  -441,  -441,  -441,  -441,  -441,  -441,   -76,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,    55,    56,
      57,    58,   594,  -441,  -441,    50,  -441,   -11,    -5,    32,
      32,  -441,  -441,  -441,  1552,  -441,  -441,  1552,  -441,  -441,
    -441,  -441,  -441,  -441,    -1,  -441,     6,  -441,    12,  -441,
    -441,  -441,  1316,  -441,    75,    30,     3,    36,     3,    85,
    -441,    65,  -441,  -441,    48,  -441,  -441,  -441,  -441,  -441,
     297,    32,    86,  -441,  -441,  -441,    72,   106,  -441,  -441,
     -28,  -441,  -441,  -441,    83,    45,  -441,  -441,  -441,    53,
     173,  -441,  -441,  1552,  -441,     3,     3,    75,  -441,  -441,
    -441,  -441,  -441,  -441,  -441,  1316,  1316,  1316,  1316,  1328,
    1328,    49,  1364,  1316,  1316,   661,  -441,  -441,   131,  -441,
    -441,    80,   114,  -441,    87,   118,   119,  -441,  -441,   121,
     125,    93,    61,   142,   122,  -441,  -441,  -441,  -441,  -441,
    -441,  -441,  -441,   147,  -441,  -441,   136,  -441,  -441,  -441,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,   172,   191,  -441,
    -441,   177,  -441,  -441,  -441,   196,  -441,  -441,   197,  -441,
    -441,  1140,  -441,   297,  -441,   201,   291,    32,  -441,  -441,
    -441,   192,    88,  1316,    67,  -441,  -441,  -441,  -441,   199,
     202,   203,  -441,  -441,  -441,  -441,  -441,   352,  -441,  -441,
     282,  -441,   787,  -441,  -441,  -441,   208,   209,   211,    64,
    -441,  1316,  1316,  1316,  1316,  1316,  1316,  -441,  1316,  -441,
    -441,  -441,  1316,  -441,  -441,  1316,  -441,  -441,  1316,  -441,
    -441,  -441,  1316,  -441,  -441,  1276,  1316,  -441,  -441,  1316,
    1276,    75,    75,  -441,    75,  -441,  1316,  -441,  1598,  -441,
    1152,  -441,  -441,  -441,  -441,   160,   161,   163,   166,   291,
    -441,  -441,  -441,  -441,  -441,  -441,  1552,   167,  -441,  -441,
     352,   136,   -21,   -17,  -441,   -16,    -7,   225,  -441,   228,
     282,   229,  -441,  -441,  -441,  -441,   232,   233,  1552,  -441,
    -441,  1316,  1316,   112,  -441,   109,  -441,  -441,   134,   221,
     114,    87,   118,   119,   121,   125,    93,    61,   142,  -441,
    -441,  -441,  -441,   244,  -441,  -441,   247,   249,  -441,  -441,
    -441,  -441,  -441,  -441,   248,  1598,  -441,  1485,   252,   151,
     253,  1316,   261,  -441,  1060,   214,   215,   341,  1316,   217,
     219,  1140,  -441,   269,  -441,  -441,  -441,  -441,  -441,  -441,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,
    -441,  -441,   254,  -441,  -441,  -441,   270,   256,  1552,   330,
     733,   745,  1228,   275,  -441,  -441,   283,    32,   239,   292,
      88,  -441,  -441,  -441,   289,  -441,  -441,  -441,   290,  -441,
    1552,  -441,  -441,  -441,  -441,   295,   298,  -441,   302,  -441,
    1316,   306,  1316,  -441,  -441,  1276,  -441,   305,    32,  -441,
    -441,  1598,  -441,   293,  -441,  -441,  -441,  -441,   301,  -441,
     326,  -441,  -441,   308,  1316,  1316,   309,   310,  -441,  1316,
    1316,  1240,  -441,   878,  1060,  -441,  -441,   315,  -441,  -441,
     311,   317,   313,   321,   318,   322,   320,   291,   327,  1552,
     153,     4,  -441,   282,   325,    37,  -441,  -441,  -441,  -441,
    -441,   332,   340,  -441,  -441,  -441,  -441,   192,  -441,  -441,
     319,   323,  -441,  -441,  1316,   299,  -441,   342,   343,  -441,
    -441,   344,   349,   347,  -441,  -441,  -441,  -441,  -441,  -441,
    -441,  1276,  -441,  1316,  -441,  1316,  -441,  1316,  -441,  -441,
    -441,  -441,   303,  -441,  -441,   304,   307,  -441,   357,  -441,
    -441,  -441,    31,  -441,    34,  -441,  -441,  -441,    32,  -441,
    -441,  -441,  1316,  1060,  1316,  1060,  1060,  -441,   969,   360,
     361,   358,   359,   429,   282,   282,   365,   192,  1316,   369,
     391,   368,  -441,  -441,  -441,  -441,  -441,  -441,  1316,  1316,
     379,   381,   378,  -441,  -441,  -441,  -441,   380,  1060,  1316,
     385,   382,  -441,  -441,   453,  -441,  -441,   388,  -441,  1316,
     386,  1060,   392,   461,  -441,  -441,   402,  -441
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -441,  -441,  -441,   103,   444,  -441,  -441,  -441,  -441,  -441,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,  -404,  -248,  -441,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -386,
    -191,  -180,  -324,  -401,   -63,  -441,   255,  -441,   257,   258,
     260,  -441,   -95,   262,   250,   259,   246,   263,  -175,    -9,
    -441,  -114,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,
    -441,  -441,  -441,  -178,  -441,  -441,  -441,  -441,   238,  -441,
    -441,  -176,  -441,  -441,  -441,  -441,  -441,  -441,  -441,  -441,
    -174,   394,  -441,  -441,  -441,   143,  -441,  -441,  -441,  -343,
    -441,  -441,    63,  -441,    73,  -441,  -441,  -441,   200,  -128,
    -441,  -441,  -441,  -441,     0,    95,  -441,    76,  -304,  -441,
    -441,  -441,  -441,   312,   463,  -441,  -441,  -441,  -441,  -262,
    -441,  -239,   -35,  -441,  -441,  -222,  -441,   230,  -441,  -441,
    -441,  -441,  -441,  -441,  -441,  -441,  -441,    -4,   393,  -441,
    -441,  -441,   251,  -441,  -441,  -441,  -441,  -441,   170,  -441,
    -441,  -441,  -441,   -25,  -441,  -441,  -441,  -441,  -441,   -45,
    -441,   -48,  -102,  -441,  -441,  -441,   120,  -441,   410,  -441,
    -223,  -441,  -441,  -441,   -49,  -441,    11,    97,  -441,  -441,
    -441,   102,  -441,    47,  -441,  -441,  -221,  -305,  -441,  -440,
    -441,  -441
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -371
static const short yytable[] =
{
     106,   106,   110,   206,    97,   280,   315,   229,   316,   140,
     419,   213,   213,   213,   213,   218,   219,   394,   223,   213,
     213,   281,   366,   297,   291,   182,   292,   438,   293,   345,
     212,   214,   215,   216,    89,    89,   483,    89,   224,   225,
      89,   430,   106,   195,   498,   456,   346,   539,    89,   508,
     509,    84,    89,    72,  -188,   192,   106,   114,    80,    73,
      73,    73,  -352,    48,   330,   332,  -351,   400,   182,   334,
    -362,    87,   227,    50,   188,   342,  -354,    88,    89,   115,
     331,   176,   221,   202,  -304,   335,   116,  -353,   280,  -352,
     315,    89,   117,  -351,   401,   119,   120,   121,   122,   123,
     124,   203,   204,  -354,   281,    48,   345,  -188,   345,  -370,
    -301,   209,   210,   208,  -353,    50,  -295,   574,   213,   213,
     213,   213,   213,   346,   213,   346,   520,   430,   213,   190,
      85,   213,   415,  -188,   213,    74,    76,    78,   213,   560,
     140,   562,   563,   561,   565,   298,    70,    71,   106,   112,
      90,    90,   112,   296,   227,   197,    90,   329,   273,   227,
     312,   532,    91,   312,   532,  -298,   312,   532,   319,   521,
     541,   306,   575,   118,   586,    91,   246,   247,   587,   220,
     229,   231,   232,    90,   196,   189,  -188,   594,  -232,  -232,
    -370,   234,   345,   312,   313,  -232,   333,   213,   458,   462,
     464,   466,    90,   505,  -232,   518,   243,   244,   112,   346,
     230,   131,   182,   182,   459,   182,   414,   233,   290,   253,
     254,   387,   291,   235,   292,   236,   293,   399,   237,   419,
     256,    90,   435,   436,   484,   239,   240,   241,   312,   313,
     481,   312,   313,   522,   523,   524,   525,   526,   527,   140,
     485,   405,   534,   263,   535,  -337,  -337,   257,   258,   249,
     250,   251,  -337,   420,   421,   259,   260,   261,   262,    77,
      79,  -337,   339,   339,   264,   266,   280,   267,   269,   274,
     303,   287,   475,   304,   305,    89,   309,   310,   140,   311,
     389,   390,   281,   391,    89,   448,   392,   398,   119,   120,
     121,   122,   123,   124,   402,   387,   403,   406,   213,   534,
     411,   412,   275,     5,     6,     7,     8,     9,    10,    11,
     458,   422,   550,   423,   551,   424,   552,   276,   277,   278,
    -323,   457,   425,    89,   434,   437,   459,   119,   120,   121,
     122,   123,   124,   439,   444,   445,   446,   449,   106,   450,
     452,   296,   455,   226,   454,    89,   -71,   140,   467,   119,
     120,   121,   122,   123,   124,   469,   276,   277,   278,   470,
     471,   279,   473,   474,   477,   493,   478,   580,   581,   106,
     480,   497,   448,   491,  -370,   494,   501,   502,   486,   495,
     496,   499,   500,   510,   511,   512,   513,   125,   592,   514,
     516,   515,   387,   517,   387,   387,   126,   127,   519,   530,
     128,   536,   129,   130,   131,   132,   133,   134,   537,   540,
     543,   135,   545,  -336,   529,   544,   106,   546,   547,   542,
     533,   140,   206,   553,   554,   556,   125,   555,   566,   567,
     570,   568,   569,   573,   213,   126,   127,   577,   578,   128,
     579,   129,   130,   131,   132,   133,   134,   582,   125,   583,
     135,   584,   585,   588,   590,   589,   591,   126,   127,   593,
     595,   128,   596,   129,   130,   131,   132,   133,   134,   559,
     597,   448,   135,   106,   195,   106,    82,   320,   325,   106,
     321,   327,   322,   491,   387,   323,   387,   387,   324,   387,
     340,   326,   549,   207,   451,   571,   572,   492,   413,   285,
     211,   328,    94,   557,   205,   341,   448,   433,   482,   479,
     472,   531,     0,     0,     0,     0,     0,    -2,     1,   387,
    -204,     0,     0,     2,     0,     0,     0,     0,     0,     0,
       3,     4,   387,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    -3,    81,     0,  -204,     0,     0,
       2,     0,     0,     0,     0,     0,     0,     3,     4,  -204,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,  -204,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,  -204,     0,     0,
       0,     0,   226,     0,    89,     0,     0,     2,   119,   120,
     121,   122,   123,   124,     0,     0,  -204,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,  -204,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,  -204,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   461,     0,    89,     0,     0,     0,
     119,   120,   121,   122,   123,   124,   463,     0,    89,     0,
       0,     0,   119,   120,   121,   122,   123,   124,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   125,     0,     0,
       0,     0,     0,     0,     0,     0,   126,   127,     0,     0,
     128,     0,   129,   130,   131,   132,   133,   134,   226,     0,
      89,   135,     0,  -207,   119,   120,   121,   122,   123,   124,
       0,     0,     0,  -207,  -207,  -207,  -207,  -207,  -207,  -207,
    -207,  -207,  -207,  -207,  -207,  -207,  -207,  -207,  -207,  -207,
    -207,  -207,  -207,  -207,  -207,  -207,  -207,  -207,  -207,  -207,
    -207,  -207,  -207,  -207,  -207,  -207,  -207,  -207,  -207,   125,
       0,     0,     0,     0,     0,     0,     0,     0,   126,   127,
       0,   125,   128,     0,   129,   130,   131,   132,   133,   134,
     126,   127,     0,   135,   128,     0,   129,   130,   131,   132,
     133,   134,     0,     0,     0,   135,     0,     0,     0,   349,
       0,    89,     0,     0,     0,   119,   120,   121,   122,   123,
     124,     0,   506,   125,     0,     0,     0,     0,     0,     0,
       0,     0,   126,   127,     0,     0,   128,     0,   129,   130,
     131,   132,   133,   134,   276,   277,   278,   135,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     350,   351,   352,   353,   354,     0,   355,   356,   357,   358,
     359,   360,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   361,   -22,
     362,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     349,     0,    89,     0,     0,     0,   119,   120,   121,   122,
     123,   124,     0,   564,   125,     0,     0,     0,     0,     0,
       0,     0,     0,   126,   127,     0,     0,   128,     0,   129,
     130,   131,   132,   133,   134,   276,   277,   278,   135,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   350,   351,   352,   353,   354,     0,   355,   356,   357,
     358,   359,   360,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   361,
     -31,   362,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   349,     0,    89,     0,     0,     0,   119,   120,   121,
     122,   123,   124,     0,     0,   125,     0,     0,     0,     0,
       0,     0,     0,     0,   126,   127,     0,     0,   128,     0,
     129,   130,   131,   132,   133,   134,   276,   277,   278,   135,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   350,   351,   352,   353,   354,     0,   355,   356,
     357,   358,   359,   360,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     361,  -193,   362,  -193,     0,     0,     0,  -193,  -193,  -193,
    -193,  -193,  -193,   349,     0,    89,     0,     0,     0,   119,
     120,   121,   122,   123,   124,     0,   125,     0,     0,     0,
       0,     0,     0,     0,     0,   126,   127,     0,     0,   128,
       0,   129,   130,   131,   132,   133,   134,     0,     0,     0,
     135,     0,  -193,  -193,  -193,  -193,  -193,     0,  -193,  -193,
    -193,  -193,  -193,  -193,   350,   351,   352,   353,   354,     0,
     355,   356,   357,   358,   359,   360,     0,     0,     0,     0,
    -193,  -193,  -193,     0,     0,     0,     0,     0,     0,   465,
       0,    89,   361,   -20,   362,   119,   120,   121,   122,   123,
     124,   349,     0,    89,     0,     0,  -193,   119,   120,   121,
     122,   123,   124,     0,     0,  -193,  -193,     0,   125,  -193,
       0,  -193,  -193,  -193,  -193,  -193,  -193,   126,   127,     0,
    -193,   128,     0,   129,   130,   131,   132,   133,   134,    89,
       0,     0,   135,   119,   120,   121,   122,   123,   124,     0,
       0,     0,   350,   351,   352,   353,   354,     0,   355,   356,
     357,   358,   359,   360,     0,     0,     0,     0,     0,     0,
       0,     0,   276,   277,   278,     0,     0,     0,     0,    89,
     361,   -29,   362,   119,   120,   121,   122,   123,   124,     0,
       0,    89,     0,     0,   125,   119,   120,   121,   122,   123,
     124,     0,     0,   126,   127,     0,   125,   128,     0,   129,
     130,   131,   132,   133,   134,   126,   127,     0,   135,   128,
       0,   129,   130,   131,   132,   133,   134,    89,     0,     0,
     135,   119,   120,   121,   122,   123,   124,     0,     0,     0,
       0,     0,   125,     0,     0,     0,     0,     0,     0,     0,
       0,   126,   127,     0,     0,   128,     0,   129,   130,   131,
     132,   133,   134,     0,     0,     0,   135,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   125,     0,     0,     0,     0,     0,     0,     0,
       0,   126,   127,     0,   125,   128,     0,   129,   130,   131,
     132,   133,   134,   126,   127,     0,   135,   128,     0,   129,
     130,   131,   132,   133,   134,     0,     0,     0,   217,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     125,     0,     0,     0,     0,     0,     0,     0,     0,   126,
     127,     0,     0,   128,     0,   129,   130,   131,   132,   133,
     134,     2,     0,     0,   222,     0,     0,     0,     0,     0,
       0,     5,     6,     7,     8,     9,    10,    11,     0,     0,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     2,     0,
       0,     0,     0,     0,     0,     0,     0,   432,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,     2,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     5,     6,     7,     8,     9,    10,
      11,     0,     0,     0,     0,     0,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40
};

static const short yycheck[] =
{
      49,    50,    50,   105,    49,   196,   229,   135,   229,    72,
     315,   125,   126,   127,   128,   129,   130,   279,   132,   133,
     134,   196,   270,   203,   202,    74,   202,   351,   202,   268,
     125,   126,   127,   128,     3,     3,   422,     3,   133,   134,
       3,   345,    91,    91,   445,   388,   268,   487,     3,   453,
     454,     1,     3,   129,     1,    90,   105,    57,     0,     4,
       4,     4,    83,     0,   255,   256,    83,    83,   117,   260,
       3,    82,   135,     0,    78,   266,    83,    82,     3,    80,
     255,     6,   131,   111,    81,   260,    80,    83,   279,   110,
     313,     3,    80,   110,   110,     7,     8,     9,    10,    11,
      12,   129,   130,   110,   279,    42,   345,     1,   347,    78,
      80,   115,   116,   113,   110,    42,    80,   557,   232,   233,
     234,   235,   236,   345,   238,   347,   469,   431,   242,    81,
      80,   245,   312,    80,   248,    80,    80,    80,   252,   543,
     203,   545,   546,   544,   548,    78,    81,    82,   197,    54,
     119,   119,    57,   202,   217,    83,   119,   252,   193,   222,
     129,   130,   130,   129,   130,    80,   129,   130,   231,    16,
     494,   220,   558,    70,   578,   130,   115,   116,   579,   130,
     308,   101,   102,   119,    98,    82,    80,   591,    82,    83,
      78,   104,   431,   129,   130,    89,   259,   311,   389,   390,
     391,   392,   119,   451,    98,   467,   113,   114,   113,   431,
      79,   123,   261,   262,   389,   264,   311,   103,   130,    97,
      98,   270,   400,   105,   400,   106,   400,   290,   107,   534,
      83,   119,    81,    82,   425,   110,   111,   112,   129,   130,
     420,   129,   130,    90,    91,    92,    93,    94,    95,   312,
     425,   300,   475,    81,   475,    82,    83,   121,   122,   117,
     118,   119,    89,   129,   130,   129,   130,   131,   132,    39,
      40,    98,   261,   262,    83,    98,   467,    81,    81,    78,
      81,    89,   410,    81,    81,     3,    78,    78,   351,    78,
     130,   130,   467,   130,     3,   358,   130,   130,     7,     8,
       9,    10,    11,    12,    79,   354,    78,    78,   422,   532,
      78,    78,    21,    16,    17,    18,    19,    20,    21,    22,
     511,   100,   513,    79,   515,    78,   517,    36,    37,    38,
      82,     1,    83,     3,    82,    82,   511,     7,     8,     9,
      10,    11,    12,    82,   130,   130,     5,   130,   397,   130,
      81,   400,    82,     1,   100,     3,   100,   420,    83,     7,
       8,     9,    10,    11,    12,    82,    36,    37,    38,   130,
      78,    80,    83,    83,    79,    82,    78,   568,   569,   428,
      78,   444,   445,   428,    78,    84,   449,   450,    83,    63,
      82,    82,    82,    78,    83,    78,    83,   106,   589,    78,
      78,    83,   451,    83,   453,   454,   115,   116,    81,    84,
     119,    79,   121,   122,   123,   124,   125,   126,    78,   100,
      78,   130,    78,   100,   473,    82,   475,    78,    81,   130,
     475,   494,   534,   130,   130,    78,   106,   130,    78,    78,
      11,    83,    83,    78,   558,   115,   116,    78,    57,   119,
      82,   121,   122,   123,   124,   125,   126,    78,   106,    78,
     130,    83,    82,    78,    11,    83,    78,   115,   116,    83,
      78,   119,    11,   121,   122,   123,   124,   125,   126,   542,
      78,   544,   130,   532,   532,   534,    42,   232,   238,   538,
     233,   245,   234,   538,   543,   235,   545,   546,   236,   548,
     262,   242,   511,   109,   361,   554,   555,   431,   308,   197,
     117,   248,    49,   538,   104,   264,   579,   347,   421,   417,
     400,   474,    -1,    -1,    -1,    -1,    -1,     0,     1,   578,
       3,    -1,    -1,     6,    -1,    -1,    -1,    -1,    -1,    -1,
      13,    14,   591,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     0,     1,    -1,     3,    -1,    -1,
       6,    -1,    -1,    -1,    -1,    -1,    -1,    13,    14,    82,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,   119,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   130,    -1,    -1,
      -1,    -1,     1,    -1,     3,    -1,    -1,     6,     7,     8,
       9,    10,    11,    12,    -1,    -1,    82,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,   119,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   130,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,     3,    -1,    -1,    -1,
       7,     8,     9,    10,    11,    12,     1,    -1,     3,    -1,
      -1,    -1,     7,     8,     9,    10,    11,    12,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   106,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   115,   116,    -1,    -1,
     119,    -1,   121,   122,   123,   124,   125,   126,     1,    -1,
       3,   130,    -1,     6,     7,     8,     9,    10,    11,    12,
      -1,    -1,    -1,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,   106,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   115,   116,
      -1,   106,   119,    -1,   121,   122,   123,   124,   125,   126,
     115,   116,    -1,   130,   119,    -1,   121,   122,   123,   124,
     125,   126,    -1,    -1,    -1,   130,    -1,    -1,    -1,     1,
      -1,     3,    -1,    -1,    -1,     7,     8,     9,    10,    11,
      12,    -1,    14,   106,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   115,   116,    -1,    -1,   119,    -1,   121,   122,
     123,   124,   125,   126,    36,    37,    38,   130,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      52,    53,    54,    55,    56,    -1,    58,    59,    60,    61,
      62,    63,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    81,
      82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,     3,    -1,    -1,    -1,     7,     8,     9,    10,
      11,    12,    -1,    14,   106,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   115,   116,    -1,    -1,   119,    -1,   121,
     122,   123,   124,   125,   126,    36,    37,    38,   130,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    52,    53,    54,    55,    56,    -1,    58,    59,    60,
      61,    62,    63,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,
      81,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,     3,    -1,    -1,    -1,     7,     8,     9,
      10,    11,    12,    -1,    -1,   106,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   115,   116,    -1,    -1,   119,    -1,
     121,   122,   123,   124,   125,   126,    36,    37,    38,   130,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    52,    53,    54,    55,    56,    -1,    58,    59,
      60,    61,    62,    63,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      80,     1,    82,     3,    -1,    -1,    -1,     7,     8,     9,
      10,    11,    12,     1,    -1,     3,    -1,    -1,    -1,     7,
       8,     9,    10,    11,    12,    -1,   106,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   115,   116,    -1,    -1,   119,
      -1,   121,   122,   123,   124,   125,   126,    -1,    -1,    -1,
     130,    -1,    52,    53,    54,    55,    56,    -1,    58,    59,
      60,    61,    62,    63,    52,    53,    54,    55,    56,    -1,
      58,    59,    60,    61,    62,    63,    -1,    -1,    -1,    -1,
      80,    81,    82,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,     3,    80,    81,    82,     7,     8,     9,    10,    11,
      12,     1,    -1,     3,    -1,    -1,   106,     7,     8,     9,
      10,    11,    12,    -1,    -1,   115,   116,    -1,   106,   119,
      -1,   121,   122,   123,   124,   125,   126,   115,   116,    -1,
     130,   119,    -1,   121,   122,   123,   124,   125,   126,     3,
      -1,    -1,   130,     7,     8,     9,    10,    11,    12,    -1,
      -1,    -1,    52,    53,    54,    55,    56,    -1,    58,    59,
      60,    61,    62,    63,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    36,    37,    38,    -1,    -1,    -1,    -1,     3,
      80,    81,    82,     7,     8,     9,    10,    11,    12,    -1,
      -1,     3,    -1,    -1,   106,     7,     8,     9,    10,    11,
      12,    -1,    -1,   115,   116,    -1,   106,   119,    -1,   121,
     122,   123,   124,   125,   126,   115,   116,    -1,   130,   119,
      -1,   121,   122,   123,   124,   125,   126,     3,    -1,    -1,
     130,     7,     8,     9,    10,    11,    12,    -1,    -1,    -1,
      -1,    -1,   106,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   115,   116,    -1,    -1,   119,    -1,   121,   122,   123,
     124,   125,   126,    -1,    -1,    -1,   130,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   106,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   115,   116,    -1,   106,   119,    -1,   121,   122,   123,
     124,   125,   126,   115,   116,    -1,   130,   119,    -1,   121,
     122,   123,   124,   125,   126,    -1,    -1,    -1,   130,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     106,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   115,
     116,    -1,    -1,   119,    -1,   121,   122,   123,   124,   125,
     126,     6,    -1,    -1,   130,    -1,    -1,    -1,    -1,    -1,
      -1,    16,    17,    18,    19,    20,    21,    22,    -1,    -1,
      -1,    -1,    -1,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     6,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,     6,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    16,    17,    18,    19,    20,    21,
      22,    -1,    -1,    -1,    -1,    -1,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned short yystos[] =
{
       0,     1,     6,    13,    14,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,   134,   135,   137,   138,   139,   233,   234,   235,   236,
     237,   242,   243,   244,   245,   246,   248,   264,   265,   268,
     269,   271,   272,   273,   274,   275,   276,   277,   278,   279,
      81,    82,   129,     4,    80,   270,    80,   270,    80,   270,
       0,     1,   137,   136,     1,    80,   140,    82,    82,     3,
     119,   130,   254,   256,   257,   258,   259,   302,   304,   305,
     306,   307,   308,   311,   312,   313,   317,   257,   302,   303,
     304,   247,   248,   249,   247,    80,    80,    80,   136,     7,
       8,     9,    10,    11,    12,   106,   115,   116,   119,   121,
     122,   123,   124,   125,   126,   130,   172,   173,   174,   175,
     177,   178,   179,   180,   181,   182,   183,   184,   185,   186,
     187,   188,   189,   190,   194,   195,   196,   197,   198,   199,
     200,   201,   202,   203,   204,   205,   206,   207,   208,   209,
     210,   212,   213,   214,   223,   317,     6,   281,   283,   284,
     285,   286,   317,   318,   319,   280,   287,   288,   280,   136,
      81,   141,   265,   266,   267,   304,    98,    83,   224,   225,
     226,   255,   111,   129,   130,   311,   305,   224,   247,   280,
     280,   281,   185,   194,   185,   185,   185,   130,   194,   194,
     130,   317,   130,   194,   185,   185,     1,   177,   241,   242,
      79,   101,   102,   103,   104,   105,   106,   107,   219,   110,
     111,   112,   220,   113,   114,   221,   115,   116,   217,   117,
     118,   119,   218,    97,    98,   222,    83,   121,   122,   129,
     130,   131,   132,    81,    83,   282,    98,    81,   289,    81,
     228,   229,   230,   265,    78,    21,    36,    37,    38,    80,
     173,   191,   193,   260,   262,   256,   227,    89,   332,   333,
     130,   206,   214,   223,   309,   310,   317,   174,    78,   314,
     315,   322,   323,    81,    81,    81,   317,   239,   240,    78,
      78,    78,   129,   130,   238,   313,   329,   330,   331,   177,
     179,   181,   182,   183,   186,   187,   188,   189,   190,   185,
     173,   191,   173,   177,   173,   191,   215,   216,   211,   319,
     211,   285,   173,   251,   253,   264,   268,   290,   291,     1,
      52,    53,    54,    55,    56,    58,    59,    60,    61,    62,
      63,    80,    82,   142,   143,   146,   151,   153,   154,   155,
     156,   157,   158,   159,   160,   161,   162,   163,   164,   165,
     166,   167,   168,   169,   170,   171,   177,   317,   231,   130,
     130,   130,   130,   261,   262,   232,   235,   237,   130,   177,
      83,   110,    79,    78,   316,   317,    78,   324,   325,   326,
     327,    78,    78,   241,   185,   174,   320,   321,   329,   330,
     129,   130,   100,    79,    78,    83,   292,   293,   294,   250,
     251,   252,    82,   291,    82,    81,    82,    82,   175,    82,
     150,   151,   152,   191,   130,   130,     5,   176,   177,   130,
     130,   228,    81,   144,   100,    82,   232,     1,   173,   191,
     192,     1,   173,     1,   173,     1,   173,    83,   263,    82,
     130,    78,   309,    83,    83,   242,   328,    79,    78,   324,
      78,   174,   320,   172,   173,   191,    83,   296,   297,   298,
     301,   302,   250,    82,    84,    63,    82,   177,   176,    82,
      82,   177,   177,   147,   148,   151,    14,   145,   150,   150,
      78,    83,    78,    83,    78,    83,    78,    83,   262,    81,
     232,    16,    90,    91,    92,    93,    94,    95,   334,   317,
      84,   326,   130,   302,   313,   329,    79,    78,   295,   332,
     100,   175,   130,    78,    82,    78,    78,    81,   149,   192,
     173,   173,   173,   130,   130,   130,    78,   296,   299,   177,
     150,   176,   150,   150,    14,   150,    78,    78,    83,    83,
      11,   317,   317,    78,   332,   172,   300,    78,    57,    82,
     173,   173,    78,    78,    83,    82,   150,   176,    78,    83,
      11,    78,   173,    83,   150,    78,    11,    78
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
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
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
  /* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;

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
        case 2:
#line 244 "gram.y"
    {
            if (err_cnt == 0)
              *gProject->Parse_TOS->yyerrstream
              << "Warning: ANSI/ISO C forbids an empty source file.\n";
            gProject->Parse_TOS->transUnit = (TransUnit*) NULL;
            yyval.transunit = (TransUnit*) NULL;
        ;}
    break;

  case 3:
#line 252 "gram.y"
    {
            if (err_cnt)
            {
                *gProject->Parse_TOS->yyerrstream
                << err_cnt << " errors found.\n";
                gProject->Parse_TOS->transUnit = (TransUnit*) NULL;
            } else {
                gProject->Parse_TOS->transUnit = yyval.transunit;
            }
        ;}
    break;

  case 4:
#line 263 "gram.y"
    {
            *gProject->Parse_TOS->yyerrstream << "Errors - Aborting parse.\n";
            gProject->Parse_TOS->transUnit = (TransUnit*) NULL;
            YYACCEPT;
        ;}
    break;

  case 5:
#line 271 "gram.y"
    {
            yyval.transunit = gProject->Parse_TOS->transUnit;
            yyval.transunit->add(yyvsp[-1].stemnt);
        ;}
    break;

  case 6:
#line 276 "gram.y"
    {
            yyval.transunit->add(yyvsp[-1].stemnt);
        ;}
    break;

  case 7:
#line 282 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->ReinitializeCtxt();
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScopes(FILE_SCOPE);
            err_top_level = 0;            
        ;}
    break;

  case 8:
#line 291 "gram.y"
    {
            yyval.stemnt = yyvsp[0].declStemnt;
        ;}
    break;

  case 9:
#line 295 "gram.y"
    {
            yyval.stemnt = yyvsp[0].functionDef;
        ;}
    break;

  case 10:
#line 299 "gram.y"
    {
            yyval.stemnt = yyvsp[0].stemnt;
        ;}
    break;

  case 11:
#line 303 "gram.y"
    {
            yyval.stemnt = yyvsp[0].stemnt;
        ;}
    break;

  case 12:
#line 307 "gram.y"
    {
            yyval.stemnt = (Statement*) NULL;
        ;}
    break;

  case 13:
#line 311 "gram.y"
    {
            yyval.stemnt = (Statement*) NULL;
        ;}
    break;

  case 14:
#line 317 "gram.y"
    {
            if (yyvsp[0].stemnt != NULL)
            {
                yyval.functionDef = new FunctionDef(yyvsp[0].stemnt->location);
                Block *blk = (Block*) yyvsp[0].stemnt;
    
                yyval.functionDef->decl = yyvsp[-1].decl;
                
                if (yyvsp[-1].decl->name &&
                    yyvsp[-1].decl->name->entry)
                    yyvsp[-1].decl->name->entry->u2FunctionDef = yyval.functionDef;
                
                // Steal internals of the compound statement
                yyval.functionDef->head = blk->head;
                yyval.functionDef->tail = blk->tail;
    
                blk->head = blk->tail = (Statement*) NULL;
                delete yyvsp[0].stemnt;    
            }
			else
			{
				delete yyvsp[-1].decl;
				yyval.functionDef = (FunctionDef*) NULL;
			}
        ;}
    break;

  case 15:
#line 345 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
            
            possibleType = true;
            yyval.decl = yyvsp[-1].decl;

            if (yyval.decl->form != NULL)
            {
                assert(err_top_level ||
                       yyval.decl->form->type == TT_Function );
    
                yyval.decl->extend(yyvsp[-2].base);
    
                /* This is adding K&R-style declarations if $3 exists */
                if (yyvsp[0].decl != NULL)
                {
                    FunctionType *fnc = (FunctionType*) (yyval.decl->form);
                    fnc->KnR_decl = true;
                    Decl *param = yyvsp[0].decl;
                    while (param != NULL)
                    {
                        Decl *next= param->next;
                            delete param ;
                        param = next;
                    }
                }
            }
        ;}
    break;

  case 16:
#line 374 "gram.y"
    {

            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
            
            yyval.decl = yyvsp[-1].decl;

            if (yyval.decl->form != NULL)
            {
                assert(err_top_level ||
                       yyval.decl->form->type == TT_Function );
                yyval.decl->extend(yyvsp[-2].base);
    
                /* This is adding K&R-style declarations if $3 exists */
                if (yyvsp[0].decl != NULL)
                {
                    FunctionType *fnc = (FunctionType*) (yyval.decl->form);
                    fnc->KnR_decl = true;
                    Decl *param = yyvsp[0].decl;
                    while (param != NULL)
                    {
                        Decl *next= param->next;
                            delete param ;
                        param = next;
                    }
                }
            }
        ;}
    break;

  case 17:
#line 407 "gram.y"
    {  
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ReEnterScope();
        ;}
    break;

  case 18:
#line 412 "gram.y"
    {
            Block*    block = new Block(*yyvsp[-4].loc);
            yyval.stemnt = block;
            block->addDecls(yyvsp[-2].decl);
            block->addStatements(ReverseList(yyvsp[-1].stemnt));
            if (gProject->Parse_TOS->transUnit)
            {
                yyCheckLabelsDefinition(gProject->Parse_TOS->transUnit->contxt.labels);
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
            }
        ;}
    break;

  case 19:
#line 425 "gram.y"
    {
            yyval.stemnt = (Statement*) NULL;
        ;}
    break;

  case 20:
#line 431 "gram.y"
    {
            yyval.stemnt = (Statement*) NULL;
        ;}
    break;

  case 22:
#line 438 "gram.y"
    {
	    /*
	     * All the statements are expected in a reversed list (because
	     * of how we parse stemnt_list2) so we need to take the
	     * non_constructor statement at the end.
	     */
            if (yyvsp[0].stemnt)
            {
	        Statement *s;

		for (s = yyvsp[0].stemnt; s->next; s = s->next) /* Traverse to the end */;
		s->next = yyvsp[-1].stemnt;
                yyval.stemnt = yyvsp[0].stemnt;
            }
        ;}
    break;

  case 23:
#line 456 "gram.y"
    {
	   yyval.stemnt = (Statement *) NULL;
	;}
    break;

  case 24:
#line 460 "gram.y"
    {
            /* Hook them up backwards, we'll reverse them later. */
            if (yyvsp[0].stemnt)
            {
                yyvsp[0].stemnt->next = yyvsp[-1].stemnt;
                yyval.stemnt = yyvsp[0].stemnt;
            }
        ;}
    break;

  case 25:
#line 469 "gram.y"
    {    /* preprocessor #line directive */
            /* Hook them up backwards, we'll reverse them later. */
            if (yyvsp[0].stemnt)
            {
                yyvsp[0].stemnt->next = yyvsp[-1].stemnt;
                yyval.stemnt = yyvsp[0].stemnt;
            }
        ;}
    break;

  case 27:
#line 488 "gram.y"
    {
            Block*    block = new Block(*yyvsp[-3].loc);
            yyval.stemnt = block;
            block->addDecls(yyvsp[-2].decl);
            block->addStatements(ReverseList(yyvsp[-1].stemnt));
            
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
        ;}
    break;

  case 28:
#line 498 "gram.y"
    {
            yyval.stemnt = (Statement*) NULL;
        ;}
    break;

  case 29:
#line 504 "gram.y"
    {
            yyval.stemnt = (Statement*) NULL;
        ;}
    break;

  case 31:
#line 511 "gram.y"
    {
	    /*
	     * All the statements are expected in a reversed list (because
	     * of how we parse stemnt_list_reentrance2) so we need to take
	     * the non_constructor statement at the end.
	     */
            if (yyvsp[0].stemnt)
            {
	        Statement *s;

		for (s = yyvsp[0].stemnt; s->next; s = s->next) /* Traverse to the end */;
		s->next = yyvsp[-1].stemnt;
                yyval.stemnt = yyvsp[0].stemnt;
            }
        ;}
    break;

  case 32:
#line 529 "gram.y"
    {
	   yyval.stemnt = (Statement *) NULL;
	;}
    break;

  case 33:
#line 533 "gram.y"
    {
            /* Hook them up backwards, we'll reverse them later. */
            if (yyvsp[0].stemnt)
            {
                yyvsp[0].stemnt->next = yyvsp[-1].stemnt;
                yyval.stemnt = yyvsp[0].stemnt;
            }
        ;}
    break;

  case 34:
#line 542 "gram.y"
    {    /* preprocessor #line directive */
            /* Hook them up backwards, we'll reverse them later. */
            if (yyvsp[0].stemnt)
            {
                yyvsp[0].stemnt->next = yyvsp[-1].stemnt;
                yyval.stemnt = yyvsp[0].stemnt;
            }
        ;}
    break;

  case 35:
#line 553 "gram.y"
    {
	    yyval.stemnt = yyvsp[0].stemnt;
	 ;}
    break;

  case 36:
#line 557 "gram.y"
    {
	    yyval.stemnt = yyvsp[0].stemnt;
	 ;}
    break;

  case 48:
#line 574 "gram.y"
    {
            delete yyvsp[0].loc;
            yyval.stemnt = (Statement*) NULL;
        ;}
    break;

  case 49:
#line 597 "gram.y"
    {
            yyval.stemnt = new ExpressionStemnt(yyvsp[-1].value,*yyvsp[0].loc);
            delete yyvsp[0].loc;
	;}
    break;

  case 50:
#line 604 "gram.y"
    {
            yyval.stemnt = new ExpressionStemnt(yyvsp[-1].value,*yyvsp[0].loc);
            delete yyvsp[0].loc;
        ;}
    break;

  case 51:
#line 611 "gram.y"
    {
            yyval.stemnt = yyvsp[0].stemnt;
            if (yyval.stemnt == NULL)
            {
              /* Sorry, we must have a statement here. */
              yyerr("Can't have a label at the end of a block! ");
              yyval.stemnt = new Statement(ST_NullStemnt,*yyvsp[-1].loc);
            }
            yyval.stemnt->addHeadLabel(yyvsp[-2].label);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 57:
#line 634 "gram.y"
    {
            yyval.stemnt = new SwitchStemnt(yyvsp[-2].value,yyvsp[0].stemnt,*yyvsp[-4].loc);
            delete yyvsp[-4].loc;
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 58:
#line 643 "gram.y"
    {
            yyval.stemnt = new Statement(ST_BreakStemnt,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 59:
#line 651 "gram.y"
    {
            yyval.stemnt = new Statement(ST_ContinueStemnt,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 60:
#line 659 "gram.y"
    {
            yyval.stemnt = new ReturnStemnt(yyvsp[-1].value,*yyvsp[-2].loc);
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 61:
#line 667 "gram.y"
    {
            yyval.stemnt = new GotoStemnt(yyvsp[-1].symbol,*yyvsp[-2].loc);
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 62:
#line 675 "gram.y"
    {
            yyval.stemnt = new Statement(ST_NullStemnt,*yyvsp[0].loc);
            delete yyvsp[0].loc;
        ;}
    break;

  case 63:
#line 682 "gram.y"
    {
            yyval.stemnt = new IfStemnt(yyvsp[-2].value,yyvsp[0].stemnt,*yyvsp[-4].loc);
            delete yyvsp[-4].loc;
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 64:
#line 691 "gram.y"
    {
            yyval.stemnt = new IfStemnt(yyvsp[-4].value,yyvsp[-2].stemnt,*yyvsp[-6].loc,yyvsp[0].stemnt);
            delete yyvsp[-6].loc;
            delete yyvsp[-5].loc;
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 65:
#line 701 "gram.y"
    {
            yyval.stemnt = new DoWhileStemnt(yyvsp[-2].value,yyvsp[-5].stemnt,*yyvsp[-6].loc);
            delete yyvsp[-6].loc;
            delete yyvsp[-4].loc;
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 66:
#line 712 "gram.y"
    {
            yyval.stemnt = new WhileStemnt(yyvsp[-2].value,yyvsp[0].stemnt,*yyvsp[-4].loc);
            delete yyvsp[-4].loc;
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 67:
#line 722 "gram.y"
    {
            yyval.stemnt = new ForStemnt(yyvsp[-6].value,yyvsp[-4].value,yyvsp[-2].value,*yyvsp[-8].loc,yyvsp[0].stemnt);
            delete yyvsp[-8].loc;
            delete yyvsp[-7].loc;
            delete yyvsp[-5].loc;
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 71:
#line 738 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                yyval.label = gProject->Parse_TOS->parseCtxt->Mk_named_label(yyvsp[0].symbol,
                                gProject->Parse_TOS->transUnit->contxt.labels);
        ;}
    break;

  case 72:
#line 746 "gram.y"
    {
            yyval.label = new Label(yyvsp[0].value);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 73:
#line 751 "gram.y"
    {
            yyval.label = new Label(yyvsp[-2].value,yyvsp[0].value);
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 74:
#line 759 "gram.y"
    {
            yyval.label = new Label(LT_Default);
            delete yyvsp[0].loc;
        ;}
    break;

  case 76:
#line 776 "gram.y"
    {
            yyval.value = new TrinaryExpr(yyvsp[-4].value,yyvsp[-2].value,yyvsp[0].value,*yyvsp[-3].loc);
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 78:
#line 785 "gram.y"
    {
            yyval.value = new AssignExpr(yyvsp[-1].assignOp,yyvsp[-2].value,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 79:
#line 789 "gram.y"
    {
            yyval.value = new AssignExpr(yyvsp[-1].assignOp,yyvsp[-2].value,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 80:
#line 795 "gram.y"
    {
            yyval.value = (Expression*) NULL;
        ;}
    break;

  case 83:
#line 805 "gram.y"
    {
           yyval.value = (Expression*) NULL;
        ;}
    break;

  case 87:
#line 816 "gram.y"
    {
            yyval.value = new BinaryExpr(BO_Or,yyvsp[-2].value,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 89:
#line 824 "gram.y"
    {
            yyval.value = new BinaryExpr(BO_And,yyvsp[-2].value,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 90:
#line 831 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_Not,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 92:
#line 839 "gram.y"
    {
            yyval.value = new BinaryExpr(BO_BitOr,yyvsp[-2].value,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 94:
#line 847 "gram.y"
    {
            yyval.value = new BinaryExpr(BO_BitXor,yyvsp[-2].value,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 96:
#line 855 "gram.y"
    {
            yyval.value = new BinaryExpr(BO_BitAnd,yyvsp[-2].value,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 97:
#line 862 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_BitNot,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 99:
#line 870 "gram.y"
    {
            yyval.value = new CastExpr(yyvsp[-2].type,yyvsp[0].value,*yyvsp[-3].loc);
            delete yyvsp[-3].loc;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 101:
#line 879 "gram.y"
    {
            yyval.value = new RelExpr(yyvsp[-1].relOp,yyvsp[-2].value,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 103:
#line 886 "gram.y"
    {
            yyval.value = new RelExpr(yyvsp[-1].relOp,yyvsp[-2].value,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 105:
#line 893 "gram.y"
    {
            yyval.value = new BinaryExpr(yyvsp[-1].binOp,yyvsp[-2].value,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 107:
#line 900 "gram.y"
    {
            yyval.value = new BinaryExpr(yyvsp[-1].binOp,yyvsp[-2].value,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 109:
#line 907 "gram.y"
    {
            yyval.value = new BinaryExpr(yyvsp[-1].binOp,yyvsp[-2].value,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 110:
#line 913 "gram.y"
    {
	    Expression *exprs[] = { yyvsp[-3].value, yyvsp[-1].value };
            yyval.value = new ConstructorExpr(yyvsp[-5].base, exprs, NoLocation);
            delete yyvsp[-4].loc;
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 111:
#line 922 "gram.y"
    {
	    Expression *exprs[] = { yyvsp[-5].value, yyvsp[-3].value, yyvsp[-1].value };
            yyval.value = new ConstructorExpr(yyvsp[-7].base, exprs, NoLocation);
            delete yyvsp[-6].loc;
            delete yyvsp[-4].loc;
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 112:
#line 932 "gram.y"
    {
	    Expression *exprs[] = { yyvsp[-7].value, yyvsp[-5].value, yyvsp[-3].value, yyvsp[-1].value };
            yyval.value = new ConstructorExpr(yyvsp[-9].base, exprs, NoLocation);
            delete yyvsp[-8].loc;
            delete yyvsp[-6].loc;
            delete yyvsp[-4].loc;
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 113:
#line 942 "gram.y"
    {
	   yyval.value = (Expression *) NULL;
	;}
    break;

  case 114:
#line 946 "gram.y"
    {
	   yyval.value = (Expression *) NULL;
	;}
    break;

  case 115:
#line 950 "gram.y"
    {
	   yyval.value = (Expression *) NULL;
	;}
    break;

  case 118:
#line 961 "gram.y"
    {
	   Symbol *sym = new Symbol();
	   Variable *var;

	   sym->name = strdup("iter");
	   var = new Variable(sym, *yyvsp[-4].loc);
	   yyval.value = new FunctionCall(var, *yyvsp[-4].loc);

	   ((FunctionCall *) yyval.value)->addArg(yyvsp[-3].value);
	   ((FunctionCall *) yyval.value)->addArg(yyvsp[-1].value);

           delete yyvsp[-4].loc;
           delete yyvsp[-2].loc;
           delete yyvsp[0].loc;
	;}
    break;

  case 119:
#line 977 "gram.y"
    {
	   yyval.value = (Expression *) NULL;
	;}
    break;

  case 130:
#line 995 "gram.y"
    {
            yyval.value = new SizeofExpr(yyvsp[-1].type,*yyvsp[-3].loc);
            delete yyvsp[-3].loc;
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 131:
#line 1002 "gram.y"
    {
            yyval.value = new SizeofExpr(yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 132:
#line 1009 "gram.y"
    {
	  yyval.value = new BrtIndexofExpr(new Variable(yyvsp[0].symbol,*yyvsp[-1].loc),*yyvsp[-1].loc);
	;}
    break;

  case 133:
#line 1013 "gram.y"
    {
	  yyval.value = new BrtIndexofExpr(new Variable(yyvsp[-1].symbol,*yyvsp[-3].loc),*yyvsp[-3].loc);
	;}
    break;

  case 134:
#line 1033 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_Minus,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 135:
#line 1039 "gram.y"
    {
            /* Unary plus is an ISO addition (for symmetry) - ignore it */
            yyval.value = yyvsp[0].value;
        ;}
    break;

  case 136:
#line 1046 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_AddrOf,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 137:
#line 1053 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_Deref,yyvsp[0].value,NoLocation);
        ;}
    break;

  case 138:
#line 1059 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_PreInc,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 139:
#line 1066 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_PreDec,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 141:
#line 1074 "gram.y"
    {
            yyval.value = new BinaryExpr(BO_Comma,yyvsp[-2].value,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 142:
#line 1081 "gram.y"
    {
            yyval.value = new Variable(yyvsp[0].symbol,NoLocation);
        ;}
    break;

  case 144:
#line 1086 "gram.y"
    {
            yyval.value = yyvsp[0].consValue;
        ;}
    break;

  case 145:
#line 1092 "gram.y"
    {
            yyval.value = yyvsp[-1].value;
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 146:
#line 1098 "gram.y"
    {
            yyval.value = (Expression*) NULL;
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 154:
#line 1116 "gram.y"
    {
            yyval.value = new IndexExpr(yyvsp[-3].value,yyvsp[-1].value,*yyvsp[-2].loc);
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 157:
#line 1128 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_PostInc,yyvsp[-1].value,*yyvsp[0].loc);
            delete yyvsp[0].loc;
        ;}
    break;

  case 158:
#line 1135 "gram.y"
    {
            yyval.value = new UnaryExpr(UO_PostDec,yyvsp[-1].value,*yyvsp[0].loc);
            delete yyvsp[0].loc;
        ;}
    break;

  case 160:
#line 1146 "gram.y"
    {
            Variable *var = new Variable(yyvsp[0].symbol,*yyvsp[-1].loc);
            BinaryExpr *be = new BinaryExpr(BO_Member,yyvsp[-2].value,var,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
            yyval.value = be;

            // Lookup the component in its struct
            // if possible.
            if (yyvsp[-2].value->etype == ET_Variable)
            {
                Variable  *var = (Variable*) yyvsp[-2].value;
                Symbol    *varName = var->name;
                SymEntry  *entry = varName->entry;

                if (entry && entry->uVarDecl)
                {
                    entry->uVarDecl->lookup(yyvsp[0].symbol);
                }
            }
        ;}
    break;

  case 161:
#line 1169 "gram.y"
    {
            Variable *var = new Variable(yyvsp[0].symbol,*yyvsp[-1].loc);
            BinaryExpr *be = new BinaryExpr(BO_PtrMember,yyvsp[-2].value,var,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
            yyval.value = be;

            // Lookup the component in its struct
            // if possible.
            if (yyvsp[-2].value->etype == ET_Variable)
            {
                Variable  *var = (Variable*) yyvsp[-2].value;
                Symbol    *varName = var->name;
                SymEntry  *entry = varName->entry;

                if (entry && entry->uVarDecl)
                {
                    entry->uVarDecl->lookup(yyvsp[0].symbol);
                }
            }
        ;}
    break;

  case 162:
#line 1192 "gram.y"
    {
            FunctionCall* fc = new FunctionCall(yyvsp[-3].value,*yyvsp[-2].loc);

            /* add function args */
            fc->addArgs(ReverseList(yyvsp[-1].value));

            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
            yyval.value = fc;
        ;}
    break;

  case 163:
#line 1205 "gram.y"
    {
            yyval.value = (Expression*) NULL;
        ;}
    break;

  case 167:
#line 1214 "gram.y"
    {
            yyval.value = yyvsp[0].value;
            yyval.value->next = yyvsp[-2].value;

            delete yyvsp[-1].loc;
        ;}
    break;

  case 168:
#line 1221 "gram.y"
    {
            yyval.value = yyvsp[0].value;
            yyval.value->next = yyvsp[-2].value;

            delete yyvsp[-1].loc;
        ;}
    break;

  case 188:
#line 1268 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ReEnterScope();
        ;}
    break;

  case 189:
#line 1273 "gram.y"
    {
            yyval.decl = (Decl*) NULL;
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
        ;}
    break;

  case 190:
#line 1279 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ReEnterScope();
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
        ;}
    break;

  case 191:
#line 1284 "gram.y"
    {   gProject->Parse_TOS->parseCtxt->SetVarParam(1, !err_top_level, 0); 
            gProject->Parse_TOS->parseCtxt->SetIsKnR(true); 
        ;}
    break;

  case 192:
#line 1288 "gram.y"
    {   yyval.decl = yyvsp[0].decl;
            gProject->Parse_TOS->parseCtxt->SetIsKnR(false); 
            gProject->Parse_TOS->parseCtxt->SetVarParam(0, !err_top_level, 1); 
            
            // Exit, but will allow re-enter for a function.
            // Hack, to handle parameters being in the function's scope.
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope(true);
        ;}
    break;

  case 193:
#line 1300 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
        ;}
    break;

  case 194:
#line 1306 "gram.y"
    {
            yyval.decl = (Decl*) NULL;
        ;}
    break;

  case 195:
#line 1310 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
        ;}
    break;

  case 196:
#line 1315 "gram.y"
    {   gProject->Parse_TOS->parseCtxt->SetVarParam(0, !err_top_level, 0); 
        ;}
    break;

  case 197:
#line 1318 "gram.y"
    {   yyval.decl = yyvsp[0].decl;
            gProject->Parse_TOS->parseCtxt->SetVarParam(0, !err_top_level, 0);
        ;}
    break;

  case 198:
#line 1324 "gram.y"
    {
            yyval.decl = yyvsp[-1].decl;
            delete yyvsp[0].loc;
        ;}
    break;

  case 199:
#line 1329 "gram.y"
    {
            yyval.decl = yyvsp[-2].decl;

			Decl*	appendDecl = yyvsp[-2].decl;
			while (appendDecl->next != NULL)
				appendDecl = appendDecl->next;

            appendDecl->next = yyvsp[0].decl;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 200:
#line 1342 "gram.y"
    {
            yyval.declStemnt = new DeclStemnt(*yyvsp[0].loc);
            yyval.declStemnt->addDecls(ReverseList(yyvsp[-1].decl));
            delete yyvsp[0].loc;
        ;}
    break;

  case 201:
#line 1349 "gram.y"
    {
            yyval.declStemnt = new DeclStemnt(*yyvsp[0].loc);
            yyval.declStemnt->addDecls(ReverseList(yyvsp[-1].decl));
            delete yyvsp[0].loc;
        ;}
    break;

  case 202:
#line 1357 "gram.y"
    {
            assert (err_top_level ||
                    yyvsp[-1].base == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
            
            yywarn("old-style declaration or incorrect type");

            possibleType = true;
            yyval.decl = yyvsp[0].decl;

            if (yyval.decl == NULL)
            {
                yyval.decl = new Decl(yyvsp[-1].base);
            }
        ;}
    break;

  case 203:
#line 1375 "gram.y"
    {
            assert (1||err_top_level ||
                    yyvsp[-1].base == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            if (yyvsp[-1].base!=gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs) {
              if (!err_top_level) {
                baseTypeFixup(yyvsp[-1].base,yyvsp[0].decl);
              }
            }
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();            
            
            possibleType = true;
            yyval.decl = yyvsp[0].decl;
            
            if (yyval.decl == NULL)
            {
                yyval.decl = new Decl(yyvsp[-1].base);
            }
        ;}
    break;

  case 204:
#line 1400 "gram.y"
    {
            yyval.base = new BaseType(BT_Int);
            gProject->Parse_TOS->parseCtxt->SetDeclCtxt(yyval.base);
        ;}
    break;

  case 207:
#line 1419 "gram.y"
    {   
            gProject->Parse_TOS->parseCtxt->PushCtxt();
            gProject->Parse_TOS->parseCtxt->ResetVarParam();
        ;}
    break;

  case 208:
#line 1424 "gram.y"
    {
            yyval.type = yyvsp[0].type;
            gProject->Parse_TOS->parseCtxt->PopCtxt(false);
        ;}
    break;

  case 209:
#line 1431 "gram.y"
    {
            assert (yyvsp[0].base == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            
            possibleType = true;
            yyval.type = yyvsp[0].base;
            if (yyval.type->isFunction())
                yyerr ("Function type not allowed as type name");
        ;}
    break;

  case 210:
#line 1440 "gram.y"
    {
            assert (yyvsp[-1].base == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            
            possibleType = true;
            yyval.type = yyvsp[0].type;
            
            Type * extended = yyval.type->extend(yyvsp[-1].base);
            if (yyval.type->isFunction())
                yyerr ("Function type not allowed as type name");
            else if (extended && 
                yyvsp[-1].base && yyvsp[-1].base->isFunction() && 
                ! extended->isPointer())
                yyerr ("Wrong type combination") ;
                
        ;}
    break;

  case 211:
#line 1463 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->SetDeclCtxt(yyval.base);
        ;}
    break;

  case 218:
#line 1479 "gram.y"
    {
            if (! gProject->Parse_TOS->transUnit ||
                gProject->Parse_TOS->transUnit->contxt.syms->current->level >= FUNCTION_SCOPE)
                 yyval.storage = yyvsp[0].storage ;             
             else
                 yyval.storage = ST_None ;              
        ;}
    break;

  case 220:
#line 1492 "gram.y"
    {
            yyval.base = (BaseType*) NULL;
        ;}
    break;

  case 222:
#line 1499 "gram.y"
    {
            yyval.base = yyvsp[0].base;

            if (!yyval.base)
            {
                yyval.base = new BaseType();
            }

            if (yyvsp[-1].storage == ST_None)
                 yyerr("Invalid use of local storage type");
            else if (yyval.base->storage != ST_None)             
                 yywarn("Overloading previous storage type specification");
            else
                 yyval.base->storage = yyvsp[-1].storage;
        ;}
    break;

  case 223:
#line 1514 "gram.y"
    { possibleType = false; ;}
    break;

  case 224:
#line 1515 "gram.y"
    {
            yyval.base = yyvsp[-2].base;

            if (yyvsp[0].base)
            {
                if ((yyvsp[0].base->typemask & BT_Long)
                    && (yyval.base->typemask & BT_Long))
                {
                   // long long : A likely C9X addition 
                   yyerr("long long support has been removed");
                }
                else
                    yyval.base->typemask |= yyvsp[0].base->typemask;

                if (yyvsp[0].base->storage != ST_None)
                    yyval.base->storage = yyvsp[0].base->storage;

                // delete $3;
            }

            /*
            std::cout << "In decl_spec: ";
            $$->printBase(std::cout,0);
            if ($$->storage == ST_Typedef)
                std::cout << "(is a typedef)";
            std::cout << std::endl;
            */
        ;}
    break;

  case 225:
#line 1544 "gram.y"
    {
            yyval.base = yyvsp[0].base;

            if (!yyval.base)
            {
                yyval.base = new BaseType();
            }

            if (TQ_None != (yyval.base->qualifier & yyvsp[-1].typeQual))
                yywarn("qualifier already specified");  
                              
            yyval.base->qualifier |= yyvsp[-1].typeQual;

        ;}
    break;

  case 226:
#line 1564 "gram.y"
    {
           yyval.base = (BaseType*) NULL;
        ;}
    break;

  case 228:
#line 1570 "gram.y"
    { possibleType = false; ;}
    break;

  case 229:
#line 1571 "gram.y"
    {
            yyval.base = yyvsp[-2].base;

            if (yyvsp[0].base)
            {
                yyval.base->typemask |= yyvsp[0].base->typemask;
                // delete $3;
            }
        ;}
    break;

  case 230:
#line 1581 "gram.y"
    {
            yyval.base = yyvsp[0].base;

            if (!yyval.base)
            {
                yyval.base = new BaseType();
            }

            if (TQ_None != (yyval.base->qualifier & yyvsp[-1].typeQual))
                yywarn("qualifier already specified");
            yyval.base->qualifier |= yyvsp[-1].typeQual;
        ;}
    break;

  case 231:
#line 1596 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->SetDeclCtxt(yyval.base);
        ;}
    break;

  case 232:
#line 1605 "gram.y"
    {
           yyvsp[0].decl->extend(gProject->Parse_TOS->parseCtxt->UseDeclCtxt());
        ;}
    break;

  case 233:
#line 1609 "gram.y"
    {
           yyvsp[-2].decl->attrib = yyvsp[0].gccAttrib;
           yyval.decl = yyvsp[-2].decl;
        ;}
    break;

  case 235:
#line 1617 "gram.y"
    {
           yyvsp[-2].decl->initializer = yyvsp[0].value;
           yyval.decl = yyvsp[-2].decl;
        ;}
    break;

  case 236:
#line 1624 "gram.y"
    {
          yyval.decl = (Decl*) NULL;
        ;}
    break;

  case 239:
#line 1634 "gram.y"
    {
            yyval.decl = yyvsp[0].decl;
        ;}
    break;

  case 240:
#line 1638 "gram.y"
    {
            yyval.decl = yyvsp[-2].decl;

			Decl*	appendDecl = yyvsp[-2].decl;
			while (appendDecl->next != NULL)
				appendDecl = appendDecl->next;

            appendDecl->next = yyvsp[0].decl;
            delete yyvsp[-1].loc;
        ;}
    break;

  case 242:
#line 1656 "gram.y"
    {
            yyval.arrayConst = new ArrayConstant(NoLocation);
            yyval.arrayConst->addElement(yyvsp[0].value);
        ;}
    break;

  case 243:
#line 1661 "gram.y"
    {
            yyval.arrayConst = yyvsp[-2].arrayConst;
            yyval.arrayConst->addElement(yyvsp[0].value);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 247:
#line 1672 "gram.y"
    {
            yyval.value = yyvsp[-2].arrayConst;
            delete yyvsp[-3].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 248:
#line 1680 "gram.y"
    {
            yyval.loc = (Location*) NULL;
        ;}
    break;

  case 249:
#line 1684 "gram.y"
    {
            delete yyvsp[0].loc;
            yyval.loc = (Location*) NULL;
        ;}
    break;

  case 257:
#line 1705 "gram.y"
    {
           TypeQual r(yyvsp[-3].typeQual);
           r.vout=yyvsp[-1].value;
           yyval.typeQual = r;
        ;}
    break;

  case 259:
#line 1714 "gram.y"
    {
            yyval.typeQual = yyvsp[-1].typeQual | yyvsp[0].typeQual;
            if (TQ_None != (yyvsp[0].typeQual & yyvsp[-1].typeQual))
                yywarn("qualifier already specified");                               
        ;}
    break;

  case 260:
#line 1722 "gram.y"
    {
            yyval.typeQual = TQ_None;
        ;}
    break;

  case 290:
#line 1763 "gram.y"
    {
            yyval.base = new BaseType(BT_UserType);
            yyval.base->typeName = yyvsp[0].symbol;
        ;}
    break;

  case 291:
#line 1770 "gram.y"
    {
            assert ((! yyval.symbol->entry) || 
                    yyval.symbol->entry->IsTagDecl()) ;
        ;}
    break;

  case 292:
#line 1778 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                yyval.base = gProject->Parse_TOS->parseCtxt->Mk_tag_ref(yyvsp[-1].typeSpec, yyvsp[0].symbol,
                                                                gProject->Parse_TOS->transUnit->contxt.tags);
            else
                yyval.base = NULL;                                         
        ;}
    break;

  case 293:
#line 1788 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                yyval.base = gProject->Parse_TOS->parseCtxt->Mk_tag_ref(yyvsp[-1].typeSpec, yyvsp[0].symbol,
                                                                gProject->Parse_TOS->transUnit->contxt.tags);
            else
                yyval.base = NULL;                                         
        ;}
    break;

  case 294:
#line 1798 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                yyval.base = gProject->Parse_TOS->parseCtxt->Mk_tag_ref(yyvsp[-1].typeSpec, yyvsp[0].symbol,
                                                                gProject->Parse_TOS->transUnit->contxt.tags);
            else
                yyval.base = NULL;                                         
        ;}
    break;

  case 295:
#line 1808 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                yyval.base = gProject->Parse_TOS->parseCtxt->Mk_tag_def(yyvsp[-1].typeSpec, yyvsp[0].symbol,
                                                            gProject->Parse_TOS->transUnit->contxt.tags);
            else
                yyval.base = NULL;                                         
        ;}
    break;

  case 296:
#line 1818 "gram.y"
    {
            yyval.base = new BaseType(yyvsp[-1].strDef);
            yyvsp[-1].strDef->_isUnion = false;
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 297:
#line 1825 "gram.y"
    {
            yyval.base = yyvsp[-3].base;
            assert (! yyval.base->stDefn);
            yyval.base->stDefn = yyvsp[-1].strDef;
            yyvsp[-1].strDef->tag = yyvsp[-3].base->tag->dup();
            yyvsp[-1].strDef->_isUnion = false;

            // Overload the incomplete definition
            yyval.base->tag->entry->uStructDef = yyval.base ;
            
//             std::cout << "struct/union/enum_type_define:"
//                          "The definition of:"
//                       << "(uStructDef:" << $1->tag->entry->uStructDef << ")"
//                       << "(uStructDef->stDefn:" << $1->tag->entry->uStructDef->stDefn << ")"
//                       << "(" << $1->tag->entry << ")" << $1->tag->name  << "$" ;
//             $1->tag->entry->scope->ShowScopeId(std::cout);
//             std::cout << " has been completed" << endl; 
            
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 298:
#line 1849 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                yyval.base = gProject->Parse_TOS->parseCtxt->Mk_tag_def(yyvsp[-1].typeSpec, yyvsp[0].symbol,
                                                            gProject->Parse_TOS->transUnit->contxt.tags);
            else
              yyval.base = NULL ;
        ;}
    break;

  case 299:
#line 1859 "gram.y"
    {
            yyval.base = new BaseType(yyvsp[-1].strDef);
            yyvsp[-1].strDef->_isUnion = true;

            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 300:
#line 1867 "gram.y"
    {
            yyval.base = yyvsp[-3].base;
            assert (! yyval.base->stDefn);
            yyval.base->stDefn = yyvsp[-1].strDef;
            yyvsp[-1].strDef->tag = yyvsp[-3].base->tag->dup();
            yyvsp[-1].strDef->_isUnion = true;

            // Overload the incomplete definition
            yyval.base->tag->entry->uStructDef = yyval.base ;
            
//             std::cout << "struct/union/enum_type_define:"
//                          "The definition of:"
//                       << "(uStructDef:" << $1->tag->entry->uStructDef << ")"
//                       << "(uStructDef->stDefn:" << $1->tag->entry->uStructDef->stDefn << ")"
//                       << "(" << $1->tag->entry << ")" << $1->tag->name  << "$" ;
//             $1->tag->entry->scope->ShowScopeId(std::cout);
//             std::cout << " has been completed" << endl; 
            
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
 
        ;}
    break;

  case 301:
#line 1892 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                yyval.base = gProject->Parse_TOS->parseCtxt->Mk_tag_def(yyvsp[-1].typeSpec,yyvsp[0].symbol,
                                                            gProject->Parse_TOS->transUnit->contxt.tags);
            else
              yyval.base = NULL;
        ;}
    break;

  case 302:
#line 1902 "gram.y"
    {
            yyval.base = new BaseType(yyvsp[-1].enDef);

            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 303:
#line 1909 "gram.y"
    {
            yyval.base = yyvsp[-3].base;
            assert (! yyval.base->enDefn);
            yyval.base->enDefn = yyvsp[-1].enDef;
            yyvsp[-1].enDef->tag = yyvsp[-3].base->tag->dup();

            // Overload the incomplete definition
            yyval.base->tag->entry->uStructDef = yyval.base ;
            
//             std::cout << "struct/union/enum_type_define:"
//                          "The definition of:"
//                       << "(uStructDef:" << $1->tag->entry->uStructDef << ")"
//                       << "(uStructDef->stDefn:" << $1->tag->entry->uStructDef->stDefn << ")"
//                       << "(" << $1->tag->entry << ")" << $1->tag->name  << "$" ;
//             $1->tag->entry->scope->ShowScopeId(std::cout);
//             std::cout << " has been completed" << endl; 
            
            delete yyvsp[-2].loc;
            delete yyvsp[0].loc;
        ;}
    break;

  case 304:
#line 1932 "gram.y"
    {  yyval.strDef = new StructDef();
           yywarn("ANSI/ISO C prohibits empty struct/union");
        ;}
    break;

  case 306:
#line 1939 "gram.y"
    {  yyval.enDef = new EnumDef();
           yywarn("ANSI/ISO C prohibits empty enum");
        ;}
    break;

  case 307:
#line 1943 "gram.y"
    {  yyval.enDef = yyvsp[-1].enDef;
        ;}
    break;

  case 308:
#line 1948 "gram.y"
    {
            yyval.loc = NULL;
        ;}
    break;

  case 309:
#line 1952 "gram.y"
    {
          yywarn("Trailing comma in enum type definition");
        ;}
    break;

  case 311:
#line 1965 "gram.y"
    {
            yyval.enDef = new EnumDef();
            yyval.enDef->addElement(yyvsp[0].enConst);
        ;}
    break;

  case 312:
#line 1970 "gram.y"
    {
            yyval.enDef = yyvsp[-2].enDef;
            yyval.enDef->addElement(yyvsp[0].enConst);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 313:
#line 1978 "gram.y"
    {
            yyval.enConst = new EnumConstant(yyvsp[0].symbol,NULL,NoLocation);
            if (gProject->Parse_TOS->transUnit)
            {
              if (gProject->Parse_TOS->transUnit->contxt.syms->IsDefined(yyvsp[0].symbol->name))
                 yyerr("Duplicate enumeration constant");
                 
              yyvsp[0].symbol->entry = gProject->Parse_TOS->transUnit->contxt.syms->Insert(
                                  mk_enum_const(yyvsp[0].symbol->name, yyval.enConst));
            }
        ;}
    break;

  case 314:
#line 1990 "gram.y"
    {
            yyval.enConst = new EnumConstant(yyvsp[-2].symbol,yyvsp[0].value,NoLocation);
            if (gProject->Parse_TOS->transUnit)
            {
              if (gProject->Parse_TOS->transUnit->contxt.syms->IsDefined(yyvsp[-2].symbol->name))
                 yyerr("Duplicate enumeration constant");
                 
              yyvsp[-2].symbol->entry = gProject->Parse_TOS->transUnit->contxt.syms->Insert(
                                  mk_enum_const(yyvsp[-2].symbol->name, yyval.enConst));
            }
        ;}
    break;

  case 316:
#line 2010 "gram.y"
    {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
            gProject->Parse_TOS->parseCtxt->PushCtxt();
        ;}
    break;

  case 317:
#line 2015 "gram.y"
    {
            assert (!err_top_level || possibleType);
             /* Safety precaution! */
             possibleType=true;
        ;}
    break;

  case 318:
#line 2021 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->PopCtxt(false);
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
            yyval.strDef = yyvsp[0].strDef ;
        ;}
    break;

  case 319:
#line 2030 "gram.y"
    {
            yyval.strDef = new StructDef();
            yyval.strDef->addComponent(ReverseList(yyvsp[-1].decl));
            delete yyvsp[0].loc;
        ;}
    break;

  case 320:
#line 2036 "gram.y"
    {
            // A useful gcc extension:
            //   naked semicolons in struct/union definitions. 
            yyval.strDef = yyvsp[-1].strDef;
            yywarn ("Empty declaration");
            delete yyvsp[0].loc;
        ;}
    break;

  case 321:
#line 2044 "gram.y"
    {
            yyval.strDef = yyvsp[-2].strDef;
            yyval.strDef->addComponent(ReverseList(yyvsp[-1].decl));
            delete yyvsp[0].loc;
        ;}
    break;

  case 322:
#line 2052 "gram.y"
    {
            possibleType = true;
            yyval.decl = yyvsp[0].decl;
        ;}
    break;

  case 323:
#line 2057 "gram.y"
    {
            possibleType = true;
            yyval.decl = new Decl (yyvsp[0].base);
            yywarn ("No field declarator");
        ;}
    break;

  case 325:
#line 2071 "gram.y"
    {   gProject->Parse_TOS->parseCtxt->SetIsFieldId(true); 
        ;}
    break;

  case 326:
#line 2074 "gram.y"
    {
            yyval.decl = yyvsp[-1].decl;
            yyval.decl->attrib = yyvsp[0].gccAttrib;
        ;}
    break;

  case 327:
#line 2079 "gram.y"
    {   gProject->Parse_TOS->parseCtxt->SetIsFieldId(true); 
        ;}
    break;

  case 328:
#line 2082 "gram.y"
    {
            yyval.decl = yyvsp[-1].decl;
            yyval.decl->attrib = yyvsp[0].gccAttrib;
            yyval.decl->next = yyvsp[-4].decl;
            delete yyvsp[-3].loc;
        ;}
    break;

  case 329:
#line 2091 "gram.y"
    {
           gProject->Parse_TOS->parseCtxt->SetIsFieldId(false); 
           Type * decl = gProject->Parse_TOS->parseCtxt->UseDeclCtxt();
           Type * extended = yyval.decl->extend(decl);
           if (yyval.decl->form &&
               yyval.decl->form->isFunction())
               yyerr ("Function type not allowed as field");
           else if (yyval.decl->form &&
                    yyval.decl->form->isArray() &&
                    ! ((ArrayType *) yyval.decl->form)->size)
               yyerr ("Unsized array not allowed as field");
           else if (extended && 
               decl && decl->isFunction() && 
               ! extended->isPointer())
               yyerr ("Wrong type combination") ;
                
        ;}
    break;

  case 330:
#line 2109 "gram.y"
    {
           Type * decl = gProject->Parse_TOS->parseCtxt->UseDeclCtxt();
           yyval.decl->extend(decl);
           if (! decl)
               yyerr ("No type specifier for bit field") ;
           else if (!yyval.decl->form)
               yyerr ("Wrong type combination") ;
        ;}
    break;

  case 332:
#line 2123 "gram.y"
    {   gProject->Parse_TOS->parseCtxt->SetIsFieldId(false); 
        ;}
    break;

  case 333:
#line 2126 "gram.y"
    {
            BitFieldType  *bf = new BitFieldType(yyvsp[0].value);
            yyval.decl = yyvsp[-3].decl;

            if (yyval.decl == NULL)
            {
                yyval.decl = new Decl(bf);
            }
            else
            {
                bf->subType = yyval.decl->form;
                yyval.decl->form = bf;
            }
        ;}
    break;

  case 335:
#line 2146 "gram.y"
    {
           yyval.decl = (Decl*) NULL;
        ;}
    break;

  case 337:
#line 2157 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->Mk_declarator (yyval.decl);
        ;}
    break;

  case 338:
#line 2163 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->Mk_func_declarator (yyval.decl);
        ;}
    break;

  case 339:
#line 2169 "gram.y"
    {
            yyval.decl = yyvsp[0].decl;
            yyval.decl->extend(yyvsp[-1].ptr);
        ;}
    break;

  case 342:
#line 2180 "gram.y"
    {  if (gProject->Parse_TOS->transUnit)
                yyval.decl = gProject->Parse_TOS->parseCtxt->Mk_direct_declarator_reentrance (yyvsp[0].symbol,
                gProject->Parse_TOS->transUnit->contxt.syms);
        ;}
    break;

  case 343:
#line 2185 "gram.y"
    {
            yyval.decl = yyvsp[-1].decl;
            delete yyvsp[-2].loc ;
            delete yyvsp[0].loc ;
        ;}
    break;

  case 346:
#line 2193 "gram.y"
    {
            yyval.decl = yyvsp[-3].decl;
            FunctionType * ft = new FunctionType(ReverseList(yyvsp[-1].decl));
            Type * extended = yyval.decl->extend(ft);
            if (extended && ! extended->isPointer())
                yyerr ("Wrong type combination") ;
                
            delete yyvsp[-2].loc ;
            delete yyvsp[0].loc ;
            // Exit, but will allow re-enter for a function.
            // Hack, to handle parameters being in the function's scope.
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope(true);

        ;}
    break;

  case 347:
#line 2209 "gram.y"
    {
            yyval.decl = yyvsp[-3].decl;
            FunctionType * ft = new FunctionType(ReverseList(yyvsp[-1].decl));
            Type * extended = yyval.decl->extend(ft);
            if (extended && ! extended->isPointer())
                yyerr ("Wrong type combination") ;

            delete yyvsp[-2].loc ;
            delete yyvsp[0].loc ;
            // Exit, but will allow re-enter for a function.
            // Hack, to handle parameters being in the function's scope.
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope(true);

        ;}
    break;

  case 348:
#line 2225 "gram.y"
    {
            yyval.decl = yyvsp[-2].decl;

			if (yyval.decl != NULL)
			{
				FunctionType* ft = new FunctionType();
				Type* extended = yyval.decl->extend(ft);
				if (extended && ! extended->isPointer())
           	 	    yyerr ("Wrong type combination") ;
			}
            
            delete yyvsp[-1].loc ;
            delete yyvsp[0].loc ;
            if (gProject->Parse_TOS->transUnit)
            {
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
                // Exit, but will allow re-enter for a function.
                // Hack, to handle parameters being in the function's scope.
                gProject->Parse_TOS->transUnit->contxt.ExitScope(true);
            }
        ;}
    break;

  case 349:
#line 2249 "gram.y"
    {
            yyval.decl = yyvsp[-3].decl;
            ArrayType * at = new ArrayType(TT_Array, yyvsp[-1].value);
            Type * extended = yyval.decl->extend(at);
            if (extended && 
                extended->isFunction())
                yyerr ("Wrong type combination") ;
              
            delete yyvsp[-2].loc ;
            delete yyvsp[0].loc ;
        ;}
    break;

  case 350:
#line 2263 "gram.y"
    {
            yyval.decl = yyvsp[-3].decl;
            ArrayType * at = new ArrayType(TT_Stream, yyvsp[-1].value);
            Type * extended = yyval.decl->extend(at);

            if (extended &&
                extended->isFunction())
                yyerr ("Wrong type combination") ;
        ;}
    break;

  case 351:
#line 2274 "gram.y"
    {
            yyval.value = yyvsp[0].consValue;
        ;}
    break;

  case 352:
#line 2278 "gram.y"
    {
            yyval.value = yyvsp[0].value;
        ;}
    break;

  case 353:
#line 2282 "gram.y"
    {
           yyval.value = yyvsp[-1].value;
        ;}
    break;

  case 354:
#line 2286 "gram.y"
    { 
            yyval.value = new Variable (yyvsp[0].symbol,NoLocation);
        ;}
    break;

  case 355:
#line 2292 "gram.y"
    {
	   yyval.value = NULL;
	;}
    break;

  case 356:
#line 2296 "gram.y"
    {
            yyval.value = yyvsp[0].value;
        ;}
    break;

  case 357:
#line 2300 "gram.y"
    {
            yyval.value = new BinaryExpr(BO_Comma,yyvsp[-2].value,yyvsp[0].value,*yyvsp[-1].loc);
            delete yyvsp[-1].loc;
        ;}
    break;

  case 358:
#line 2310 "gram.y"
    {
            yyval.ptr = new PtrType(yyvsp[0].typeQual);    
        ;}
    break;

  case 360:
#line 2317 "gram.y"
    {
            yyval.ptr = yyvsp[0].ptr;
            yyval.ptr->subType = yyvsp[-1].ptr;
        ;}
    break;

  case 362:
#line 2330 "gram.y"
    {  gProject->Parse_TOS->parseCtxt->IncrVarParam(1);
          if (gProject->Parse_TOS->transUnit)
              gProject->Parse_TOS->transUnit->contxt.EnterScope();
          gProject->Parse_TOS->parseCtxt->PushCtxt();
        ;}
    break;

  case 363:
#line 2336 "gram.y"
    {
          // Exit, but will allow re-enter for a function.
          // Hack, to handle parameters being in the function's scope.
          gProject->Parse_TOS->parseCtxt->PopCtxt(true);
          gProject->Parse_TOS->parseCtxt->IncrVarParam(-1);
          yyval.decl = yyvsp[0].decl;
       ;}
    break;

  case 364:
#line 2346 "gram.y"
    {  if (gProject->Parse_TOS->transUnit)
               yyval.decl = gProject->Parse_TOS->parseCtxt->Mk_direct_declarator_reentrance (yyvsp[0].symbol,
                gProject->Parse_TOS->transUnit->contxt.syms);
        ;}
    break;

  case 365:
#line 2351 "gram.y"
    {  yyval.decl = yyvsp[-2].decl;
           if (gProject->Parse_TOS->transUnit)
           {
              yyval.decl = gProject->Parse_TOS->parseCtxt->Mk_direct_declarator_reentrance (yyvsp[0].symbol,
                gProject->Parse_TOS->transUnit->contxt.syms);
              yyval.decl->next = yyvsp[-2].decl;
           }
        ;}
    break;

  case 367:
#line 2367 "gram.y"
    {
            /* Convert a TYPEDEF_NAME back into a normal IDENT */
            yyval.symbol = yyvsp[0].symbol;
            yyval.symbol->entry = (SymEntry*) NULL;
        ;}
    break;

  case 370:
#line 2388 "gram.y"
    {
           yyval.decl = (Decl*) NULL;
        ;}
    break;

  case 371:
#line 2392 "gram.y"
    { gProject->Parse_TOS->parseCtxt->IncrVarParam(1); 
        ;}
    break;

  case 372:
#line 2395 "gram.y"
    { gProject->Parse_TOS->parseCtxt->IncrVarParam(-1); 
           yyval.decl = yyvsp[0].decl;
        ;}
    break;

  case 373:
#line 2401 "gram.y"
    {   gProject->Parse_TOS->parseCtxt->IncrVarParam(1);
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
            gProject->Parse_TOS->parseCtxt->PushCtxt();
        ;}
    break;

  case 374:
#line 2407 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->PopCtxt(true);
            gProject->Parse_TOS->parseCtxt->IncrVarParam(-1);
            yyval.decl = yyvsp[0].decl ;
        ;}
    break;

  case 376:
#line 2416 "gram.y"
    {
            BaseType *bt = new BaseType(BT_Ellipsis);

            yyval.decl = new Decl(bt);
            yyval.decl->next = yyvsp[-2].decl;
        ;}
    break;

  case 378:
#line 2426 "gram.y"
    {
            yyval.decl = yyvsp[0].decl;
            yyval.decl->next = yyvsp[-2].decl;
        ;}
    break;

  case 379:
#line 2433 "gram.y"
    {   
            gProject->Parse_TOS->parseCtxt->PushCtxt();
        ;}
    break;

  case 380:
#line 2437 "gram.y"
    {
            gProject->Parse_TOS->parseCtxt->PopCtxt(true);
            yyval.decl = yyvsp[0].decl;
        ;}
    break;

  case 381:
#line 2444 "gram.y"
    {
            assert (err_top_level ||
                    yyvsp[-1].base == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            possibleType = true;
            yyval.decl = yyvsp[0].decl;
            Type * decl = gProject->Parse_TOS->parseCtxt->UseDeclCtxt();
            Type * extended = yyval.decl->extend(decl);             
            if (yyval.decl->form &&
                yyval.decl->form->isFunction())
                yyerr ("Function type not allowed");
            else if (extended && 
                decl && decl->isFunction() && 
                ! extended->isPointer())
                yyerr ("Wrong type combination") ;
        ;}
    break;

  case 382:
#line 2460 "gram.y"
    {
            assert (err_top_level ||
                    yyvsp[-1].base == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            possibleType = true;
            yyval.decl = new Decl(yyvsp[0].type);
            
            Type * decl = gProject->Parse_TOS->parseCtxt->UseDeclCtxt();
            Type * extended = yyval.decl->extend(decl);
            if (yyval.decl->form &&
                yyval.decl->form->isFunction())
                yyerr ("Function type not allowed for parameter");
            else if (extended && 
                decl && decl->isFunction() && 
                ! extended->isPointer())
                yyerr ("Wrong type combination") ;
        ;}
    break;

  case 383:
#line 2477 "gram.y"
    {
            possibleType = true;
            yyval.decl = new Decl(yyvsp[0].base);
            if (yyval.decl->form &&
                yyval.decl->form->isFunction())
                yyerr ("Function type not allowed for parameter");
        ;}
    break;

  case 384:
#line 2490 "gram.y"
    {
            yyval.type = yyvsp[0].ptr;
        ;}
    break;

  case 385:
#line 2494 "gram.y"
    {
            yyval.type = yyvsp[0].type;
        ;}
    break;

  case 386:
#line 2498 "gram.y"
    {
            yyval.type = yyvsp[0].type;
            yyval.type->extend(yyvsp[-1].ptr);
        ;}
    break;

  case 388:
#line 2508 "gram.y"
    {
            yyval.type = yyvsp[-1].type;
        ;}
    break;

  case 389:
#line 2512 "gram.y"
    {
            yyval.type = new ArrayType(TT_Array, yyvsp[-1].value);
        ;}
    break;

  case 390:
#line 2516 "gram.y"
    {
            ArrayType *at = new ArrayType(TT_Array, yyvsp[-1].value);
            yyval.type = yyvsp[-3].type;
            yyval.type->extend(at);
            Type * extended = yyval.type->extend(at) ;
            if (extended && 
                extended->isFunction())
                yyerr ("Wrong type combination") ;
        ;}
    break;

  case 391:
#line 2526 "gram.y"
    {
            yyval.type = new FunctionType(ReverseList(yyvsp[-1].decl));
        ;}
    break;

  case 392:
#line 2530 "gram.y"
    {
            FunctionType * ft = new FunctionType(ReverseList(yyvsp[-1].decl));
            yyval.type = yyvsp[-3].type;
            Type * extended = yyval.type->extend(ft) ;
            if (extended && 
                ! extended->isPointer())
                yyerr ("Wrong type combination") ;
                
        ;}
    break;

  case 393:
#line 2546 "gram.y"
    {
            yyval.gccAttrib = (GccAttrib*) NULL;
        ;}
    break;

  case 395:
#line 2553 "gram.y"
    {
                yyval.gccAttrib = yyvsp[-2].gccAttrib;
                delete yyvsp[-4].loc;
                delete yyvsp[-3].loc;
                delete yyvsp[-1].loc;
                delete yyvsp[0].loc;
            ;}
    break;

  case 396:
#line 2563 "gram.y"
    {
                /* The lexer ate some unsupported option. */
                yyval.gccAttrib = new GccAttrib( GCC_Unsupported);
            ;}
    break;

  case 397:
#line 2568 "gram.y"
    {
                yyval.gccAttrib = new GccAttrib( GCC_Packed );
            ;}
    break;

  case 398:
#line 2572 "gram.y"
    {
                yyval.gccAttrib = new GccAttrib( GCC_CDecl );
            ;}
    break;

  case 399:
#line 2576 "gram.y"
    {
                yyval.gccAttrib = new GccAttrib( GCC_Const );
            ;}
    break;

  case 400:
#line 2580 "gram.y"
    {
                yyval.gccAttrib = new GccAttrib( GCC_NoReturn );
            ;}
    break;

  case 401:
#line 2584 "gram.y"
    {
                yyval.gccAttrib = new GccAttrib( GCC_Aligned );

                if (yyvsp[-1].consValue->ctype == CT_Int)
                {
                    IntConstant    *iCons = (IntConstant*) yyvsp[-1].consValue;

                    yyval.gccAttrib->value = iCons->lng;
                }

                delete yyvsp[-2].loc;
                delete yyvsp[0].loc;
            ;}
    break;

  case 402:
#line 2598 "gram.y"
    {
                yyval.gccAttrib = new GccAttrib( GCC_Mode );

                yyval.gccAttrib->mode = yyvsp[-1].symbol;

                delete yyvsp[-2].loc;
                delete yyvsp[0].loc;
            ;}
    break;

  case 403:
#line 2607 "gram.y"
    {
                yyval.gccAttrib = new GccAttrib( GCC_Format );
    
                yyval.gccAttrib->mode = yyvsp[-5].symbol;

                if (yyvsp[-3].consValue->ctype == CT_Int)
                {
                    IntConstant    *iCons = (IntConstant*) yyvsp[-3].consValue;

                    yyval.gccAttrib->strIdx = iCons->lng;
                }

                if (yyvsp[-1].consValue->ctype == CT_Int)
                {
                    IntConstant    *iCons = (IntConstant*) yyvsp[-1].consValue;

                    yyval.gccAttrib->first = iCons->lng;
                }

                delete yyvsp[-6].loc;
                delete yyvsp[0].loc;
            ;}
    break;


    }

/* Line 999 of yacc.c.  */
#line 4667 "gram.tab.c"

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


#line 2631 "gram.y"


/*******************************************************/

