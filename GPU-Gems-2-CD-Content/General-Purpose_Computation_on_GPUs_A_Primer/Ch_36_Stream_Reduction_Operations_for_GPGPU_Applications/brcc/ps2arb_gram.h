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
/* Line 1252 of yacc.c.  */
#line 101 "ps2arb_gram.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE ps2arb_lval;



