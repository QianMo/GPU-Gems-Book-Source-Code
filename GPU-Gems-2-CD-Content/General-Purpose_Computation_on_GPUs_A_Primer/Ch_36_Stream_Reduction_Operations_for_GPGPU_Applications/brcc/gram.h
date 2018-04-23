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




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif





