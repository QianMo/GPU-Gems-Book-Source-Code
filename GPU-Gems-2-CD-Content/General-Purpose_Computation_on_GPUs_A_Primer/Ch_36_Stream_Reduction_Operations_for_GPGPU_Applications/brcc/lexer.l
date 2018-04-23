
%{
/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

    CTool Library
    Copyright (C) 1998-2001	Shaun Flisakowski

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

    o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */

#undef yywrap
extern "C" int yywrap(void);

#define  YY_BUFFER_STATE_DEFD

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cerrno>

#include "config.h"
#include "gram.h"
#include "lexer.h"
#include "token.h"
#include "symbol.h"
#include "express.h"
#include "stemnt.h"
#include "project.h"

#ifdef   WINDOWS
#    ifndef  __STDC__
#    define  __STDC__    1
#    endif

int isatty(int) { return 1; }

#endif  /* WINDOWS */

int   err_top_level =0;
int   err_cnt = 0;
int   err_tok = 0;
bool  possibleType = true;
bool  possibleTag = false;
bool  possibleLabel = false;

bool file_ppln=false;
int  in_preprocess = 0;
int  in_charconst = 0;

int  gcc2_paren_cnt = 0;

char  tokn_buff[MAX_TOKN_LEN];
char *tokn_end = &tokn_buff[MAX_TOKN_LEN-1];
char *tokn_ptr=tokn_buff;

int cnt_nl(char *txt, int len);

bool  isWide;

#undef YY_INPUT

/* Scan input from a file pointer (default) */

std::string line, last_line;
std::istream *yyinstream;

#define YY_INPUT(buf,result,max_size) \
    do { \
        last_line=line; \
        for(result=0; result<max_size; result++) \
         { \
          char ch; \
          if(!yyinstream->read(&ch, 1)) \
           { \
            if(result==0) result=YY_NULL; \
            break; \
           } \
          if(result==0) line=""; \
          line+=ch; \
          buf[result]=ch; \
          if(ch=='\n') { result++; break; } \
         } \
    } while(0)

#define YY_USER_ACTION \
    { \
        gProject->Parse_TOS->yycolno = gProject->Parse_TOS->yynxtcol; \
        gProject->Parse_TOS->yynxtcol += yyleng; \
    }


#undef HERE
#define HERE Location(gProject->Parse_TOS->yylineno, \
       gProject->Parse_TOS->yycolno, gProject->Parse_TOS->transUnit->filename )

#undef SETPOS_TOKEN
#define SETPOS_TOKEN(TOKEN) \
	do { \
          lvalp->loc = new Location( \
                gProject->Parse_TOS->yylineno, \
                gProject->Parse_TOS->yycolno,  \
                gProject->Parse_TOS->transUnit->filename ); \
	} while(0)

#undef  LAST_STATE
#define LAST_STATE() \
        do { \
          if (in_preprocess) \
            BEGIN(PP); \
          else \
            BEGIN(INITIAL); \
        } while (0)  

#undef  BEGIN_PP
#define BEGIN_PP() \
        do { \
          BEGIN(PP); \
          in_preprocess = 1; \
        } while(0)

YYSTYPE my_var;

EXTERN int yylex ARGS((YYSTYPE *lvalp));

/*  To give some args to yylex */
#undef YY_DECL
#ifdef  __STDC__
/*  #define YY_DECL int yylex YY_PROTO((YYSTYPE *lvalp))  */
#define YY_DECL int yylex (YYSTYPE *lvalp)
#else
#define YY_DECL int yylex(lvalp) YYSTYPE *lvalp;
#endif

%}
%x Start PP CMMT CC STR PPLN GCC_ELIM GCC_ATTRIB
%a 9000
%o 9000
/* %option nounput */

dot		"."
digit		[0-9]
octdigit	[0-7]
hexdigit	[0-9a-fA-F]
digits          {digit}+
alpha		[a-zA-Z_$]
alphanum	{alpha}|{digit}

usuffix         [uU]
lsuffix         [lL]
intsuffix       {usuffix}{lsuffix}?|{lsuffix}{usuffix}?
intnum		{digit}+{intsuffix}?
octnum		0{octdigit}+{intsuffix}?
hexnum		0[xX]{hexdigit}+{intsuffix}?

exponent	[Ee][+-]?{digits}
floatsuffix     [fFlL]

whitespace      [ \t\f\v]
allwhite        [ \t\f\b\v\r\n]

pp_strt         ^{whitespace}*"#"{whitespace}*

%%

<INITIAL>"/*"       { BEGIN(CMMT); }
<INITIAL>"'"        { BEGIN(CC); isWide = false; }
<INITIAL>L"'"       { BEGIN(CC); isWide = true; }
<INITIAL>\"         { BEGIN(STR);   tokn_ptr = tokn_buff; isWide = false; }
<INITIAL>L\"        { BEGIN(STR);   tokn_ptr = tokn_buff; isWide = true; }

<INITIAL>"//*"      { // Ambiguous C++ style comment, must parse as
                      //    '/' '/*' to be Ansi compliant 
                         yywarn(
                            "Ambiguous C++ comment, use '/ /*' -or- '// *'");
                         yyless(2);
                         SETPOS_TOKEN(DIV);
                         return(DIV);
                    }

<INITIAL>"//"[^*].*$ { /* C++ style comment */
#if 0
			 /* Brook supports C++ comments.  --Jeremy. */
                         if (cpp_comment_warn){
                             cpp_comment_warn = 0;
                             yywarn("C++ style comment encountered");
                         }
#endif
                     }

<INITIAL>"auto"     {
                      lvalp->storage = ST_Auto;
                      return(err_tok = AUTO);
                    }
<INITIAL>"extern"   {
                      lvalp->storage = ST_Extern;
                      return(err_tok = EXTRN);
                    }
<INITIAL>"register" {
                      lvalp->storage = ST_Register;
                      return(err_tok = REGISTR);
                    }
<INITIAL>"static"   {
                      lvalp->storage = ST_Static;
                      return(err_tok = STATIC);
                    }
<INITIAL>"typedef"  {
                      lvalp->storage = ST_Typedef;
                      return(err_tok = TYPEDEF);
                    }

<INITIAL>"const"    {
                      lvalp->typeQual = TQ_Const;
                      return(err_tok = CONST);
                    }
<INITIAL>"volatile" {
                      lvalp->typeQual = TQ_Volatile;
                      return(err_tok = VOLATILE);
                    }
<INITIAL>"out"      {
                      lvalp->typeQual = TQ_Out;
                      return(err_tok = OUT);
                    }
<INITIAL>"vout"     {
                      lvalp->typeQual = TQ_Vout;
                      return(err_tok = VOUT);
                    }
<INITIAL>"reduce"   {
                      lvalp->typeQual = TQ_Reduce;
                      return(err_tok = REDUCE);
                    }
<INITIAL>"iter"     {
                      lvalp->typeQual = TQ_Iter;
                      return(err_tok = ITER);
                    }
<INITIAL>"kernel"   {
                      lvalp->typeQual = TQ_Kernel;
                      return(err_tok = KERNEL);
                    }

<INITIAL>"void"     {
                      lvalp->base = new BaseType(BT_Void);
                      return(err_tok = VOID);
                    }
<INITIAL>"char"     {
                      lvalp->base = new BaseType(BT_Char);
                      return(err_tok = CHAR);
                    }
<INITIAL>"short"    {
                      lvalp->base = new BaseType(BT_Short);
                      return(err_tok = SHORT);
                    }
<INITIAL>"int"      {
                      lvalp->base = new BaseType(BT_Int);
                      return(err_tok = INT);
                    }
<INITIAL>"long"     {
                      lvalp->base = new BaseType(BT_Long);
                      return(err_tok = LONG);
                    }
<INITIAL>"float"    {
                      lvalp->base = new BaseType(BT_Float);
                      return(err_tok = FLOAT);
                    }
<INITIAL>"float"[2-4] {
		      int n = yytext[yyleng -1] - '1'; /* There is no float1 */

                      lvalp->base = new BaseType(BT_Float * (1<<n));
                      return(err_tok = FLOAT + n);
                    }
<INITIAL>"fixed"    {
                      lvalp->base = new BaseType(BT_Fixed);
                      return(err_tok = FIXED);
                    }
<INITIAL>"fixed"[2-4] {
		      int n = yytext[yyleng -1] - '1'; /* There is no float1 */
                      lvalp->base = new BaseType(BT_Fixed * (1<<n));
                      return(err_tok = FIXED + n);
                    }
<INITIAL>"half"    {
                      lvalp->base = new BaseType(BT_Half);
                      return(err_tok = HALF);
                    }
<INITIAL>"half"[2-4] {
		      int n = yytext[yyleng -1] - '1'; /* There is no float1 */

                      lvalp->base = new BaseType(BT_Half * (1<<n));
                      return(err_tok = HALF + n);
                    }
<INITIAL>"double"   {
                      lvalp->base = new BaseType(BT_Double);
                      return(err_tok = DOUBLE);
                    }
<INITIAL>"double2"   {
                      lvalp->base = new BaseType(BT_Double2);
                      return(err_tok = DOUBLE2);
                    }
<INITIAL>"signed"   {
                      lvalp->base = new BaseType(BT_Signed);
                      return(err_tok = SGNED);
                    }
<INITIAL>"unsigned" {
                      lvalp->base = new BaseType(BT_UnSigned);
                      return(err_tok = UNSGNED);
                    }                   
<INITIAL>"enum"     {
                      lvalp->typeSpec = BT_Enum;
                      possibleTag = true;
                      return(err_tok = ENUM);
                    }
<INITIAL>"struct"   { 
                      lvalp->typeSpec = BT_Struct;
                      possibleTag = true;
                      return(err_tok = STRUCT);
                    }
<INITIAL>"union"    { 
                      lvalp->typeSpec = BT_Union;
                      possibleTag = true;
                      return(err_tok = UNION);
                    }
<INITIAL>"break"    { SETPOS_TOKEN(BREAK); return(BREAK); }
<INITIAL>"case"     { SETPOS_TOKEN(CASE); return(CASE); }
<INITIAL>"continue" { SETPOS_TOKEN(CONT); return(CONT); }
<INITIAL>"default"  { SETPOS_TOKEN(DEFLT); return(DEFLT); }
<INITIAL>"do"       { SETPOS_TOKEN(DO); return(DO); }
<INITIAL>"else"     { SETPOS_TOKEN(ELSE); return(ELSE); }
<INITIAL>"for"      { SETPOS_TOKEN(FOR); return(FOR); }
<INITIAL>"goto"     { SETPOS_TOKEN(GOTO); 
                      possibleLabel = true;
                      return(GOTO); }
<INITIAL>"if"       { SETPOS_TOKEN(IF); return(IF); }
<INITIAL>"return"   { SETPOS_TOKEN(RETURN); return(RETURN); }
<INITIAL>"sizeof"   { SETPOS_TOKEN(SIZEOF); return(SIZEOF); }
<INITIAL>"indexof"  { SETPOS_TOKEN(INDEXOF); return(INDEXOF); }
<INITIAL>"switch"   { SETPOS_TOKEN(SWITCH); return(SWITCH); }
<INITIAL>"while"    { SETPOS_TOKEN(WHILE); return(WHILE); }

  /* GCC2 Extensions */
<INITIAL>"asm"        { BEGIN(GCC_ELIM); }
<INITIAL>"__asm"      { BEGIN(GCC_ELIM); }
<INITIAL>"__asm__"    { BEGIN(GCC_ELIM); }

<GCC_ELIM>"("[^()]*   { gcc2_paren_cnt++; }
<GCC_ELIM>[^()]*      { /* Throw away */ }
<GCC_ELIM>")"         { if (--gcc2_paren_cnt == 0)
                          BEGIN(INITIAL);
                      }

<INITIAL>"__attribute"    { BEGIN(GCC_ATTRIB); return(err_tok = ATTRIBUTE); }
<INITIAL>"__attribute__"  { BEGIN(GCC_ATTRIB); return(err_tok = ATTRIBUTE); }

<GCC_ATTRIB>"aligned"     { BEGIN(INITIAL); return(err_tok = ALIGNED); }
<GCC_ATTRIB>"__aligned"   { BEGIN(INITIAL); return(err_tok = ALIGNED); }
<GCC_ATTRIB>"__aligned__" { BEGIN(INITIAL); return(err_tok = ALIGNED); }
<GCC_ATTRIB>"packed"      { BEGIN(INITIAL); return(err_tok = PACKED); }
<GCC_ATTRIB>"__packed"    { BEGIN(INITIAL); return(err_tok = PACKED); }
<GCC_ATTRIB>"__packed__"  { BEGIN(INITIAL); return(err_tok = PACKED); }
<GCC_ATTRIB>"cdecl"       { BEGIN(INITIAL); return(err_tok = CDECL); }
<GCC_ATTRIB>"__cdecl"     { BEGIN(INITIAL); return(err_tok = CDECL); }
<GCC_ATTRIB>"__cdecl__"   { BEGIN(INITIAL); return(err_tok = CDECL); }

<GCC_ATTRIB>{whitespace}+ {  /* space/tab/formfeed/vertical tab (ignore) */ }
<GCC_ATTRIB>"("           { SETPOS_TOKEN(LPAREN); return(LPAREN); }

<GCC_ATTRIB>"format"      { BEGIN(INITIAL); return(err_tok = FORMAT); }
<GCC_ATTRIB>"__format"    { BEGIN(INITIAL); return(err_tok = FORMAT); }
<GCC_ATTRIB>"__format__"  { BEGIN(INITIAL); return(err_tok = FORMAT); }
<GCC_ATTRIB>"mode"        { BEGIN(INITIAL); return(err_tok = MODE); }
<GCC_ATTRIB>"__mode"      { BEGIN(INITIAL); return(err_tok = MODE); }
<GCC_ATTRIB>"__mode__"    { BEGIN(INITIAL); return(err_tok = MODE); }
<GCC_ATTRIB>"const"       { BEGIN(INITIAL); return(err_tok = CONST); }
<GCC_ATTRIB>"__const"     { BEGIN(INITIAL); return(err_tok = CONST); }
<GCC_ATTRIB>"__const__"   { BEGIN(INITIAL); return(err_tok = CONST); }
<GCC_ATTRIB>"noreturn"    { BEGIN(INITIAL); return(err_tok = NORETURN); }
<GCC_ATTRIB>"__noreturn"  { BEGIN(INITIAL); return(err_tok = NORETURN); }
<GCC_ATTRIB>"__noreturn__" { BEGIN(INITIAL); return(err_tok = NORETURN); }

<INITIAL>"__const"  {
                      lvalp->typeQual = TQ_Const;
                      return(err_tok = CONST);
                    }

<INITIAL>"__const__"  {
                        lvalp->typeQual = TQ_Const;
                        return(err_tok = CONST);
                      }
<INITIAL>"__signed" {
                      lvalp->type = new BaseType(BT_Signed);
                      return(err_tok = SGNED);
                    }
<INITIAL>"__signed__" {
                        lvalp->type = new BaseType(BT_Signed);
                        return(err_tok = SGNED);
                      }
<INITIAL>"__volatile" {
                        lvalp->typeQual = TQ_Volatile;
                        return(err_tok = VOLATILE);
                      }
<INITIAL>"__volatile__" {
                          lvalp->typeQual = TQ_Volatile;
                          return(err_tok = VOLATILE);
                        }

<INITIAL>"typeof"      { }
<INITIAL>"__typeof"    { }
<INITIAL>"__typeof__"  { }

<INITIAL>"inline"      { }
<INITIAL>"__inline"    { }
<INITIAL>"__inline__"  { }

<INITIAL>"__extension__"  { }

<INITIAL>"alignof"      { }
<INITIAL>"__alignof"    { }
<INITIAL>"__alignof__"  { }

<INITIAL>"__imag"     { }
<INITIAL>"__imag__"   { }
<INITIAL>"__real"     { }
<INITIAL>"__real__"   { }
<INITIAL>"__complex"    { }
<INITIAL>"__complex__"  { }
<INITIAL>"__iterator"   { }
<INITIAL>"__iterator__" { }
<INITIAL>"__label__"    { }

<INITIAL>"__cdecl"      { /* Windows only non-standard C crud */ }
<INITIAL>"__stdcall"    { /* Windows only non-standard C crud */ }
<INITIAL>"__fastcall"   { /* Windows only non-standard C crud */ }


<INITIAL>"+"        { lvalp->binOp = BO_Plus; return(PLUS); }
<INITIAL>"-"        { lvalp->binOp = BO_Minus; return(MINUS); }
<INITIAL>"*"        { lvalp->binOp = BO_Mult; return(STAR); }
<INITIAL>"/"        { lvalp->binOp = BO_Div; return(DIV); }
<INITIAL>"%"        { lvalp->binOp = BO_Mod; return(MOD); }

<INITIAL>"+="       { lvalp->assignOp = AO_PlusEql; return(ASSIGN); }
<INITIAL>"-="       { lvalp->assignOp = AO_MinusEql; return(ASSIGN); }
<INITIAL>"*="       { lvalp->assignOp = AO_MultEql; return(ASSIGN); }
<INITIAL>"/="       { lvalp->assignOp = AO_DivEql; return(ASSIGN); }
<INITIAL>"%="       { lvalp->assignOp = AO_ModEql; return(ASSIGN); }

<INITIAL>"!"        { SETPOS_TOKEN(NOT); return(NOT); }
<INITIAL>"&&"       { SETPOS_TOKEN(AND); return(AND); }
<INITIAL>"||"       { SETPOS_TOKEN(OR); return(OR); }

<INITIAL>"~"        { SETPOS_TOKEN(B_NOT); return(B_NOT); }
<INITIAL>"&"        { SETPOS_TOKEN(B_AND); return(B_AND); }
<INITIAL>"|"        { SETPOS_TOKEN(B_OR); return(B_OR); }
<INITIAL>"^"        { SETPOS_TOKEN(B_XOR); return(B_XOR); }

<INITIAL>"&="       { lvalp->assignOp = AO_BitAndEql; return(ASSIGN); }
<INITIAL>"|="       { lvalp->assignOp = AO_BitOrEql; return(ASSIGN); }
<INITIAL>"^="       { lvalp->assignOp = AO_BitXorEql; return(ASSIGN); }

<INITIAL>"<<"       { lvalp->binOp = BO_Shl; return(L_SHIFT); }
<INITIAL>">>"       { lvalp->binOp = BO_Shr; return(R_SHIFT); }
<INITIAL>"<<="      { lvalp->assignOp = AO_ShlEql; return(ASSIGN); }
<INITIAL>">>="      { lvalp->assignOp = AO_ShrEql; return(ASSIGN); }

<INITIAL>"=="       { lvalp->relOp = RO_Equal; return(COMP_EQ); }
<INITIAL>"<"        { lvalp->relOp = RO_Less; return(COMP_LESS); }
<INITIAL>"<="       { lvalp->relOp = RO_LessEql; return(COMP_ARITH); }
<INITIAL>">"        { lvalp->relOp = RO_Grtr; return(COMP_GRTR); }
<INITIAL>">="       { lvalp->relOp = RO_GrtrEql; return(COMP_ARITH); }
<INITIAL>"!="       { lvalp->relOp = RO_NotEqual; return(COMP_EQ); }

<INITIAL>"="        { lvalp->assignOp = AO_Equal; return(EQ); }
<INITIAL>"++"       { SETPOS_TOKEN(INCR); return(INCR); }
<INITIAL>"--"       { SETPOS_TOKEN(DECR); return(DECR); }

<INITIAL>"("        { SETPOS_TOKEN(LPAREN);
                      possibleType = true;
                      return(LPAREN);
                    }
<INITIAL>")"        { SETPOS_TOKEN(RPAREN); return(RPAREN); }
<INITIAL>"["        { SETPOS_TOKEN(LBRCKT); return(LBRCKT); }
<INITIAL>"]"        { SETPOS_TOKEN(RBRCKT); return(RBRCKT); }
<INITIAL>"{"        { SETPOS_TOKEN(LBRACE); 
                      possibleTag = false;
                      return(LBRACE); }
<INITIAL>"}"        { SETPOS_TOKEN(RBRACE); return(RBRACE); }

<INITIAL>"."        { SETPOS_TOKEN(DOT); return(DOT); }
<INITIAL>"->"       { SETPOS_TOKEN(ARROW); return(ARROW); }

<INITIAL>"?"        { SETPOS_TOKEN(QUESTMARK); return(QUESTMARK); }
<INITIAL>":"        { SETPOS_TOKEN(COLON); return(COLON); }
<INITIAL>";"        { SETPOS_TOKEN(SEMICOLON); return(SEMICOLON); }
<INITIAL>","        { SETPOS_TOKEN(COMMA); return(COMMA); }
<INITIAL>"..."      { SETPOS_TOKEN(ELLIPSIS); return(ELLIPSIS); }

	/* Unused (invalid) characters */
<INITIAL>"`"        { SETPOS_TOKEN(BACKQUOTE); return(BACKQUOTE); }
<INITIAL>"@"        { SETPOS_TOKEN(AT); return(AT); }

	/* Preprocessor Stuff */
<INITIAL>{pp_strt}\n       { gProject->Parse_TOS->yylineno++;
                             gProject->Parse_TOS->yynxtcol = 0; }

<INITIAL>{pp_strt}"line"{whitespace}* { BEGIN(PPLN); }
<INITIAL>{pp_strt}"file"{whitespace}* { file_ppln=true; BEGIN(PPLN); }
<INITIAL>{pp_strt}{digit} { BEGIN(PPLN); yyless(1); }
<INITIAL>{pp_strt} { BEGIN_PP(); tokn_ptr = tokn_buff; *tokn_ptr++ = '#'; }

<PPLN>[^\n]*	{ 
                       BEGIN_PP();
                       if (get_lineno(file_ppln,yytext,&(lvalp->stemnt)))
							return (err_tok = PP_LINE);
                }

<PP>\\(\n)             {  /* Preprocessor continuation line */
                         gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 0;
			 if (tokn_ptr + 2 < tokn_end) {
			    *tokn_ptr++ = '\\';
			    *tokn_ptr++ = '\n';
			 } else {
                            yywarn("Preprocessor directive too long");
			 }
                       }

<PP>\n                 {  /* End of this preprocessor logical line */
                         *tokn_ptr = '\0';
                         file_ppln=false;
                         in_preprocess = 0;
                         yyless(0);        /* Hack! */
			 LAST_STATE();
                         lvalp->stemnt =
                           new PPDirective(std::string(tokn_buff,tokn_ptr-tokn_buff),HERE);
                         return(err_tok = PP_DIR);
                       }

<PP>([^\n\\]|\\[^\r\n])+ {
			    int n = yyleng;
			    if (tokn_ptr + n < tokn_end) {
				memcpy(tokn_ptr,yytext,n);
				tokn_ptr += n;
			    } else {
			       yywarn("Preprocessor directive too long");
			    }
                         }

<INITIAL>{alpha}{alphanum}*  {    /* Identifier */

                         lvalp->symbol = new Symbol();
                       
                         lvalp->symbol->name = yytext;

                         if (possibleTag)
                         {
                           /* Check the tag table and insert a  */
                           /* ptr back to the definition.  */
                           SymEntry *se
                              = gProject->Parse_TOS->transUnit->contxt.tags->Lookup(
                                    lvalp->symbol->name);
                           lvalp->symbol->entry = se;
                           possibleTag = false;
                           return(err_tok = TAG_NAME);
                         }
                         else if (possibleLabel)
                         {
                           /* Check the tag table and insert a  */
                           /* ptr back to the definition.  */
                           SymEntry *se
                              = gProject->Parse_TOS->transUnit->contxt.labels->LookupAt(
                                    lvalp->symbol->name,FUNCTION_SCOPE);
                           lvalp->symbol->entry = se;
                           if (!se)
                               gProject->Parse_TOS->transUnit->contxt.labels
                                   ->InsertAt(lvalp->symbol->entry = mk_label(lvalp->symbol->name, NULL), 
                                       FUNCTION_SCOPE);

                           possibleLabel = false;
                           return(err_tok = LABEL_NAME);
                         }
                         else {
                           /* Check the symbol table and insert a  */
                           /* ptr back to the definition.  */
                           SymEntry *se
                              = gProject->Parse_TOS->transUnit->contxt.syms->Lookup(
                                    lvalp->symbol->name);
                           lvalp->symbol->entry = se;
 
                            if (possibleType)
                            {
                              if (se && se->IsTypeDef())
                              {
                                  return(err_tok = TYPEDEF_NAME);
                              }
                            }
                         }

                         return(err_tok = IDENT);
                       }

<INITIAL>{octnum}  |
<INITIAL>{intnum}  |
<INITIAL>{hexnum}   {  /* An integer */
                         long i = strtol(yytext, (char**) NULL, 0);

                         if (errno == ERANGE)
                         {
                            unsigned long ui;
                            ui = strtoul(yytext, (char**) NULL, 0);
                            lvalp->consValue = new UIntConstant(ui,HERE);
                         }
                         else
                         { 
                             lvalp->consValue = new IntConstant(i,HERE);
                         } 

                         return(err_tok = INUM);
                    }
          
<INITIAL>{digits}{dot}{digits}{exponent}?{floatsuffix}? |
<INITIAL>{digits}{dot}{exponent}?{floatsuffix}?         |
<INITIAL>{dot}{digits}{exponent}?{floatsuffix}?         |
<INITIAL>{digits}{exponent}{floatsuffix}? {
                         /*
                         ** Note: The floatsuffix, if any, will be
                         ** ignored by atof().
                         */
                         lvalp->consValue = new FloatConstant(atof(yytext),HERE);
                         return(err_tok = RNUM);
                       }

<STR>\"{allwhite}*\"   {   /* String Pasting */
                         cnt_nl(yytext,yyleng);
                       }

<STR>\"                {   /* Closing quote */
                         LAST_STATE();
                         *tokn_ptr = '\0';
                         lvalp->consValue =
                           new StringConstant(std::string(tokn_buff,tokn_ptr-tokn_buff),HERE,isWide);
                         return(err_tok = STRING);
                       }

<STR>\n                { /* Error - unterminated string constant */
                         yyerr("Unterminated string constant starting");
                         LAST_STATE();
                         return(INVALID);
                       }

<STR>\\[0-7]+          { /* octal escape sequence */
                         uint result;
                         char tmpOct[20];

                         strncpy(tmpOct,yytext,yyleng);
                         tmpOct[yyleng] = '\0';
                         tmpOct[4] = '\0';

                         if (yyleng > 4)
                             yyless(4);

                         if (sscanf(tmpOct+1, "%o", &result ) < 1)
                         {
                             yyerr("Invalid octal escape sequence");
                             LAST_STATE();
                             return(INVALID);
                         }

                             /* error, constant is out-of-bounds */
                         if ( result > 0xff ){
                            yyerr("Escape sequence out-of-bounds");
                         }

                         if (tokn_ptr < tokn_end)
                            *tokn_ptr++ = result;
                         else
                            yywarn("String constant too long");
                       }

<STR>\\x{hexdigit}+    {   /* hex escape sequence - ISO C */
                         uint result;

                         if (sscanf(yytext+2, "%x", &result ) < 1)
                         {
                             yyerr("Invalid hex escape sequence");
                             LAST_STATE();
                             return(INVALID);
                         }

                             /* error, constant is out-of-bounds */
                         if ( result > 0xff ){
                            yyerr("Escape sequence out-of-bounds");
                         } else if (yyleng > 4){
                            yywarn("Hexadecimal escape exceeds two chars");
                         }

                         if (tokn_ptr < tokn_end)
                            *tokn_ptr++ = result;
                         else
                            yywarn("String constant too long");
                       }

<STR>\\{digits}        {  /* Bad escape sequence */
                         yywarn("Bad escape sequence in string");
                        *tokn_ptr++ = '?';
                       }

<STR>\\n               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = '\n';
                         else
                            yywarn("String constant too long");
                       }

<STR>\\t               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = '\t';
                         else
                            yywarn("String constant too long");
                       }

<STR>\\r               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = '\r';
                         else
                            yywarn("String constant too long");
                       }

<STR>\\b               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = '\b';
                         else
                            yywarn("String constant too long");
                       }

<STR>\\f               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = '\f';
                         else
                            yywarn("String constant too long");
                       }

<STR>\\v               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = '\v';
                         else
                            yywarn("String constant too long");
                       }

<STR>\\a               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = '\a';
                         else
                            yywarn("String constant too long");
                       }

<STR>\\e               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = ESC_VAL;
                         else
                            yywarn("String constant too long");
                       }

<STR>\\(\n)            {  /* String continuation */
                         gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 0;
                       }

<STR>\\.               {
                         if (tokn_ptr < tokn_end)
                           *tokn_ptr++ = yytext[1];
                         else
                            yywarn("String constant too long");
                       }

<STR>[^\\\n\"]+        {
                         int n = yyleng;
                         if (tokn_ptr + n < tokn_end) {
                             memcpy(tokn_ptr,yytext,n);
                             tokn_ptr += n;
                         } else 
                            yywarn("String constant too long");
                       }

<CMMT>[^*\n/\\]*       {   /* Inside C-style comment */ }

<CMMT>[^*\n/\\]*\n     {
                         gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 0;
                         in_preprocess = 0;
                       }
<CMMT>"/"[^*\n]        { }
<CMMT>\\\n             {
                         gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 0;
                       }
<CMMT>\\[^\n]          {
                       }
<CMMT>"/"\n            {
                         gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 0;
                         in_preprocess = 0;
                       }
<CMMT>"/*"             { yywarn("/* inside comment"); }
<CMMT>"*"+[^*/\n\\]*   {   /* Stars */ }
<CMMT>"*"+[^*/\n\\]*\n {
                         gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 0;
                         in_preprocess = 0;
                       }
<CMMT>"*"+"/"          {
                         LAST_STATE();
                       }

<CC>\\[0-7]{1,3}"'"    {   /* octal escape sequence */
                         uint result;

                         if (sscanf(yytext+1, "%o", &result ) < 1)
                         {
                             yyerr("Invalid octal escape sequence");
                             LAST_STATE();
                             return(INVALID);
                         }
                         
                             /* error, constant is out-of-bounds */
                         if ( result > 0xff ){
                             yywarn("Escape sequence out-of-bounds");
                             result = 0;
                         }

                         lvalp->consValue = new CharConstant((char) result, HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\x{hexdigit}+"'"  {   /* hex escape sequence - ISO C */
                         uint result;

                         if (sscanf(yytext+2, "%x", &result ) < 1)
                         {
                             yyerr("Invalid hex escape sequence");
                             LAST_STATE();
                             return(INVALID);
                         }

                             /* error, constant is out-of-bounds */
                         if ( result > 0xff ){
                             yywarn("Escape sequence out-of-bounds");
                             result = 0;
                         } else if (yyleng > 5){
                             yywarn("Hexadecimal escape exceeds two chars");
                             result = 0;
                         }

                         lvalp->consValue = new CharConstant((char) result, HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\{digits}"'"      {  /* Bad escape sequence */
                         yywarn("Bad escape sequence in char constant");
                         lvalp->consValue = new CharConstant('\0',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\n                 { /* Error unterminated char constant */
                         gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 0;
                         yywarn("Unterminated char constant");
                         lvalp->consValue = new CharConstant('\0',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\n"'"             {
                         lvalp->consValue = new CharConstant('\n',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\t"'"             {
                         lvalp->consValue = new CharConstant('\t',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\r"'"             {
                         lvalp->consValue = new CharConstant('\r',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\b"'"             {
                         lvalp->consValue = new CharConstant('\b',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\f"'"             {
                         lvalp->consValue = new CharConstant('\f',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\v"'"             {
                         lvalp->consValue = new CharConstant('\v',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\a"'"             {
                         lvalp->consValue = new CharConstant('\a',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\e"'"             {
                         lvalp->consValue = new CharConstant(ESC_VAL,HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\\n"'"            {
                         gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 1;  /* Oddity */
                         lvalp->consValue = new CharConstant('\n',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>\\."'"             {
                         lvalp->consValue = new CharConstant(yytext[1],HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>[^']"'"            {
                         lvalp->consValue = new CharConstant(*yytext,HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>"'"                { /* Empty */
                         yywarn("Empty character constant");
                         lvalp->consValue = new CharConstant('\0',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<CC>[^\\\n][^']+"'"    { /* Multiple characters */
                         yywarn("Invalid character constant");
                         lvalp->consValue = new CharConstant('\0',HERE,isWide);
                         LAST_STATE();
                         return(err_tok = CHAR_CONST);
                       }

<INITIAL>{whitespace}+ {  /* space/tab/formfeed/vertical tab (ignore) */ }

<INITIAL,CMMT>\n|\r    { gProject->Parse_TOS->yylineno++;
                         gProject->Parse_TOS->yynxtcol = 0;
                         in_preprocess = 0;
                       }

<CMMT><<EOF>>          { 
                         yyerr("EOF reached inside comment");
                         in_preprocess = 0;
                         return(0);
                       }

<CC><<EOF>>            { 
                         yyerr("EOF reached inside character constant");
                         return(0);
                       }

<STR><<EOF>>           { 
                         yyerr("EOF reached inside string constant");
                         return(0);
                       }

.                      { /* Any unknown char is an error */
                        *gProject->Parse_TOS->yyerrstream
                           << "Error: Line " << gProject->Parse_TOS->yylineno
                           << ": Illegal Character";
                        if ((yytext[0] >= ' ') && (yytext[0] <= '~'))
                          *gProject->Parse_TOS->yyerrstream
                           << ": '" << yytext[0] << "'\n";
                        else
                          *gProject->Parse_TOS->yyerrstream
                           << ", ASCII: " << std::hex << (int)(yytext[0]) 
                                 << " (hex)\n" << std::dec;
                       }

%%

/************************************************************************
//
// yywrap() - This function is called by the lexer [yylex() or sslex()] when
//            the end-of-file (or end-of-string for sslex()) is reached.
//            It gets the next file/string ready (if any) and returns 1
//            which indicates to the lexer that there are no more files
//            remaining.  The lexer returns 0 to the parser, indicating
//            no more tokens are remaining. The parser function, yyparse(),
//            then returns 0, indicating that the parse is done.  This
//            behaviour allows a single item to be parsed to be placed
//            on the stack and parsed without the entire stack being consumed.
//
// **********************************************************************/

int yywrap(void)
{
    in_preprocess = 0;
    return 1;
}

/***********************************************************************/
int cnt_nl(char *txt, int len)
{
  int ret = 0;

  do {
    switch (*txt++) {
      case '\n':
        ret++;
        gProject->Parse_TOS->yylineno++;
        gProject->Parse_TOS->yynxtcol = 0;
        break;

      default:
        break;
    }
  } while(--len > 0);

  return ret;
}

/***********************************************************************/
/*  Process a #line directive */
int
get_lineno(bool file_ppln, char *txt, Statement **stement)
{
    int   ln = 0;
    *stement = NULL ;
    int   oldyylineno = gProject->Parse_TOS->yylineno;
    do {
        if (txt)
        {
            if (file_ppln)
                ln = 0;
            else
            {
                if (sscanf(txt,"%d", &ln) < 1)
                    break;

                gProject->Parse_TOS->yylineno = ln - 1;
            }
           
            txt = strchr(txt,'"');

            if (txt)
            {
                char *pnt_end;
    
                txt++;

                pnt_end = strchr(txt,'"');
                if (pnt_end)
                    *pnt_end = '\0';
                else
                    break;
                        
                int entered_or_exited;
                if (sscanf(pnt_end+1,"%d", &entered_or_exited) < 1)
                    entered_or_exited = -1;
               
                switch (entered_or_exited)
				{
                  case 2:
                      // new line number into the previous included file
                      gProject->Parse_TOS->yylineno = oldyylineno;
                      *stement = new EndInclStemnt(HERE);
                      if (! file_ppln)
                          gProject->Parse_TOS->yylineno = ln - 1;
                      gProject->Parse_TOS->filename = txt;
                      return (*stement != NULL);
                      
                  case 1:
                      // new line number into a new include file
                      gProject->Parse_TOS->yylineno = oldyylineno;
                      if (! file_ppln)
                          gProject->Parse_TOS->yylineno = ln - 1;
                      gProject->Parse_TOS->filename = txt;
                      *stement = new InclStemnt(gProject->Parse_TOS->filename, HERE);
                      return (*stement != NULL);
                    
                  case 3:
                      // new line number introduced by the gcc preprocessor 
                      // into the same file
                      if (gProject->Parse_TOS->filename == txt && 
                          ln > oldyylineno)
                      {
                          gProject->Parse_TOS->filename = txt;
                          return 0;
                      }
                      /* No break */
                  default:
                      yywarn("Malformed #line directive");
                      /* No break */
                  case -1:
                      // new line number into the same file
                      gProject->Parse_TOS->yylineno = oldyylineno;
                      gProject->Parse_TOS->filename = txt;
                      *stement = new FileLineStemnt(gProject->Parse_TOS->filename, ln, HERE);
                      if (! file_ppln)
                          gProject->Parse_TOS->yylineno = ln - 1;
                      return (*stement != NULL);
                  }
              }
          }

        return 0;
      } while (0);

    yywarn("Malformed #line directive");
    return 0;
}

/***********************************************************************/

void print_caret(const std::string &s, int pos)
{
    const char *str;
    int i = 0;

    for (str=s.c_str(); *str && (i<pos); i++, str++)
        if (*str=='\t')
            *gProject->Parse_TOS->yyerrstream << "\t";
        else
            *gProject->Parse_TOS->yyerrstream << " ";

    *gProject->Parse_TOS->yyerrstream << "^\n";
}
 
/***********************************************************************/

void yywarn(char *s)
{
    *gProject->Parse_TOS->yyerrstream
         << gProject->Parse_TOS->transUnit->filename << ":" << gProject->Parse_TOS->yylineno
         << ": Warning - " << s << ":\n" << line;
    print_caret(line, gProject->Parse_TOS->yycolno);
}

/***********************************************************************/

int yyerr(char *s, std::string & str)
{
    *gProject->Parse_TOS->yyerrstream
      << gProject->Parse_TOS->filename << ":" << gProject->Parse_TOS->yylineno
      << ": " << s << str << ":\n" << line;
    print_caret(line, gProject->Parse_TOS->yycolno);

    ++err_top_level;
    if (++err_cnt >= 10){
        *gProject->Parse_TOS->yyerrstream
          << "Too many errors (" << err_cnt << " detected) - quitting.\n";
        return(1);
    }
    return(0); 
}

/***********************************************************************/

int yyerr(char *s)
{
    *gProject->Parse_TOS->yyerrstream
      << gProject->Parse_TOS->filename << ":" << gProject->Parse_TOS->yylineno
      << ": " << s << ":\n" << line;
    print_caret(line, gProject->Parse_TOS->yycolno);

    ++err_top_level;
    if (++err_cnt >= 10){
        *gProject->Parse_TOS->yyerrstream
          << "Too many errors (" << err_cnt << " detected) - quitting.\n";
        return(1);
    }
    return(0); 
}

/***********************************************************************/

int yyerror(char *s)
{
    *gProject->Parse_TOS->yyerrstream
      << gProject->Parse_TOS->transUnit->filename << ":" << gProject->Parse_TOS->yylineno
      << ": Error (" << s << ") before '"
      << toksym(err_tok,0) << "'\n" << line;
    print_caret(line, gProject->Parse_TOS->yycolno);

    /*
     * This does _NOTHING_.  Bison, at least, ignores the return value of
     * yyerror().  I don't know why cTool does this.  --Jeremy.
     */
    if (++err_cnt >= 10){
        *gProject->Parse_TOS->yyerrstream
          << "Too many errors (" << err_cnt << " detected) - quitting.\n";
        return(1);
    }
    return(0);
}

/***********************************************************************/
/* created 11/6/99 */
void function_to_eliminate_gcc_warning()
{
  yyunput(0,"");
}

/***********************************************************************/
