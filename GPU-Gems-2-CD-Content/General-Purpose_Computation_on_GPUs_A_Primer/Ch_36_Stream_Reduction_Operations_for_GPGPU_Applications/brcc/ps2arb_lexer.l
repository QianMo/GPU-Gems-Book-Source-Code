/* Scanner for ps2.0 Assembly */

%option noyywrap

%{
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <sstream> 
#include "ps2arb.h"

#define YY_INPUT(buf,result,max_size) {            \
  int c = ps2arb_ps20code->get();                  \
  result = (c == EOF) ? YY_NULL : (buf[0] = c, 1); \
}

#define YY_NO_UNPUT 1

#include "ps2arb_gram.h"

int scopecount = 0;
int ps_lineno=0;

char * copy_string (const char * in) {
	return strdup (in);
}

%}
R_PSHEADER          ps("."|"_")[0-9]("."|"_")[0-9a-z]
R_FLOAT             ("+"|"-")?(("."[0-9]+)|([0-9]+"."[0-9]*)|([0-9]+))(e("+"|"-")?[0-9]+)?(f?)
R_NEGATE            "-"
R_SWIZZLEMASK       "."[rgbaw-z]([rgbaw-z]?)([rgbaw-z]?)([rgbaw-z]?)
R_COLORREG          "v"[0-1]
R_TEXCOORDREG       "t"[0-7]
R_OUTPUTREG         ("oC"[0-3])|"oDepth"
R_SAMPLEREG         "s"([0-9]|("1"[0-5]))
R_CONSTREG	        "c"[0-9]([0-9]?)
R_TEMPREG           "r"[0-9]([0-9]?)
R_TEXKILL           "texkill"
R_NOP               "nop"
R_SINCOS            "sincos"
R_UNARY_OP          "abs"|"exp"|"log"|"frc"|"rcp"|"rsq"|"nrm"
R_MOV               "mov"
R_BINARY_OP         "pow"|"add"|"crs"|"dp3"|"dp4"|"mul"|"max"|"min"|("m3x"[2-4])|("m4x"[3-4])|"sub"|"texld"|"texldp"|"texldb"
R_TRINARY_OP        "cmp"|"dp2add"|"lrp"|"mad"
R_OP_FLAGS          "_sat"("_pp"?)|"_pp"("_sat"?)
R_DCLTEX            "dcl_2d"|"dcl_cube"|"dcl_volume"
R_DCL               "dcl"
R_DEF               "def"
R_COMMA             ","
R_NEWLINE           ("\r"|"\n")+
R_WHITESPACE        " "|"\t"
R_COMMENT           ("//"|";")[^\n^\r]*("\n"|"\r")
R_ENDLESS_COMMENT   ("//"|";")[^\n^\r]*
%%
{R_NEGATE}  {
	ps2arb_lval.s="-";
	return PS_NEGATE;
}
{R_SWIZZLEMASK}       {
                          ps2arb_lval.s = copy_string(yytext+1);
                          return PS_SWIZZLEMASK;} 
{R_COLORREG}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_COLORREG;} 
{R_TEMPREG}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_TEMPREG;} 
{R_TEXCOORDREG}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_TEXCOORDREG;} 
{R_OUTPUTREG}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_OUTPUTREG;}
{R_SAMPLEREG}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_SAMPLEREG;} 
{R_CONSTREG}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_CONSTREG;} 
{R_FLOAT}       {
                          ps2arb_lval.f = (float) strtod(yytext,NULL);
                          return PS_FLOAT;}
{R_TEXKILL}       {
                          ps2arb_lval.s = copy_string(yytext);	
                          return PS_TEXKILL;} 
{R_NOP}       {
                          ps2arb_lval.s = copy_string(yytext);	
                          return PS_NOP;} 
{R_SINCOS}       {
                          ps2arb_lval.s = copy_string(yytext);	
                          return PS_SINCOS;}
{R_MOV}       {
                          ps2arb_lval.s = copy_string(yytext);	
                          return PS_MOV;}
{R_UNARY_OP}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_UNARY_OP;}
{R_BINARY_OP}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_BINARY_OP;}
{R_TRINARY_OP}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_TRINARY_OP;}
{R_PSHEADER}       {
			  int a=0,b=0;
                          sscanf(yytext,"ps_%d_%d",&a,&b);
                          sscanf(yytext,"ps.%d.%d",&a,&b);
                          ps2arb_lval.f = a + 0.1f * b;
                          return PS_PSHEADER;
		   }
{R_OP_FLAGS}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_OP_FLAGS;}
{R_DCLTEX}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_DCLTEX;}
{R_DCL}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_DCL;}
{R_DEF}       {
                          ps2arb_lval.s = copy_string(yytext);
                          return PS_DEF;}
{R_COMMA}       {
                          ps2arb_lval.s = ",";
                          return PS_COMMA;}
{R_COMMENT}	{
			char * yyt=yytext;
			while (yyt[0]==';'||yyt[0]=='/'||yyt[0]=='#') yyt++;
			ps_lineno++;
			ps2arb_lval.s = copy_string (yyt);
			unsigned int n=strlen(ps2arb_lval.s);
			if (n)
				ps2arb_lval.s[n-1]=0;
			return PS_COMMENT;
		}
{R_ENDLESS_COMMENT}  {
			char * yyt=yytext;
			while (yyt[0]==';'||yyt[0]=='/'||yyt[0]=='#') yyt++;
			ps2arb_lval.s = copy_string (yyt);
			return PS_ENDLESS_COMMENT;
		}
{R_NEWLINE}	{
			  ps2arb_lval.f=0;
                          int i=0; 
                          while (yytext[i])
                              if (yytext[i++]=='\n') {ps_lineno++;ps2arb_lval.f++;}
			  return PS_NEWLINE;
		}
{R_WHITESPACE}       

.                   {printf ("error %s",yytext);return yytext[0];}

%%

