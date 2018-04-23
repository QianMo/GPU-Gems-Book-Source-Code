%{
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
%}

/* The next line makes the parser re-entrant. */
%pure_parser

%start program

%token <symbol>     IDENT TAG_NAME LABEL_NAME TYPEDEF_NAME

%token <consValue>  STRING LSTRING
%token <consValue>  CHAR_CONST LCHAR_CONST
%token <consValue>  INUM RNUM
%token <stemnt>     PP_DIR PP_LINE

%token <loc>        INVALID

/* the reserved words */
%token <typeQual>   CONST VOLATILE OUT REDUCE VOUT ITER KERNEL
%token <storage>    AUTO EXTRN REGISTR STATIC TYPEDEF
%token <base>       VOID CHAR SHORT INT LONG SGNED UNSGNED
/* IMPORTANT: Keep all the FLOATN's next to each other in order! */
%token <base>       FLOAT FLOAT2 FLOAT3 FLOAT4 FIXED FIXED2 FIXED3 FIXED4 HALF HALF2 HALF3 HALF4 DOUBLE DOUBLE2
%token <typeSpec>   ENUM STRUCT UNION 

%token <loc>        BREAK CASE CONT DEFLT DO ELSE
%token <loc>        IF
%token <loc>        FOR
%token <loc>        GOTO RETURN SWITCH WHILE 

%token <loc>        PLUS_EQ MINUS_EQ STAR_EQ DIV_EQ MOD_EQ
%token <loc>        B_AND_EQ B_OR_EQ B_XOR_EQ
%token <loc>        L_SHIFT_EQ R_SHIFT_EQ
%token <loc>        EQUAL LESS_EQ GRTR_EQ NOT_EQ 
 
%token <loc>        RPAREN RBRCKT LBRACE RBRACE
%token <loc>        SEMICOLON COMMA ELLIPSIS
	
%token <loc>        LB_SIGN DOUB_LB_SIGN
%token <loc>        BACKQUOTE AT

/* Gcc Extensions */
%token              ATTRIBUTE ALIGNED PACKED CDECL MODE FORMAT NORETURN

/* Add precedence rules to solve dangling else s/r conflict */
%nonassoc IF
%nonassoc ELSE

/* Define the operator tokens and their precedences. */
%left               COMMA_OP
%right <assignOp>   EQ ASSIGN
%right <loc>        QUESTMARK COLON COMMA_SEP
%left  <loc>        OR
%left  <loc>        AND
%left  <loc>        B_OR
%left  <loc>        B_XOR
%left  <loc>        B_AND
%left  <relOp>      COMP_EQ
%left  <relOp>      COMP_ARITH COMP_LESS COMP_GRTR LESS GRTR
%left  <binOp>      L_SHIFT R_SHIFT
%left  <binOp>      PLUS MINUS
%left  <binOp>      STAR DIV MOD
%right              CAST
%right <loc>        UNARY NOT B_NOT SIZEOF INDEXOF INCR DECR  
%left               HYPERUNARY
%left  <loc>        ARROW DOT LPAREN LBRCKT

%type  <symbol>      ident
%type  <symbol>      typename_as_ident any_ident
%type  <symbol>      tag_ref
%type  <symbol>      field_ident

%type  <storage>     storage_class local_storage_class local_or_global_storage_class
%type  <typeQual>    type_qual type_qual_token type_qual_list opt_type_qual_list

%type  <base>        type_spec type_spec_reentrance 
%type  <base>        enum_type_define enum_tag_ref enum_tag_def
%type  <base>        struct_type_define struct_tag_ref struct_tag_def
%type  <base>        union_type_define union_tag_ref union_tag_def
%type  <base>        typedef_name
%type  <type>        type_name type_name_bis

%type  <ptr>         pointer pointer_reentrance pointer_start
%type  <type>        abs_decl abs_decl_reentrance
%type  <type>        direct_abs_decl_reentrance_bis direct_abs_decl_reentrance

%type  <decl>        comp_decl comp_decl_list comp_decl_list_reentrance
%type  <strDef>      struct_or_union_definition field_list field_list_reentrance
%type  <decl>        array_decl stream_decl
%type  <decl>        direct_declarator_reentrance_bis direct_declarator_reentrance
%type  <decl>        func_declarator declarator declarator_reentrance_bis
%type  <decl>        opt_declarator comp_declarator
%type  <decl>        simple_comp bit_field
%type  <decl>        init_decl decl

%type  <value>       opt_const_expr const_expr expr opt_expr
%type  <value>       comma_constants comma_expr assign_expr dimension_constraint
%type  <value>       prim_expr paren_expr postfix_expr
%type  <value>       subscript_expr comp_select_expr postinc_expr postdec_expr
%type  <value>       direct_comp_select indirect_comp_select

%type  <value>       log_or_expr log_and_expr log_neg_expr
%type  <value>       bitwise_or_expr bitwise_and_expr bitwise_neg_expr
%type  <value>       bitwise_xor_expr cast_expr equality_expr
%type  <value>       relational_expr shift_expr additive_expr mult_expr
%type  <value>       unary_expr unary_minus_expr unary_plus_expr
%type  <value>       sizeof_expr indexof_expr addr_expr indirection_expr
%type  <value>       preinc_expr predec_expr constructor_expr
%type  <value>       iter_constructor_expr iter_constructor_arg

%type  <value>       func_call cond_expr
%type  <value>       opt_expr_list expr_list
%type  <value>       initializer initializer_reentrance width

%type  <label>       label named_label case_label deflt_label
%type  <decl>        opt_init_decl_list init_decl_list init_decl_list_reentrance

%type  <decl>        func_spec
%type  <functionDef> func_def 
%type  <stemnt>      top_level_decl cmpnd_stemnt cmpnd_stemnt_reentrance
%type  <stemnt>      stemnt_list stemnt_list_reentrance
%type  <stemnt>      stemnt_list2 stemnt_list_reentrance2
%type  <stemnt>      constructor_stemnt non_constructor_stemnt
%type  <stemnt>      opt_stemnt_list opt_stemnt_list_reentrance 
%type  <stemnt>      stemnt stemnt_reentrance
%type  <stemnt>      expr_stemnt labeled_stemnt cond_stemnt
%type  <stemnt>      iter_stemnt switch_stemnt break_stemnt continue_stemnt
%type  <stemnt>      return_stemnt goto_stemnt null_stemnt
%type  <stemnt>      do_stemnt while_stemnt for_stemnt
%type  <stemnt>      if_stemnt if_else_stemnt

%type  <consValue>   constant

%type  <symbol>      enum_constant
%type  <enConst>     enum_const_def
%type  <enDef>       enum_definition enum_def_list enum_def_list_reentrance

%type  <declStemnt>  decl_stemnt
%type  <decl>        old_style_declaration declaration
%type  <decl>        declaration_list opt_declaration_list
%type  <decl>        opt_KnR_declaration_list

%type  <base>        no_decl_specs decl_specs decl_specs_reentrance_bis
%type  <base>        decl_specs_reentrance opt_decl_specs_reentrance
%type  <base>        comp_decl_specs comp_decl_specs_reentrance opt_comp_decl_specs

%type  <decl>        param_list param_decl param_decl_bis
%type  <decl>        opt_param_type_list param_type_list param_type_list_bis
%type  <decl>        ident_list ident_list_reentrance

%type  <arrayConst>  initializer_list

%type  <loc>         opt_comma
%type  <loc>         opt_trailing_comma

%type  <binOp>       add_op mult_op shift_op
%type  <relOp>       relation_op equality_op
%type  <assignOp>    assign_op

%type  <transunit>   program trans_unit

%type  <gccAttrib>   gcc_attrib opt_gcc_attrib gcc_inner

%{
/* 1 if we explained undeclared var errors.  */
/*  static int undeclared_variable_notice = 0;  */
%}

%%
/*** INPUT RULE OF THIS SET:
                   program (STARTING RULE)
     CALLED INPUTS: decl_stemnt, cmpnd_stemnt, decl_specs, declarator,
                   opt_KnR_declaration_list
  NO REENTRANCE ***/
program:  /* emtpy source file */ 
        {
            if (err_cnt == 0)
              *gProject->Parse_TOS->yyerrstream
              << "Warning: ANSI/ISO C forbids an empty source file.\n";
            gProject->Parse_TOS->transUnit = (TransUnit*) NULL;
            $$ = (TransUnit*) NULL;
        }
       | trans_unit
        {
            if (err_cnt)
            {
                *gProject->Parse_TOS->yyerrstream
                << err_cnt << " errors found.\n";
                gProject->Parse_TOS->transUnit = (TransUnit*) NULL;
            } else {
                gProject->Parse_TOS->transUnit = $$;
            }
        }
       | error
        {
            *gProject->Parse_TOS->yyerrstream << "Errors - Aborting parse.\n";
            gProject->Parse_TOS->transUnit = (TransUnit*) NULL;
            YYACCEPT;
        }
        ;

trans_unit:  top_level_decl top_level_exit 
        {
            $$ = gProject->Parse_TOS->transUnit;
            $$->add($1);
        }
          |  trans_unit top_level_decl top_level_exit
        {
            $$->add($2);
        }
        ;

top_level_exit: /* Safety precaution! */
        {
            gProject->Parse_TOS->parseCtxt->ReinitializeCtxt();
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScopes(FILE_SCOPE);
            err_top_level = 0;            
        }
        ;
        
top_level_decl: decl_stemnt
        {
            $$ = $1;
        }
              | func_def
        {
            $$ = $1;
        }
              | PP_DIR
        {
            $$ = $1;
        }
              | PP_LINE
        {
            $$ = $1;
        }
              | error SEMICOLON
        {
            $$ = (Statement*) NULL;
        }
              | error RBRACE top_level_exit
        {
            $$ = (Statement*) NULL;
        }
        ;

func_def:  func_spec cmpnd_stemnt
        {
            if ($2 != NULL)
            {
                $$ = new FunctionDef($2->location);
                Block *blk = (Block*) $2;
    
                $$->decl = $1;
                
                if ($1->name &&
                    $1->name->entry)
                    $1->name->entry->u2FunctionDef = $$;
                
                // Steal internals of the compound statement
                $$->head = blk->head;
                $$->tail = blk->tail;
    
                blk->head = blk->tail = (Statement*) NULL;
                delete $2;    
            }
			else
			{
				delete $1;
				$$ = (FunctionDef*) NULL;
			}
        }
        ;
        
func_spec:  decl_specs func_declarator opt_KnR_declaration_list
        {
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
            
            possibleType = true;
            $$ = $2;

            if ($$->form != NULL)
            {
                assert(err_top_level ||
                       $$->form->type == TT_Function );
    
                $$->extend($1);
    
                /* This is adding K&R-style declarations if $3 exists */
                if ($3 != NULL)
                {
                    FunctionType *fnc = (FunctionType*) ($$->form);
                    fnc->KnR_decl = true;
                    Decl *param = $3;
                    while (param != NULL)
                    {
                        Decl *next= param->next;
                            delete param ;
                        param = next;
                    }
                }
            }
        }
         |  no_decl_specs declarator opt_KnR_declaration_list
        {

            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
            
            $$ = $2;

            if ($$->form != NULL)
            {
                assert(err_top_level ||
                       $$->form->type == TT_Function );
                $$->extend($1);
    
                /* This is adding K&R-style declarations if $3 exists */
                if ($3 != NULL)
                {
                    FunctionType *fnc = (FunctionType*) ($$->form);
                    fnc->KnR_decl = true;
                    Decl *param = $3;
                    while (param != NULL)
                    {
                        Decl *next= param->next;
                            delete param ;
                        param = next;
                    }
                }
            }
        }
        ;        
/*** INPUT RULES OF THIS SET:
                   cmpnd_stemnt (CALLING INPUTS: program) 
     CALLED INPUTS: opt_declaration_list, stemnt
  NO REENTRANCE ***/
cmpnd_stemnt:  LBRACE 
        {  
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ReEnterScope();
        }
            opt_declaration_list opt_stemnt_list RBRACE
        {
            Block*    block = new Block(*$1);
            $$ = block;
            block->addDecls($3);
            block->addStatements(ReverseList($4));
            if (gProject->Parse_TOS->transUnit)
            {
                yyCheckLabelsDefinition(gProject->Parse_TOS->transUnit->contxt.labels);
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
            }
        }
            |  error RBRACE
        {
            $$ = (Statement*) NULL;
        }
        ;
        
opt_stemnt_list:  /* Nothing */
        {
            $$ = (Statement*) NULL;
        }
               |  stemnt_list
        ;
        
stemnt_list: non_constructor_stemnt stemnt_list2
        {
	    /*
	     * All the statements are expected in a reversed list (because
	     * of how we parse stemnt_list2) so we need to take the
	     * non_constructor statement at the end.
	     */
            if ($2)
            {
	        Statement *s;

		for (s = $2; s->next; s = s->next) /* Traverse to the end */;
		s->next = $1;
                $$ = $2;
            }
        }
	;

stemnt_list2: /* Empty */
	{
	   $$ = (Statement *) NULL;
	}
            | stemnt_list2 stemnt
        {
            /* Hook them up backwards, we'll reverse them later. */
            if ($2)
            {
                $2->next = $1;
                $$ = $2;
            }
        }
          | stemnt_list2 PP_LINE
        {    /* preprocessor #line directive */
            /* Hook them up backwards, we'll reverse them later. */
            if ($2)
            {
                $2->next = $1;
                $$ = $2;
            }
        }
      ;

/*** INPUT RULES OF THIS SET:
                   stemnt (CALLING INPUTS: cmpnd_stemnt)
     CALLED INPUTS: expr, opt_expr, const_expr, any_ident, ident, 
                   opt_declaration_list
     NO REENTRANCE ***/     
stemnt: stemnt_reentrance
      ;
      
cmpnd_stemnt_reentrance:  LBRACE opt_declaration_list opt_stemnt_list_reentrance RBRACE
        {
            Block*    block = new Block(*$1);
            $$ = block;
            block->addDecls($2);
            block->addStatements(ReverseList($3));
            
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
        }
            |  error RBRACE
        {
            $$ = (Statement*) NULL;
        }
        ;   
            
opt_stemnt_list_reentrance:  /* Nothing */
        {
            $$ = (Statement*) NULL;
        }
               |  stemnt_list_reentrance
        ;
        
stemnt_list_reentrance: non_constructor_stemnt stemnt_list_reentrance2
        {
	    /*
	     * All the statements are expected in a reversed list (because
	     * of how we parse stemnt_list_reentrance2) so we need to take
	     * the non_constructor statement at the end.
	     */
            if ($2)
            {
	        Statement *s;

		for (s = $2; s->next; s = s->next) /* Traverse to the end */;
		s->next = $1;
                $$ = $2;
            }
        }
      ;

stemnt_list_reentrance2: /* Empty */
	{
	   $$ = (Statement *) NULL;
	}
          | stemnt_list_reentrance2 stemnt_reentrance
        {
            /* Hook them up backwards, we'll reverse them later. */
            if ($2)
            {
                $2->next = $1;
                $$ = $2;
            }
        }
          | stemnt_list_reentrance2 PP_LINE
        {    /* preprocessor #line directive */
            /* Hook them up backwards, we'll reverse them later. */
            if ($2)
            {
                $2->next = $1;
                $$ = $2;
            }
        }
      ;
      
stemnt_reentrance:  non_constructor_stemnt
	 {
	    $$ = $1;
	 }
		 |  constructor_stemnt
	 {
	    $$ = $1;
	 }
      ;

non_constructor_stemnt:  expr_stemnt
      |  labeled_stemnt
      |  cmpnd_stemnt_reentrance
      |  cond_stemnt
      |  iter_stemnt
      |  switch_stemnt
      |  break_stemnt
      |  continue_stemnt
      |  return_stemnt
      |  goto_stemnt
      |  null_stemnt
      |  error SEMICOLON
        {
            delete $2;
            $$ = (Statement*) NULL;
        }
      ;

/*
 * A constructor statement is just like any other expression statement.  The
 * only reason we parse it separately is to avoid the ambiguity of a
 * constructor statement at the top of a statement list.  This code:
 *
 *	{
 *	   variable declarations ...
 *	   float2(b);
 *	   statements
 *	}
 *
 * Can parse "float2(b)" as either a variable declaration or a statement.
 * We impose the constraint that the above is _always_ a variable
 * declaration by insisting the first statement of a statement list cannot
 * be a constructor.
 */
constructor_stemnt: constructor_expr SEMICOLON
	{
            $$ = new ExpressionStemnt($1,*$2);
            delete $2;
	}
      ;

expr_stemnt:  expr SEMICOLON
        {
            $$ = new ExpressionStemnt($1,*$2);
            delete $2;
        }
      ;

labeled_stemnt: label COLON stemnt_reentrance
        {
            $$ = $3;
            if ($$ == NULL)
            {
              /* Sorry, we must have a statement here. */
              yyerr("Can't have a label at the end of a block! ");
              $$ = new Statement(ST_NullStemnt,*$2);
            }
            $$->addHeadLabel($1);
            delete $2;
        }
        ;

cond_stemnt:  if_stemnt
           |  if_else_stemnt
        ;

iter_stemnt:  do_stemnt
           |  while_stemnt
           |  for_stemnt
        ;

switch_stemnt: SWITCH LPAREN expr RPAREN stemnt_reentrance
        {
            $$ = new SwitchStemnt($3,$5,*$1);
            delete $1;
            delete $2;
            delete $4;
        }
        ;

break_stemnt: BREAK SEMICOLON
        {
            $$ = new Statement(ST_BreakStemnt,*$1);
            delete $1;
            delete $2;
        }
        ;

continue_stemnt: CONT SEMICOLON
        {
            $$ = new Statement(ST_ContinueStemnt,*$1);
            delete $1;
            delete $2;
        }
        ;

return_stemnt:  RETURN opt_expr SEMICOLON
        {
            $$ = new ReturnStemnt($2,*$1);
            delete $1;
            delete $3;
        }
        ;

goto_stemnt:  GOTO LABEL_NAME SEMICOLON
        {
            $$ = new GotoStemnt($2,*$1);
            delete $1;
            delete $3;
        }
        ;

null_stemnt:  SEMICOLON
        {
            $$ = new Statement(ST_NullStemnt,*$1);
            delete $1;
        }
        ;

if_stemnt:  IF LPAREN expr RPAREN stemnt_reentrance    %prec IF
        {
            $$ = new IfStemnt($3,$5,*$1);
            delete $1;
            delete $2;
            delete $4;
        }
        ;

if_else_stemnt:  IF LPAREN expr RPAREN stemnt_reentrance ELSE stemnt_reentrance
        {
            $$ = new IfStemnt($3,$5,*$1,$7);
            delete $1;
            delete $2;
            delete $4;
            delete $6;
        }
        ;

do_stemnt:  DO stemnt_reentrance WHILE LPAREN expr RPAREN SEMICOLON
        {
            $$ = new DoWhileStemnt($5,$2,*$1);
            delete $1;
            delete $3;
            delete $4;
            delete $6;
            delete $7;
        }
        ;

while_stemnt:  WHILE LPAREN expr RPAREN stemnt_reentrance
        {
            $$ = new WhileStemnt($3,$5,*$1);
            delete $1;
            delete $2;
            delete $4;
        }
        ;

for_stemnt: FOR LPAREN opt_expr SEMICOLON
                       opt_expr SEMICOLON opt_expr RPAREN stemnt_reentrance
        {
            $$ = new ForStemnt($3,$5,$7,*$1,$9);
            delete $1;
            delete $2;
            delete $4;
            delete $6;
            delete $8;
        }
        ;

label:  named_label
     |  case_label
     |  deflt_label
        ;

named_label:  ident
        {
            if (gProject->Parse_TOS->transUnit)
                $$ = gProject->Parse_TOS->parseCtxt->Mk_named_label($1,
                                gProject->Parse_TOS->transUnit->contxt.labels);
        }
        ;

case_label:  CASE const_expr
        {
            $$ = new Label($2);
            delete $1;
        }
          |  CASE const_expr ELLIPSIS const_expr
        {
            $$ = new Label($2,$4);
            delete $1;
            delete $3;
        }
        ;

deflt_label:  DEFLT
        {
            $$ = new Label(LT_Default);
            delete $1;
        }
        ;
/*** INPUT RULES OF THIS SET:
                opt_expr, 
                expr, 
                opt_const_expr, 
                const_expr, 
                cond_expr, 
                assign_expr
     THIS INPUT SET HAS DIRECT REENTRANCE
     CALLED INPUTS: type_name_bis, ident, any_ident
     ***/
cond_expr:  log_or_expr
         |  log_or_expr QUESTMARK expr COLON cond_expr
        {
            $$ = new TrinaryExpr($1,$3,$5,*$2);
            delete $2;
            delete $4;
        }
        ;

assign_expr:  cond_expr
           |  unary_expr assign_op assign_expr
        {
            $$ = new AssignExpr($2,$1,$3,NoLocation);
        }
           |  unary_expr assign_op constructor_expr
        {
            $$ = new AssignExpr($2,$1,$3,NoLocation);
        }
        ;

opt_const_expr:    /* Nothing */
        {
            $$ = (Expression*) NULL;
        }
              | const_expr
        ;

const_expr: expr
        ;

opt_expr:  /* Nothing */
        {
           $$ = (Expression*) NULL;
        }
        |  expr
        ;

expr:    comma_expr
        ;

log_or_expr:  log_and_expr
           |  log_or_expr OR log_and_expr
        {
            $$ = new BinaryExpr(BO_Or,$1,$3,*$2);
            delete $2;
        }
        ;

log_and_expr:  bitwise_or_expr
            |  log_and_expr AND bitwise_or_expr
        {
            $$ = new BinaryExpr(BO_And,$1,$3,*$2);
            delete $2;
        }
        ;

log_neg_expr:  NOT cast_expr
        {
            $$ = new UnaryExpr(UO_Not,$2,*$1);
            delete $1;
        }
        ;

bitwise_or_expr:  bitwise_xor_expr
               |  bitwise_or_expr B_OR bitwise_xor_expr
        {
            $$ = new BinaryExpr(BO_BitOr,$1,$3,*$2);
            delete $2;
        }
        ;

bitwise_xor_expr:  bitwise_and_expr
                |  bitwise_xor_expr B_XOR bitwise_and_expr
        {
            $$ = new BinaryExpr(BO_BitXor,$1,$3,*$2);
            delete $2;
        }
        ;

bitwise_and_expr:  equality_expr
                |  bitwise_and_expr B_AND equality_expr
        {
            $$ = new BinaryExpr(BO_BitAnd,$1,$3,*$2);
            delete $2;
        }
        ;

bitwise_neg_expr:  B_NOT cast_expr
        {
            $$ = new UnaryExpr(UO_BitNot,$2,*$1);
            delete $1;
        }
        ;

cast_expr:  unary_expr
         |  LPAREN type_name_bis RPAREN cast_expr     %prec CAST
        {
            $$ = new CastExpr($2,$4,*$1);
            delete $1;
            delete $3;
        }
        ;

equality_expr:  relational_expr
             |  equality_expr equality_op relational_expr
        {
            $$ = new RelExpr($2,$1,$3,NoLocation);
        }
        ;

relational_expr:  shift_expr
               |  relational_expr relation_op shift_expr
        {
            $$ = new RelExpr($2,$1,$3,NoLocation);
        }
        ;

shift_expr:  additive_expr
          |  shift_expr shift_op additive_expr
        {
            $$ = new BinaryExpr($2,$1,$3,NoLocation);
        }
        ;

additive_expr:  mult_expr
             |  additive_expr add_op mult_expr
        {
            $$ = new BinaryExpr($2,$1,$3,NoLocation);
        }
        ;

mult_expr:  cast_expr
         |  mult_expr mult_op cast_expr
        {
            $$ = new BinaryExpr($2,$1,$3,NoLocation);
        }
        ;

constructor_expr: FLOAT2 LPAREN assign_expr COMMA assign_expr RPAREN
        {
	    Expression *exprs[] = { $3, $5 };
            $$ = new ConstructorExpr($1, exprs, NoLocation);
            delete $2;
            delete $4;
            delete $6;
        }
                | FLOAT3 LPAREN assign_expr COMMA
		                assign_expr COMMA assign_expr RPAREN
        {
	    Expression *exprs[] = { $3, $5, $7 };
            $$ = new ConstructorExpr($1, exprs, NoLocation);
            delete $2;
            delete $4;
            delete $6;
            delete $8;
        }
                | FLOAT4 LPAREN assign_expr COMMA assign_expr COMMA
		                assign_expr COMMA assign_expr RPAREN
        {
	    Expression *exprs[] = { $3, $5, $7, $9 };
            $$ = new ConstructorExpr($1, exprs, NoLocation);
            delete $2;
            delete $4;
            delete $6;
            delete $8;
            delete $10;
        }
                | FLOAT2 LPAREN error RPAREN
	{
	   $$ = (Expression *) NULL;
	}
                | FLOAT3 LPAREN error RPAREN
	{
	   $$ = (Expression *) NULL;
	}
                | FLOAT4 LPAREN error RPAREN
	{
	   $$ = (Expression *) NULL;
	}
	;

iter_constructor_arg: assign_expr
		    | constructor_expr
	;

iter_constructor_expr: ITER LPAREN iter_constructor_arg COMMA
				   iter_constructor_arg RPAREN
	{
	   Symbol *sym = new Symbol();
	   Variable *var;

	   sym->name = strdup("iter");
	   var = new Variable(sym, *$2);
	   $$ = new FunctionCall(var, *$2);

	   ((FunctionCall *) $$)->addArg($3);
	   ((FunctionCall *) $$)->addArg($5);

           delete $2;
           delete $4;
           delete $6;
	}
		     | ITER LPAREN error RPAREN
        {
	   $$ = (Expression *) NULL;
	}
	;

unary_expr:  postfix_expr
          |  sizeof_expr
          |  unary_minus_expr
          |  unary_plus_expr
          |  log_neg_expr
          |  bitwise_neg_expr
          |  addr_expr
          |  indirection_expr
          |  preinc_expr
          |  predec_expr
        ;

sizeof_expr:  SIZEOF LPAREN type_name RPAREN   %prec HYPERUNARY
        {
            $$ = new SizeofExpr($3,*$1);
            delete $1;
            delete $2;
            delete $4;
        }
           |  SIZEOF unary_expr 
        {
            $$ = new SizeofExpr($2,*$1);
            delete $1;
        }
        ;

indexof_expr: INDEXOF ident
	{
	  $$ = new BrtIndexofExpr(new Variable($2,*$1),*$1);
	}
        |  INDEXOF LPAREN ident RPAREN
	{
	  $$ = new BrtIndexofExpr(new Variable($3,*$1),*$1);
	}
        ;

/*
            if ($2->etype==ET_Variable) {
	        $$ = new BrtIndexofExpr((Variable *)$2,*$1);
	    } else {
               err_cnt++;
               (*$1).printLocation(*gProject->Parse_TOS->yyerrstream);
               *gProject->Parse_TOS->yyerrstream
	       << "Error: Indexof must operate on an identifier\n";

               $$ = new BrtIndexofExpr(new Variable(new Symbol,*$1),*$1);
            }
*/


unary_minus_expr:  MINUS cast_expr    %prec UNARY
        {
            $$ = new UnaryExpr(UO_Minus,$2,NoLocation);
        }
        ;

unary_plus_expr:  PLUS cast_expr      %prec UNARY
        {
            /* Unary plus is an ISO addition (for symmetry) - ignore it */
            $$ = $2;
        }
        ;

addr_expr:  B_AND cast_expr             %prec UNARY
        {
            $$ = new UnaryExpr(UO_AddrOf,$2,*$1);
            delete $1;
        }
        ;

indirection_expr:  STAR cast_expr     %prec UNARY
        {
            $$ = new UnaryExpr(UO_Deref,$2,NoLocation);
        }
        ;

preinc_expr:  INCR unary_expr
        {
            $$ = new UnaryExpr(UO_PreInc,$2,*$1);
            delete $1;
        }
        ;

predec_expr:  DECR unary_expr
        {
            $$ = new UnaryExpr(UO_PreDec,$2,*$1);
            delete $1;
        }
        ;

comma_expr:  assign_expr
          |  comma_expr COMMA assign_expr    %prec COMMA_OP
        {
            $$ = new BinaryExpr(BO_Comma,$1,$3,*$2);
            delete $2;
        }
        ;

prim_expr:  ident
        {
            $$ = new Variable($1,NoLocation);
        }
         |  paren_expr
         |  constant
        {
            $$ = $1;
        }
        ;

paren_expr: LPAREN expr RPAREN
        {
            $$ = $2;
            delete $1;
            delete $3;
        }
          | LPAREN error RPAREN
        {
            $$ = (Expression*) NULL;
            delete $1;
            delete $3;
        }
        ;

postfix_expr: prim_expr
            | subscript_expr
            | comp_select_expr
            | func_call
            | postinc_expr
            | postdec_expr
            | indexof_expr
        ;

subscript_expr: 
        postfix_expr LBRCKT expr RBRCKT
        {
            $$ = new IndexExpr($1,$3,*$2);
            delete $2;
            delete $4;
        }
        ;

comp_select_expr: direct_comp_select
                | indirect_comp_select
        ;

postinc_expr: postfix_expr INCR
        {
            $$ = new UnaryExpr(UO_PostInc,$1,*$2);
            delete $2;
        }
        ;

postdec_expr: postfix_expr DECR
        {
            $$ = new UnaryExpr(UO_PostDec,$1,*$2);
            delete $2;
        }
        ;

field_ident: any_ident
        ;
        
direct_comp_select: 
        postfix_expr DOT field_ident
        {
            Variable *var = new Variable($3,*$2);
            BinaryExpr *be = new BinaryExpr(BO_Member,$1,var,*$2);
            delete $2;
            $$ = be;

            // Lookup the component in its struct
            // if possible.
            if ($1->etype == ET_Variable)
            {
                Variable  *var = (Variable*) $1;
                Symbol    *varName = var->name;
                SymEntry  *entry = varName->entry;

                if (entry && entry->uVarDecl)
                {
                    entry->uVarDecl->lookup($3);
                }
            }
        }
        ;
        
indirect_comp_select: postfix_expr ARROW field_ident
        {
            Variable *var = new Variable($3,*$2);
            BinaryExpr *be = new BinaryExpr(BO_PtrMember,$1,var,*$2);
            delete $2;
            $$ = be;

            // Lookup the component in its struct
            // if possible.
            if ($1->etype == ET_Variable)
            {
                Variable  *var = (Variable*) $1;
                Symbol    *varName = var->name;
                SymEntry  *entry = varName->entry;

                if (entry && entry->uVarDecl)
                {
                    entry->uVarDecl->lookup($3);
                }
            }
        }
        ;
        
func_call: postfix_expr LPAREN opt_expr_list RPAREN
        {
            FunctionCall* fc = new FunctionCall($1,*$2);

            /* add function args */
            fc->addArgs(ReverseList($3));

            delete $2;
            delete $4;
            $$ = fc;
        }
        ;  
             
opt_expr_list:  /* Nothing */
        {
            $$ = (Expression*) NULL;
        }
             | expr_list
        ;

expr_list:  assign_expr
	 |  constructor_expr
         |  expr_list COMMA assign_expr    %prec COMMA_SEP
        {
            $$ = $3;
            $$->next = $1;

            delete $2;
        }
         |  expr_list COMMA constructor_expr    %prec COMMA_SEP
        {
            $$ = $3;
            $$->next = $1;

            delete $2;
        }
        ;

add_op:  PLUS
      |  MINUS
        ;

mult_op:  STAR
       |  DIV
       |  MOD
        ;

equality_op:  COMP_EQ
        ;

relation_op:  COMP_ARITH
           |  COMP_LESS
           |  COMP_GRTR
        ;

shift_op:  L_SHIFT
        |  R_SHIFT
        ;
        
assign_op:  EQ
         |  ASSIGN
        ;
        
constant:   INUM
          | RNUM
          | CHAR_CONST
          | LCHAR_CONST
          | STRING
          | LSTRING
        ;        
/*** INPUT RULES OF THIS SET:
                   decl_stemnt          (CALLING INPUTS: program) 
                   opt_KnR_declaration_list (CALLING INPUTS: program) 
                   opt_declaration_list (CALLING INPUTS: cmpnd_stemnt, stemnt) 
     CALLED INPUTS: decl_specs, opt_init_decl_list 
  NO REENTRANCE ***/
opt_KnR_declaration_list:  
        {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ReEnterScope();
        }
                      /* Nothing */
        {
            $$ = (Decl*) NULL;
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
        }
                        |
        {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ReEnterScope();
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
        }
        {   gProject->Parse_TOS->parseCtxt->SetVarParam(1, !err_top_level, 0); 
            gProject->Parse_TOS->parseCtxt->SetIsKnR(true); 
        }
                           declaration_list
        {   $$ = $3;
            gProject->Parse_TOS->parseCtxt->SetIsKnR(false); 
            gProject->Parse_TOS->parseCtxt->SetVarParam(0, !err_top_level, 1); 
            
            // Exit, but will allow re-enter for a function.
            // Hack, to handle parameters being in the function's scope.
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope(true);
        }
        ;

opt_declaration_list:
        {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
        }
                      /* Nothing */
        {
            $$ = (Decl*) NULL;
        }
                    |
        {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
        }
        {   gProject->Parse_TOS->parseCtxt->SetVarParam(0, !err_top_level, 0); 
        }
                       declaration_list
        {   $$ = $3;
            gProject->Parse_TOS->parseCtxt->SetVarParam(0, !err_top_level, 0);
        }
        ;

declaration_list:  declaration SEMICOLON
        {
            $$ = $1;
            delete $2;
        }
                |  declaration SEMICOLON declaration_list
        {
            $$ = $1;

			Decl*	appendDecl = $1;
			while (appendDecl->next != NULL)
				appendDecl = appendDecl->next;

            appendDecl->next = $3;
            delete $2;
        }
        ;

decl_stemnt:  old_style_declaration SEMICOLON
        {
            $$ = new DeclStemnt(*$2);
            $$->addDecls(ReverseList($1));
            delete $2;
        }
           |
              declaration SEMICOLON
        {
            $$ = new DeclStemnt(*$2);
            $$->addDecls(ReverseList($1));
            delete $2;
        }
        ;

old_style_declaration: no_decl_specs opt_init_decl_list
        {
            assert (err_top_level ||
                    $1 == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();
            
            yywarn("old-style declaration or incorrect type");

            possibleType = true;
            $$ = $2;

            if ($$ == NULL)
            {
                $$ = new Decl($1);
            }
        }
        ;
        
declaration:  decl_specs opt_init_decl_list
        {
            assert (1||err_top_level ||
                    $1 == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            if ($1!=gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs) {
              if (!err_top_level) {
                baseTypeFixup($1,$2);
              }
            }
            gProject->Parse_TOS->parseCtxt->ResetDeclCtxt();            
            
            possibleType = true;
            $$ = $2;
            
            if ($$ == NULL)
            {
                $$ = new Decl($1);
            }
        }
        ;
/*** INPUT RULES OF THIS SET:
                  no_decl_specs (CALLING INPUTS: program, decl_stemnt)
                  decl_specs (CALLING INPUTS: program, opt_KnR_declaration_list, decl_stemnt)
     CALLED INPUTS: decl_specs_reentrance_bis
  NO REENTRANCE ***/
no_decl_specs: /* Nothing: returns "int" as default decl_specs */
        {
            $$ = new BaseType(BT_Int);
            gProject->Parse_TOS->parseCtxt->SetDeclCtxt($$);
        }
        ;
        
decl_specs: decl_specs_reentrance_bis 
        ;         
/*** INPUT RULES OF THIS SET
                   type_name: (CALLING INPUTS: EXPR SET)
     CALLED INPUTS: decl_specs_reentrance_bis, abs_decl_reentrance
     REENTRANCE VIA: abs_decl_reentrance, ..., declarator, expr CHAIN, 
                     abs_decl_reentrance, ..., decl_specs_reentrance_bis, ... CHAIN, 
                     decl_specs_reentrance_bis, ..., enum_type_define, expr CHAIN,
                     decl_specs_reentrance_bis, ..., declarator, expr CHAIN ***/
abs_decl: abs_decl_reentrance
        ;
        
type_name:
        {   
            gProject->Parse_TOS->parseCtxt->PushCtxt();
            gProject->Parse_TOS->parseCtxt->ResetVarParam();
        }
            type_name_bis
        {
            $$ = $2;
            gProject->Parse_TOS->parseCtxt->PopCtxt(false);
        }
        ;
        
type_name_bis:  decl_specs_reentrance_bis
        {
            assert ($1 == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            
            possibleType = true;
            $$ = $1;
            if ($$->isFunction())
                yyerr ("Function type not allowed as type name");
        }
         | decl_specs_reentrance_bis abs_decl
        {
            assert ($1 == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            
            possibleType = true;
            $$ = $2;
            
            Type * extended = $$->extend($1);
            if ($$->isFunction())
                yyerr ("Function type not allowed as type name");
            else if (extended && 
                $1 && $1->isFunction() && 
                ! extended->isPointer())
                yyerr ("Wrong type combination") ;
                
        }
        ;      
/*** INPUT RULES OF THIS SET: 
                    decl_specs_reentrance_bis (CALLING INPUTS: opt_param_type_list, type_name, decl_spec)
     CALLED INPUTS: type_spec_reentrance, type_qual
     REENTRANCE VIA : ..., type_name CHAIN,
                      ..., opt_param_type_list ***/
                      
decl_specs_reentrance_bis: decl_specs_reentrance
        {
            gProject->Parse_TOS->parseCtxt->SetDeclCtxt($$);
        }
        ;
          
local_or_global_storage_class: EXTRN
                             | STATIC
                             | TYPEDEF
        ;

local_storage_class: AUTO
                   | REGISTR
        ;
        
storage_class: local_or_global_storage_class
             | local_storage_class
        {
            if (! gProject->Parse_TOS->transUnit ||
                gProject->Parse_TOS->transUnit->contxt.syms->current->level >= FUNCTION_SCOPE)
                 $$ = $1 ;             
             else
                 $$ = ST_None ;              
        }        
        ;

type_spec: type_spec_reentrance
        ;   
        
opt_decl_specs_reentrance:   /* Nothing */
        {
            $$ = (BaseType*) NULL;
        }
              | decl_specs_reentrance
        ;

decl_specs_reentrance:  storage_class opt_decl_specs_reentrance
        {
            $$ = $2;

            if (!$$)
            {
                $$ = new BaseType();
            }

            if ($1 == ST_None)
                 yyerr("Invalid use of local storage type");
            else if ($$->storage != ST_None)             
                 yywarn("Overloading previous storage type specification");
            else
                 $$->storage = $1;
        }
          |  type_spec { possibleType = false; } opt_decl_specs_reentrance
        {
            $$ = $1;

            if ($3)
            {
                if (($3->typemask & BT_Long)
                    && ($$->typemask & BT_Long))
                {
                   // long long : A likely C9X addition 
                   yyerr("long long support has been removed");
                }
                else
                    $$->typemask |= $3->typemask;

                if ($3->storage != ST_None)
                    $$->storage = $3->storage;

                // delete $3;
            }

            /*
            std::cout << "In decl_spec: ";
            $$->printBase(std::cout,0);
            if ($$->storage == ST_Typedef)
                std::cout << "(is a typedef)";
            std::cout << std::endl;
            */
        }
          |  type_qual opt_decl_specs_reentrance
        {
            $$ = $2;

            if (!$$)
            {
                $$ = new BaseType();
            }

            if (TQ_None != ($$->qualifier & $1))
                yywarn("qualifier already specified");  
                              
            $$->qualifier |= $1;

        }
        ;
/*** INPUT RULES OF THIS SET:
                   comp_decl_specs (CALLING INPUTS: field_list)
     CALLED INPUTS: type_spec_reentrance, type_qual
     REENTRANCE VIA: type_spec_reentrance, field_list CHAIN ***/
opt_comp_decl_specs:   /* Nothing */
        {
           $$ = (BaseType*) NULL;
        }
              | comp_decl_specs_reentrance
        ;

comp_decl_specs_reentrance:  type_spec_reentrance { possibleType = false; } opt_comp_decl_specs
        {
            $$ = $1;

            if ($3)
            {
                $$->typemask |= $3->typemask;
                // delete $3;
            }
        }
               |  type_qual opt_comp_decl_specs
        {
            $$ = $2;

            if (!$$)
            {
                $$ = new BaseType();
            }

            if (TQ_None != ($$->qualifier & $1))
                yywarn("qualifier already specified");
            $$->qualifier |= $1;
        }
        ;
        
comp_decl_specs: comp_decl_specs_reentrance
        {
            gProject->Parse_TOS->parseCtxt->SetDeclCtxt($$);
        }
        ;
/*** INPUT RULES OF THIS SET: 
                   opt_init_decl_list (CALLING INPUTS: decl_stemnt, opt_KnR_declaration_list)
     CALLED INPUTS: declarator opt_gcc_attrib initializer
  NO REENTRANCE  ***/
decl: declarator
        {
           $1->extend(gProject->Parse_TOS->parseCtxt->UseDeclCtxt());
        }
      opt_gcc_attrib
        {
           $1->attrib = $3;
           $$ = $1;
        }
        ;
        
init_decl: decl
         | decl EQ initializer
        {
           $1->initializer = $3;
           $$ = $1;
        }
        ;
        
opt_init_decl_list:  /* Nothing */
        {
          $$ = (Decl*) NULL;
        }
                  |  init_decl_list
        ;

init_decl_list: init_decl_list_reentrance
        ;

init_decl_list_reentrance: init_decl
        {
            $$ = $1;
        }
              | init_decl_list_reentrance COMMA init_decl        %prec COMMA_OP
        {
            $$ = $1;

			Decl*	appendDecl = $1;
			while (appendDecl->next != NULL)
				appendDecl = appendDecl->next;

            appendDecl->next = $3;
            delete $2;
        }
        ;
/*** INPUT RULES OF THIS SET:
                   initializer  (CALLING INPUTS: opt_init_decl_list)
     CALLED INPUTS: assign_expr
  NO REENTRANCE  ***/
initializer: initializer_reentrance
        ;
initializer_list:  initializer_reentrance
        {
            $$ = new ArrayConstant(NoLocation);
            $$->addElement($1);
        }
                |  initializer_list COMMA initializer_reentrance        %prec COMMA_OP
        {
            $$ = $1;
            $$->addElement($3);
            delete $2;
        }
        ;

initializer_reentrance:  assign_expr
                      |  constructor_expr
                      |  iter_constructor_expr
                      |  LBRACE initializer_list opt_comma RBRACE
        {
            $$ = $2;
            delete $1;
            delete $4;
        }
        ;
        
opt_comma:    /* Nothing */
        {
            $$ = (Location*) NULL;
        }
         |  COMMA    %prec COMMA_SEP
        {
            delete $1;
            $$ = (Location*) NULL;
        }
        ;

/*** INPUT RULES OF THIS SET:
                   type_qual (CALLING INPUTS: decl_specs, comp_decl_specs), 
                   opt_type_qual_list (CALLING INPUTS: pointer)
   NO CALLED INPUTS
   NO REENTRANCE ***/
type_qual: type_qual_token
        ;
        
type_qual_token: CONST
               | VOLATILE
               | OUT
               | REDUCE
               | ITER
               | KERNEL
               | VOUT LBRCKT opt_const_expr RBRCKT
        {
           TypeQual r($1);
           r.vout=$3;
           $$ = r;
        }
        ;

type_qual_list: type_qual_token
              | type_qual_list type_qual_token
        {
            $$ = $1 | $2;
            if (TQ_None != ($2 & $1))
                yywarn("qualifier already specified");                               
        }
        ;

opt_type_qual_list:    /* Nothing */
        {
            $$ = TQ_None;
        }
        |   type_qual_list
        ;
/*** INPUT RULE OF THIS SET: 
                   type_spec (CALLING INPUTS: decl_specs, comp_decl_specs), 
     CALLED INPUTS: any_ident, field_list, enum_def_list
     REENTRANCE VIA: field_list, comp_decl_specs CHAIN
     REENTRANCE VIA: enum_def_list, assign_expr CHAIN ***/  
type_spec_reentrance: enum_type_define
         |  struct_type_define
         |  union_type_define
         |  enum_tag_ref 
         |  struct_tag_ref
         |  union_tag_ref
         |  typedef_name
         |  VOID
         |  CHAR
         |  SHORT
         |  INT
         |  LONG
         |  FLOAT
         |  FLOAT2
         |  FLOAT3
         |  FLOAT4
         |  FIXED
         |  FIXED2
         |  FIXED3
         |  FIXED4
         |  HALF
         |  HALF2
         |  HALF3
         |  HALF4
         |  DOUBLE
         |  DOUBLE2
         |  SGNED
         |  UNSGNED
        ;

typedef_name:  TYPEDEF_NAME
        {
            $$ = new BaseType(BT_UserType);
            $$->typeName = $1;
        }
        ;
        
tag_ref:    TAG_NAME
        {
            assert ((! $$->entry) || 
                    $$->entry->IsTagDecl()) ;
        }
        ;
        
        
struct_tag_ref:  STRUCT tag_ref
        {
            if (gProject->Parse_TOS->transUnit)
                $$ = gProject->Parse_TOS->parseCtxt->Mk_tag_ref($1, $2,
                                                                gProject->Parse_TOS->transUnit->contxt.tags);
            else
                $$ = NULL;                                         
        }
        ;
        
union_tag_ref:  UNION tag_ref
        {
            if (gProject->Parse_TOS->transUnit)
                $$ = gProject->Parse_TOS->parseCtxt->Mk_tag_ref($1, $2,
                                                                gProject->Parse_TOS->transUnit->contxt.tags);
            else
                $$ = NULL;                                         
        }
        ;
        
enum_tag_ref:  ENUM tag_ref
        {
            if (gProject->Parse_TOS->transUnit)
                $$ = gProject->Parse_TOS->parseCtxt->Mk_tag_ref($1, $2,
                                                                gProject->Parse_TOS->transUnit->contxt.tags);
            else
                $$ = NULL;                                         
        }
        ;
        
struct_tag_def:   STRUCT tag_ref
        {
            if (gProject->Parse_TOS->transUnit)
                $$ = gProject->Parse_TOS->parseCtxt->Mk_tag_def($1, $2,
                                                            gProject->Parse_TOS->transUnit->contxt.tags);
            else
                $$ = NULL;                                         
        }
        ;   
              
struct_type_define: STRUCT LBRACE struct_or_union_definition RBRACE
        {
            $$ = new BaseType($3);
            $3->_isUnion = false;
            delete $2;
            delete $4;
        }
                  | struct_tag_def LBRACE struct_or_union_definition RBRACE
        {
            $$ = $1;
            assert (! $$->stDefn);
            $$->stDefn = $3;
            $3->tag = $1->tag->dup();
            $3->_isUnion = false;

            // Overload the incomplete definition
            $$->tag->entry->uStructDef = $$ ;
            
//             std::cout << "struct/union/enum_type_define:"
//                          "The definition of:"
//                       << "(uStructDef:" << $1->tag->entry->uStructDef << ")"
//                       << "(uStructDef->stDefn:" << $1->tag->entry->uStructDef->stDefn << ")"
//                       << "(" << $1->tag->entry << ")" << $1->tag->name  << "$" ;
//             $1->tag->entry->scope->ShowScopeId(std::cout);
//             std::cout << " has been completed" << endl; 
            
            delete $2;
            delete $4;
        }
        ;

union_tag_def:   UNION tag_ref
        {
            if (gProject->Parse_TOS->transUnit)
                $$ = gProject->Parse_TOS->parseCtxt->Mk_tag_def($1, $2,
                                                            gProject->Parse_TOS->transUnit->contxt.tags);
            else
              $$ = NULL ;
        }
        ;   
              
union_type_define:  UNION LBRACE struct_or_union_definition RBRACE
        {
            $$ = new BaseType($3);
            $3->_isUnion = true;

            delete $2;
            delete $4;
        }
                  | union_tag_def LBRACE struct_or_union_definition RBRACE
        {
            $$ = $1;
            assert (! $$->stDefn);
            $$->stDefn = $3;
            $3->tag = $1->tag->dup();
            $3->_isUnion = true;

            // Overload the incomplete definition
            $$->tag->entry->uStructDef = $$ ;
            
//             std::cout << "struct/union/enum_type_define:"
//                          "The definition of:"
//                       << "(uStructDef:" << $1->tag->entry->uStructDef << ")"
//                       << "(uStructDef->stDefn:" << $1->tag->entry->uStructDef->stDefn << ")"
//                       << "(" << $1->tag->entry << ")" << $1->tag->name  << "$" ;
//             $1->tag->entry->scope->ShowScopeId(std::cout);
//             std::cout << " has been completed" << endl; 
            
            delete $2;
            delete $4;
 
        }
        ;
        
enum_tag_def:   ENUM tag_ref
        {
            if (gProject->Parse_TOS->transUnit)
                $$ = gProject->Parse_TOS->parseCtxt->Mk_tag_def($1,$2,
                                                            gProject->Parse_TOS->transUnit->contxt.tags);
            else
              $$ = NULL;
        }
        ;   
              
enum_type_define:  ENUM LBRACE enum_definition RBRACE
        {
            $$ = new BaseType($3);

            delete $2;
            delete $4;
        }
                | enum_tag_def LBRACE enum_definition RBRACE
        {
            $$ = $1;
            assert (! $$->enDefn);
            $$->enDefn = $3;
            $3->tag = $1->tag->dup();

            // Overload the incomplete definition
            $$->tag->entry->uStructDef = $$ ;
            
//             std::cout << "struct/union/enum_type_define:"
//                          "The definition of:"
//                       << "(uStructDef:" << $1->tag->entry->uStructDef << ")"
//                       << "(uStructDef->stDefn:" << $1->tag->entry->uStructDef->stDefn << ")"
//                       << "(" << $1->tag->entry << ")" << $1->tag->name  << "$" ;
//             $1->tag->entry->scope->ShowScopeId(std::cout);
//             std::cout << " has been completed" << endl; 
            
            delete $2;
            delete $4;
        }
        ;
         
struct_or_union_definition: 
        {  $$ = new StructDef();
           yywarn("ANSI/ISO C prohibits empty struct/union");
        }        
                          |  field_list       
        ;
        
enum_definition: 
        {  $$ = new EnumDef();
           yywarn("ANSI/ISO C prohibits empty enum");
        }        
               |   enum_def_list opt_trailing_comma    
        {  $$ = $1;
        }        
        ;
        
opt_trailing_comma:    /* Nothing */
        {
            $$ = NULL;
        }
                  | COMMA    %prec COMMA_SEP
        {
          yywarn("Trailing comma in enum type definition");
        }
        ;

/*** INPUT RULES OF THIS SET:
                   enum_def_list (CALLING INPUTS: type_spec_reentrance)
     CALLED INPUTS: any_ident, assign_expr
     REENTRANCE VIA: assign_expr CHAIN ***/  
enum_def_list:  enum_def_list_reentrance
        ;
        
enum_def_list_reentrance:  enum_const_def
        {
            $$ = new EnumDef();
            $$->addElement($1);
        }
             |  enum_def_list COMMA enum_const_def        %prec COMMA_OP
        {
            $$ = $1;
            $$->addElement($3);
            delete $2;
        }
        ;

enum_const_def:  enum_constant
        {
            $$ = new EnumConstant($1,NULL,NoLocation);
            if (gProject->Parse_TOS->transUnit)
            {
              if (gProject->Parse_TOS->transUnit->contxt.syms->IsDefined($1->name))
                 yyerr("Duplicate enumeration constant");
                 
              $1->entry = gProject->Parse_TOS->transUnit->contxt.syms->Insert(
                                  mk_enum_const($1->name, $$));
            }
        }
              |  enum_constant EQ assign_expr 
        {
            $$ = new EnumConstant($1,$3,NoLocation);
            if (gProject->Parse_TOS->transUnit)
            {
              if (gProject->Parse_TOS->transUnit->contxt.syms->IsDefined($1->name))
                 yyerr("Duplicate enumeration constant");
                 
              $1->entry = gProject->Parse_TOS->transUnit->contxt.syms->Insert(
                                  mk_enum_const($1->name, $$));
            }
        }
        ;

enum_constant:  any_ident
        ;
/*** INPUT RULE OF THIS SET:
                   field_list (CALLING RULES:) 
     CALLED INPUTS: comp_decl_specs comp_decl_list
   REENTRANCE VIA: comp_decl_specs, type_spec_reentrance CHAIN ***/
field_list:
        {
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
            gProject->Parse_TOS->parseCtxt->PushCtxt();
        }
        {
            assert (!err_top_level || possibleType);
             /* Safety precaution! */
             possibleType=true;
        }
            field_list_reentrance
        {
            gProject->Parse_TOS->parseCtxt->PopCtxt(false);
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope();
            $$ = $3 ;
        }
        ;         

field_list_reentrance:  comp_decl SEMICOLON
        {
            $$ = new StructDef();
            $$->addComponent(ReverseList($1));
            delete $2;
        }
          |  field_list_reentrance SEMICOLON
        {
            // A useful gcc extension:
            //   naked semicolons in struct/union definitions. 
            $$ = $1;
            yywarn ("Empty declaration");
            delete $2;
        }
          |  field_list_reentrance comp_decl SEMICOLON
        {
            $$ = $1;
            $$->addComponent(ReverseList($2));
            delete $3;
        }
        ;
        
comp_decl:  comp_decl_specs comp_decl_list
        {
            possibleType = true;
            $$ = $2;
        }
         |  comp_decl_specs
        {
            possibleType = true;
            $$ = new Decl ($1);
            yywarn ("No field declarator");
        }
        ;
/*** INPUT RULE OF THIS SET:
                   comp_decl_list (CALLING INPUTS: field_list)
     CALLED INPUTS: opt_gcc_attrib declarator, cond_expr
   NO REENTRANCE VIA declarator ??? ***/
comp_decl_list: comp_decl_list_reentrance
        ;
              
comp_decl_list_reentrance:
        {   gProject->Parse_TOS->parseCtxt->SetIsFieldId(true); 
        } 
                           comp_declarator opt_gcc_attrib
        {
            $$ = $2;
            $$->attrib = $3;
        }
                        |  comp_decl_list_reentrance COMMA
        {   gProject->Parse_TOS->parseCtxt->SetIsFieldId(true); 
        } 
                           comp_declarator opt_gcc_attrib   %prec COMMA_OP
        {
            $$ = $4;
            $$->attrib = $5;
            $$->next = $1;
            delete $2;
        }
        ;

comp_declarator:  simple_comp
        {
           gProject->Parse_TOS->parseCtxt->SetIsFieldId(false); 
           Type * decl = gProject->Parse_TOS->parseCtxt->UseDeclCtxt();
           Type * extended = $$->extend(decl);
           if ($$->form &&
               $$->form->isFunction())
               yyerr ("Function type not allowed as field");
           else if ($$->form &&
                    $$->form->isArray() &&
                    ! ((ArrayType *) $$->form)->size)
               yyerr ("Unsized array not allowed as field");
           else if (extended && 
               decl && decl->isFunction() && 
               ! extended->isPointer())
               yyerr ("Wrong type combination") ;
                
        }
               |  bit_field
        {
           Type * decl = gProject->Parse_TOS->parseCtxt->UseDeclCtxt();
           $$->extend(decl);
           if (! decl)
               yyerr ("No type specifier for bit field") ;
           else if (!$$->form)
               yyerr ("Wrong type combination") ;
        }
        ;

simple_comp:  declarator
        ;

bit_field:  opt_declarator COLON 
        {   gProject->Parse_TOS->parseCtxt->SetIsFieldId(false); 
        }
            width
        {
            BitFieldType  *bf = new BitFieldType($4);
            $$ = $1;

            if ($$ == NULL)
            {
                $$ = new Decl(bf);
            }
            else
            {
                bf->subType = $$->form;
                $$->form = bf;
            }
        }
        ;

width:  cond_expr
        ;
        
opt_declarator:  /* Nothing */
        {
           $$ = (Decl*) NULL;
        }
              |  declarator
        ;
/*** INPUT RULE OF THIS SET:
                   declarator (CALLING INPUTS: program, opt_init_decl_list, comp_decl_list)
     CALLED INPUTS: pointer, ident, ident_list, param_type_list, opt_const_expr
     REENTRANCE VIA param_type_list
     REENTRANCE VIA opt_const_expr ***/
declarator: declarator_reentrance_bis
        {
            gProject->Parse_TOS->parseCtxt->Mk_declarator ($$);
        } 
        ;
        
func_declarator: declarator_reentrance_bis
        {
            gProject->Parse_TOS->parseCtxt->Mk_func_declarator ($$);
        } 
        ;
        
declarator_reentrance_bis: pointer direct_declarator_reentrance_bis
        {
            $$ = $2;
            $$->extend($1);
        }
          | direct_declarator_reentrance_bis
        ;

direct_declarator_reentrance_bis:  direct_declarator_reentrance
       ;

direct_declarator_reentrance:  ident
        {  if (gProject->Parse_TOS->transUnit)
                $$ = gProject->Parse_TOS->parseCtxt->Mk_direct_declarator_reentrance ($1,
                gProject->Parse_TOS->transUnit->contxt.syms);
        }
        |  LPAREN declarator_reentrance_bis RPAREN
        {
            $$ = $2;
            delete $1 ;
            delete $3 ;
        }
        |  array_decl
        |  stream_decl
        |  direct_declarator_reentrance LPAREN param_type_list RPAREN
        {
            $$ = $1;
            FunctionType * ft = new FunctionType(ReverseList($3));
            Type * extended = $$->extend(ft);
            if (extended && ! extended->isPointer())
                yyerr ("Wrong type combination") ;
                
            delete $2 ;
            delete $4 ;
            // Exit, but will allow re-enter for a function.
            // Hack, to handle parameters being in the function's scope.
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope(true);

        }
        |  direct_declarator_reentrance LPAREN ident_list RPAREN
        {
            $$ = $1;
            FunctionType * ft = new FunctionType(ReverseList($3));
            Type * extended = $$->extend(ft);
            if (extended && ! extended->isPointer())
                yyerr ("Wrong type combination") ;

            delete $2 ;
            delete $4 ;
            // Exit, but will allow re-enter for a function.
            // Hack, to handle parameters being in the function's scope.
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.ExitScope(true);

        }
        |  direct_declarator_reentrance LPAREN RPAREN
        {
            $$ = $1;

			if ($$ != NULL)
			{
				FunctionType* ft = new FunctionType();
				Type* extended = $$->extend(ft);
				if (extended && ! extended->isPointer())
           	 	    yyerr ("Wrong type combination") ;
			}
            
            delete $2 ;
            delete $3 ;
            if (gProject->Parse_TOS->transUnit)
            {
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
                // Exit, but will allow re-enter for a function.
                // Hack, to handle parameters being in the function's scope.
                gProject->Parse_TOS->transUnit->contxt.ExitScope(true);
            }
        }
        ;

array_decl: direct_declarator_reentrance LBRCKT opt_const_expr RBRCKT
        {
            $$ = $1;
            ArrayType * at = new ArrayType(TT_Array, $3);
            Type * extended = $$->extend(at);
            if (extended && 
                extended->isFunction())
                yyerr ("Wrong type combination") ;
              
            delete $2 ;
            delete $4 ;
        }
        ;       

stream_decl: direct_declarator_reentrance COMP_LESS comma_constants COMP_GRTR
        {
            $$ = $1;
            ArrayType * at = new ArrayType(TT_Stream, $3);
            Type * extended = $$->extend(at);

            if (extended &&
                extended->isFunction())
                yyerr ("Wrong type combination") ;
        }
        ;
dimension_constraint:  constant
        {
            $$ = $1;
        }
          | func_call
        {
            $$ = $1;
        }
          | LPAREN expr RPAREN
        {
           $$ = $2;
        }
          | ident
        { 
            $$ = new Variable ($1,NoLocation);
        }
        ;

comma_constants:  /* Empty */
	{
	   $$ = NULL;
	}
          | dimension_constraint
        {
            $$ = $1;
        }
          |  comma_constants COMMA dimension_constraint
        {
            $$ = new BinaryExpr(BO_Comma,$1,$3,*$2);
            delete $2;
        }
	;
/*** INPUT RULE OF THIS SET:
                  pointer (CALLING RULES: declarator) 
     CALLED INPUTS: opt_type_qual_list
   NO REENTRANCE ***/
pointer_start:  STAR opt_type_qual_list
        {
            $$ = new PtrType($2);    
        }
        ;

pointer_reentrance:  pointer_start
       |  pointer_reentrance pointer_start
        {
            $$ = $2;
            $$->subType = $1;
        }
        ;
        
pointer:  pointer_reentrance
        ;
/*** INPUT RULE OF THIS SET:
                   ident_list (CALLING INPUTS: declarator) 
     CALLED INPUTS: ident
   NO REENTRANCE ***/
ident_list:
       {  gProject->Parse_TOS->parseCtxt->IncrVarParam(1);
          if (gProject->Parse_TOS->transUnit)
              gProject->Parse_TOS->transUnit->contxt.EnterScope();
          gProject->Parse_TOS->parseCtxt->PushCtxt();
        }
              ident_list_reentrance
       {
          // Exit, but will allow re-enter for a function.
          // Hack, to handle parameters being in the function's scope.
          gProject->Parse_TOS->parseCtxt->PopCtxt(true);
          gProject->Parse_TOS->parseCtxt->IncrVarParam(-1);
          $$ = $2;
       }
        ;

ident_list_reentrance: ident
        {  if (gProject->Parse_TOS->transUnit)
               $$ = gProject->Parse_TOS->parseCtxt->Mk_direct_declarator_reentrance ($1,
                gProject->Parse_TOS->transUnit->contxt.syms);
        }
          | ident_list_reentrance COMMA ident        %prec COMMA_OP
        {  $$ = $1;
           if (gProject->Parse_TOS->transUnit)
           {
              $$ = gProject->Parse_TOS->parseCtxt->Mk_direct_declarator_reentrance ($3,
                gProject->Parse_TOS->transUnit->contxt.syms);
              $$->next = $1;
           }
        }
        ;
/*** TERMINAL INPUT RULE OF THIS SET:
                   ident (CALLING INPUTS: stemnt, any_ident) ***/
ident: IDENT
        ;
/*** TERMINAL INPUT RULE OF THIS SET:
                   typename_as_ident (CALLING INPUTS: field_ident, any_ident) ***/
typename_as_ident: TYPEDEF_NAME
        {
            /* Convert a TYPEDEF_NAME back into a normal IDENT */
            $$ = $1;
            $$->entry = (SymEntry*) NULL;
        }
        ;

/*** INPUT RULE OF THIS SET:
                   any_ident (CALLING INPUTS: stemnt, EXPR SET)
     CALLED INPUTS: ident, typename_as_ident
  NO REENTRANCE ***/
any_ident: ident
         | typename_as_ident
        ;
/*** INPUT RULE OF THIS SET: 
                   opt_param_type_list (CALLING INPUTS: abs_decl_reentrance)
                   param_type_list (CALLING INPUTS: declarator)
     CALLED INPUTS: decl_specs declarator abs_decl
     REENTRANCE VIA: abs_decl_reentrance
     REENTRANCE VIA: declarator ***/
opt_param_type_list:  /* Nothing */
        {
           $$ = (Decl*) NULL;
        }
                   |  
        { gProject->Parse_TOS->parseCtxt->IncrVarParam(1); 
        }
                      param_type_list_bis
        { gProject->Parse_TOS->parseCtxt->IncrVarParam(-1); 
           $$ = $2;
        }
        ;

param_type_list:
        {   gProject->Parse_TOS->parseCtxt->IncrVarParam(1);
            if (gProject->Parse_TOS->transUnit)
                gProject->Parse_TOS->transUnit->contxt.EnterScope();
            gProject->Parse_TOS->parseCtxt->PushCtxt();
        }
                 param_type_list_bis
        {
            gProject->Parse_TOS->parseCtxt->PopCtxt(true);
            gProject->Parse_TOS->parseCtxt->IncrVarParam(-1);
            $$ = $2 ;
        }
        ;

param_type_list_bis: param_list
               | param_list COMMA ELLIPSIS        %prec COMMA_OP
        {
            BaseType *bt = new BaseType(BT_Ellipsis);

            $$ = new Decl(bt);
            $$->next = $1;
        }
        ;

param_list: param_decl
          | param_list COMMA param_decl        %prec COMMA_OP
        {
            $$ = $3;
            $$->next = $1;
        }
        ;
        
param_decl:
        {   
            gProject->Parse_TOS->parseCtxt->PushCtxt();
        }
                     param_decl_bis
        {
            gProject->Parse_TOS->parseCtxt->PopCtxt(true);
            $$ = $2;
        }
        ;
        
param_decl_bis: decl_specs_reentrance_bis declarator
        {
            assert (err_top_level ||
                    $1 == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            possibleType = true;
            $$ = $2;
            Type * decl = gProject->Parse_TOS->parseCtxt->UseDeclCtxt();
            Type * extended = $$->extend(decl);             
            if ($$->form &&
                $$->form->isFunction())
                yyerr ("Function type not allowed");
            else if (extended && 
                decl && decl->isFunction() && 
                ! extended->isPointer())
                yyerr ("Wrong type combination") ;
        }
          | decl_specs_reentrance_bis abs_decl_reentrance
        {
            assert (err_top_level ||
                    $1 == gProject->Parse_TOS->parseCtxt->curCtxt->decl_specs);
            possibleType = true;
            $$ = new Decl($2);
            
            Type * decl = gProject->Parse_TOS->parseCtxt->UseDeclCtxt();
            Type * extended = $$->extend(decl);
            if ($$->form &&
                $$->form->isFunction())
                yyerr ("Function type not allowed for parameter");
            else if (extended && 
                decl && decl->isFunction() && 
                ! extended->isPointer())
                yyerr ("Wrong type combination") ;
        }
          | decl_specs_reentrance_bis
        {
            possibleType = true;
            $$ = new Decl($1);
            if ($$->form &&
                $$->form->isFunction())
                yyerr ("Function type not allowed for parameter");
        }
        ;
/*** INPUT RULE OF THIS SET:
                   abs_decl_reentrance (CALLING INPUTS: abs_decl, opt_param_type_list)
     CALLED INPUTS: pointer, opt_const_expr, opt_param_type_list
     REENTRANCE VIA: opt_param_type_list ***/
abs_decl_reentrance:  pointer
        {
            $$ = $1;
        }
        |  direct_abs_decl_reentrance_bis
        {
            $$ = $1;
        }
        |  pointer direct_abs_decl_reentrance_bis
        {
            $$ = $2;
            $$->extend($1);
        }
        ;

direct_abs_decl_reentrance_bis:  direct_abs_decl_reentrance
        ;

direct_abs_decl_reentrance:  LPAREN abs_decl_reentrance RPAREN
        {
            $$ = $2;
        }
        |  LBRCKT opt_const_expr RBRCKT
        {
            $$ = new ArrayType(TT_Array, $2);
        }
        |  direct_abs_decl_reentrance LBRCKT opt_const_expr RBRCKT
        {
            ArrayType *at = new ArrayType(TT_Array, $3);
            $$ = $1;
            $$->extend(at);
            Type * extended = $$->extend(at) ;
            if (extended && 
                extended->isFunction())
                yyerr ("Wrong type combination") ;
        }
        |  LPAREN opt_param_type_list RPAREN
        {
            $$ = new FunctionType(ReverseList($2));
        }
        |  direct_abs_decl_reentrance LPAREN opt_param_type_list RPAREN
        {
            FunctionType * ft = new FunctionType(ReverseList($3));
            $$ = $1;
            Type * extended = $$->extend(ft) ;
            if (extended && 
                ! extended->isPointer())
                yyerr ("Wrong type combination") ;
                
        }
        ;
/*** INPUT RULES OF THIS SET:
                    opt_gcc_attrib (CALLING INPUTS: opt_init_decl_list comp_decl_list)  
  NO CALLED UNTERMINAL INPUTS
     CALLED   TERMINAL INPUTS: ident
  NO REENTRANCE ***/
opt_gcc_attrib:  /* Nothing */
        {
            $$ = (GccAttrib*) NULL;
        }
        |  gcc_attrib
        ;

gcc_attrib:    ATTRIBUTE LPAREN LPAREN gcc_inner RPAREN RPAREN
            {
                $$ = $4;
                delete $2;
                delete $3;
                delete $5;
                delete $6;
            }
            ;

gcc_inner:  /* Nothing */
            {
                /* The lexer ate some unsupported option. */
                $$ = new GccAttrib( GCC_Unsupported);
            }
         |   PACKED
            {
                $$ = new GccAttrib( GCC_Packed );
            }
         |   CDECL
            {
                $$ = new GccAttrib( GCC_CDecl );
            }
         |   CONST
            {
                $$ = new GccAttrib( GCC_Const );
            }
         |   NORETURN
            {
                $$ = new GccAttrib( GCC_NoReturn );
            }
         |   ALIGNED LPAREN INUM RPAREN
            {
                $$ = new GccAttrib( GCC_Aligned );

                if ($3->ctype == CT_Int)
                {
                    IntConstant    *iCons = (IntConstant*) $3;

                    $$->value = iCons->lng;
                }

                delete $2;
                delete $4;
            }
         |   MODE LPAREN ident RPAREN
            {
                $$ = new GccAttrib( GCC_Mode );

                $$->mode = $3;

                delete $2;
                delete $4;
            }
         |   FORMAT LPAREN ident COMMA INUM COMMA INUM RPAREN
            {
                $$ = new GccAttrib( GCC_Format );
    
                $$->mode = $3;

                if ($5->ctype == CT_Int)
                {
                    IntConstant    *iCons = (IntConstant*) $5;

                    $$->strIdx = iCons->lng;
                }

                if ($7->ctype == CT_Int)
                {
                    IntConstant    *iCons = (IntConstant*) $7;

                    $$->first = iCons->lng;
                }

                delete $2;
                delete $8;
            }
            ;

%%

/*******************************************************/
