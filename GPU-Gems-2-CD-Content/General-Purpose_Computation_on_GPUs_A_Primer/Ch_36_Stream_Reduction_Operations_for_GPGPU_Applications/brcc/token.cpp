
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
/*  ###############################################################
    ##
    ##     C Tree Builder
    ##
    ##     File:         token.c
    ##
    ##     Programmer:   Shaun Flisakowski
    ##     Date:         Dec 24, 1994
    ##
    ###############################################################  */

#include    <cstdio>
#include    <cstring>
#include    <cstdlib>
#include    <cassert>

#include    "gram.h"

/*  ###############################################################  */

char *toksym(int tok, int white)
{
    switch (tok)
    {
    /* Special Tokens */
    default:
    case INVALID:
        return("Invalid Token");

    case IDENT:
        return("Identifier");

    case TAG_NAME:
        return("Tag Name");

    case TYPEDEF_NAME:
        return("Typedef Name");

    case STRING:
        return("String Constant");

    case LSTRING:
        return("Wide String Constant");

    case CHAR_CONST:
        return("Character Constant");

    case LCHAR_CONST:
        return("Wide Character Constant");

    case INUM:
        return("Integer Numeric Constant");

    case RNUM:
        return("Real Numeric Constant");

    
    /* Regular keywords */
    case AUTO:
        if (white)
          return("auto ");
        else
          return("auto");

    case BREAK:
        return("break");

    case CASE:
        if (white)
          return("case ");
        else
          return("case");

    case CHAR:
        if (white)
          return("char ");
        else
          return("char");

    case CONST:
        if (white)
          return("const ");
        else
          return("const");

    case CONT:
        return("continue");

    case DEFLT:
        return("default");

    case DO:
        if (white)
          return("do ");
        else
          return("do");

    case DOUBLE:
        if (white)
          return("double ");
        else
          return("double");

    case ELSE:
        return("else");

    case ENUM:
        if (white)
          return("enum ");
        else
          return("enum");

    case EXTRN:
        if (white)
          return("extern ");
        else
          return("extern");

    case FLOAT:
        if (white)
          return("float ");
        else
          return("float");

    case FLOAT2:
        if (white)
          return("float2 ");
        else
          return("float2");

    case FLOAT3:
        if (white)
          return("float3 ");
        else
          return("float3");

    case FLOAT4:
        if (white)
          return("float4 ");
        else
          return("float4");

    case FOR:
        if (white)
          return("for ");
        else
          return("for");

    case GOTO:
        if (white)
          return("goto ");
        else
          return("goto");

    case IF:
        if (white)
          return("if ");
        else
          return("if");

    case INT:
        if (white)
          return("int ");
        else
          return("int");

    case KERNEL:
        if (white)
          return("kernel ");
        else
          return("kernel");

    case LONG:
        if (white)
          return("long ");
        else
          return("long");

    case OUT:
        if (white)
          return("out ");
        else
          return("out");
    case REDUCE:
        if (white)
	  return ("reduce ");
	else
	  return ("reduce");
    case REGISTR:
        if (white)
          return("register ");
        else
          return("register");

    case RETURN:
        if (white)
          return("return ");
        else
          return("return");

    case SHORT:
        if (white)
          return("short ");
        else
          return("short");

    case SGNED:
        if (white)
          return("signed ");
        else
          return("signed");

    case SIZEOF:
        return("sizeof");

    case STATIC:
        if (white)
          return("static ");
        else
          return("static");

    case STRUCT:
        if (white)
          return("struct ");
        else
          return("struct");

    case SWITCH:
        if (white)
          return("switch ");
        else
          return("switch");

    case TYPEDEF:
        if (white)
          return("typedef ");
        else
          return("typedef");

    case UNION:
        if (white)
          return("union ");
        else
          return("union");

    case UNSGNED:
        if (white)
          return("unsigned ");
        else
          return("unsigned");

    case VOID:
        if (white)
          return("void ");
        else
          return("void");

    case VOLATILE:
        if (white)
          return("volatile ");
        else
          return("volatile");

    case WHILE:
        if (white)
          return("while ");
        else
          return("while");
    
    // Gcc Extensions
    case ATTRIBUTE:
        if (white)
          return("__attribute__ ");
        else
          return("__attribute__");

    case ALIGNED:
        if (white)
          return("aligned ");
        else
          return("aligned");

    case PACKED:
        if (white)
          return("packed ");
        else
          return("packed");

    // Operators
    case PLUS:
        return("+");

    case MINUS:
        return("-");

    case STAR:
        return("*");

    case DIV:
        return("/");

    case MOD:
        return("%");
    

    case EQ:
        return("=");

    case PLUS_EQ:
        return("+=");

    case MINUS_EQ:
        return("-=");

    case STAR_EQ:
        return("*=");

    case DIV_EQ:
        return("/=");

    case MOD_EQ:
        return("%=");
    

    case NOT:
        return("!");

    case AND:
        return("&&");

    case OR:
        return("||");

    case B_NOT:
        return("~");

    case B_AND:
        return("&");

    case B_OR:
        return("|");

    case B_XOR:
        return("^");
    

    case B_AND_EQ:
        return("&=");

    case B_OR_EQ:
        return("|=");

    case B_XOR_EQ:
        return("^=");
    

    case L_SHIFT:
        return("<<");

    case R_SHIFT:
        return(">>");

    case L_SHIFT_EQ:
        return("<<=");

    case R_SHIFT_EQ:
        return(">>=");
    

    case EQUAL:
        return("==");

    case LESS:
        return("<");

    case LESS_EQ:
        return("<=");

    case GRTR:
        return(">");

    case GRTR_EQ:
        return(">=");

    case NOT_EQ:
        return("!=");
    

    case ASSIGN:
        return("Invalid (assignment)");

    case INCR:
        return("++");

    case DECR:
        return("--");
    

    case LPAREN:
        return("(");

    case RPAREN:
        return(")");

    case LBRCKT:
        return("[");

    case RBRCKT:
        return("]");

    case LBRACE:
        return("{");

    case RBRACE:
        return("}");
    

    case DOT:
        return(".");

    case ARROW:
        return("->");
    

    case QUESTMARK:
        return("?");

    case COLON:
        return(":");

    case SEMICOLON:
        return(";");

    case COMMA:
        return(",");

    case ELLIPSIS:
        return("...");
    

    case LB_SIGN:
        return("#");

    case DOUB_LB_SIGN:
        return("##");


    /* Illegal? */    
    case BACKQUOTE:
        return("`");

    case AT:
        return("@");

    case PP_LINE:
        return("#line");
    }
}

/*  ###############################################################  */

