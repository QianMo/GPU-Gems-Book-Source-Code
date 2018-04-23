
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
/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
    o+
    o+     File:         context.cpp
    o+
    o+     Programmer:   Patrick Baudin
    o+     Date:         Oct 30, 2000
    o+
    o+     Manage parsing context for SymEntry insertion.
    o+
    o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */

#include <iostream>

#include "config.h"

#include "parseenv.h"
#include "context.h"
#include "symbol.h"
#include "project.h"
#include "decl.h"
#include "stemnt.h"

//#define STATISTICS

#define PARSE_CTXT_TBL_SIZE (16)

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */

EXTERN int yyerr ARGS((char* s, std::string& str));
EXTERN int yyerr ARGS((char* s));
EXTERN int yywarn ARGS((char* s));
EXTERN int err_top_level;

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void yyCheckLabelsDefinition (SymTbl* labels)
{
	if (labels->clevel == FUNCTION_SCOPE && labels->current)
	{
		HashTbl* htab = labels->current->htab;
		if (htab)
			for (int j = 0; j < htab->tsize; j++)
				for (SymEntry* curr = htab->tab[j]; curr; curr = curr->next)
				{
					assert(curr->IsLabelDecl());
					if (! curr->u2LabelPosition)
						yyerr ("undefined label - ", curr->name);
				}
	}
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
ParseCtxt::ParseCtxt (void) :
  size(PARSE_CTXT_TBL_SIZE), 
  isFieldId(false)
{
  tabCtxt = new ParseEnvCtxt [size];
  curCtxt = tabCtxt;
  ResetDeclCtxt();
  curCtxt->varParam = 0;
  curCtxt->possibleDuplication = NULL;
}
/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
ParseCtxt::~ParseCtxt (void)
{
    delete [] tabCtxt;
#ifdef STATISTICS
    std::cout << "ParseCtxt::size=" << size << std::endl;
#endif
}
/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void 
ParseCtxt::SetDeclCtxt (BaseType* decl_specsCtxt)
{
#ifdef DECL_DEBUG
	std::cout << "SetDeclCtxt: ";
	if (curCtxt->decl_specs)
	{
		std::cout << "(Overloading previous setting!: ";
		curCtxt->decl_specs->printBase(std::cout, 0);
		std::cout << ", used: " << curCtxt->nb_decl_uses << ")";
	}  
#endif
  
	curCtxt->decl_specs = decl_specsCtxt;
	curCtxt->nb_decl_uses = 0;
  
#ifdef DECL_DEBUG
	curCtxt->decl_specs->printBase(std::cout, 0);
	std::cout << "\n";
#endif
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void ParseCtxt::ResetDeclCtxt (void)
{
  curCtxt->decl_specs = (BaseType*) NULL; 
  curCtxt->nb_decl_uses = 0;
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
Type* ParseCtxt::UseDeclCtxt (void)
{
  curCtxt->nb_decl_uses++; 
  if (curCtxt->nb_decl_uses == 1)
    return curCtxt->decl_specs;
  else
    return curCtxt->decl_specs->dup();
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void ParseCtxt::PushCtxt (void)
{
    int varParam = curCtxt->varParam;
    curCtxt++;
    if (curCtxt == tabCtxt + size)
    {
        int old_size = size;
        size += size;
        ParseEnvCtxt* old_tabCtxt = tabCtxt;
        tabCtxt = new ParseEnvCtxt [size];
        for (int j = 0; j < old_size; j++)
            tabCtxt[j] = old_tabCtxt[j];
        delete [] old_tabCtxt;
        curCtxt = tabCtxt + old_size;
    }    
    curCtxt->varParam = varParam;
    curCtxt->possibleDuplication = NULL;
    curCtxt->isKnR = false;
    ResetDeclCtxt();
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void ParseCtxt::PopCtxt (bool assert_checking)
{
  assert(curCtxt != tabCtxt);
  
  int varParam = curCtxt->varParam;
  curCtxt--;
  
  assert(! assert_checking || curCtxt->varParam == varParam);
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void ParseCtxt::SetVarParam (int val, bool assert_checking, int should_be)
{
  assert(! assert_checking || should_be == curCtxt->varParam);
  
  curCtxt->varParam = val;
}
/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void ParseCtxt::ReinitializeCtxt (void)
{
	curCtxt = tabCtxt;
	ResetDeclCtxt();
	isFieldId = false;
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
BaseType* ParseCtxt::Mk_tag_ref (BaseTypeSpec tagType, Symbol* tagSym, SymTbl* tags)
{
	if (! tagSym)
		return NULL;

	BaseType* result = new BaseType(tagType);
	result->tag = tagSym;

	if (tagSym->entry)
	{
//     std::cout << "struct/union/enum_tag_ref:"
//       "There is a previous tag entry:"
//               << "(" << tagSym->entry << ")" << tagSym->name << "$";
//     tagSym->entry->scope->ShowScopeId(std::cout);
//     std::cout << " which have to be consistent" << endl;

		if (! tagSym->entry->uStructDef || tagSym->entry->uStructDef->typemask != tagType)
		{
			// ... which isn't compatible.
			yyerr ("Unconsistant tag use: ", tagSym->name);
			tagSym->entry = tags->Insert(mk_tag(tagSym->name, result));
		}
	}
	else
	{
		tagSym->entry = tags->Insert(mk_tag(tagSym->name, result));
    
//     std::cout << "struct/union/enum_tag_ref:"
//       "There is no tag entry:"
//               << "(" << tagSym->entry << ")" << tagSym->name  << "$";
//     tagSym->entry->scope->ShowScopeId(std::cout);
//     std::cout << " has been created" << endl;
	}

	return result;
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
BaseType* ParseCtxt::Mk_tag_def (BaseTypeSpec tagType, Symbol* tagSym, SymTbl* tags)
{
	if (! tagSym)
		return NULL;
    
	BaseType* result = new BaseType(tagType);
	result->tag = tagSym;
  
	if (tagSym->entry)
	{
  
//     std::cout << "struct/union/enum_tag_ref:"
//                      "There is a previous tag entry:"
//               << "(uStructDef:" << tagSym->entry->uStructDef << ")"
//               << "(uStructDef->stDefn:" << tagSym->entry->uStructDef->stDefn << ")"
//               << "(uStructDef->enDefn:" << tagSym->entry->uStructDef->enDefn << ")"
//               << "(" << tagSym->entry << ")" << tagSym->name << "$";
//     tagSym->entry->scope->ShowScopeId(std::cout);
//     std::cout << " which have to be consistent" << endl;
        
		if (! tagSym->entry->uStructDef || tagSym->entry->uStructDef->typemask != tagType)
		{
			// ... which isn't compatible.
			yyerr ("Unconsistant tag use: ", tagSym->name);
			tagSym->entry = tags->Insert(mk_tag(tagSym->name, result));
		}
	    else if ((tagSym->entry->uStructDef->stDefn || tagSym->entry->uStructDef->enDefn)
			&& tagSym->entry->scope->level == tags->clevel)
		{
			// ... has alreay a definition at this level.
			yyerr ("struct/union/enum tag redeclared: ", tagSym->name);
			tagSym->entry = tags->Insert(mk_tag(tagSym->name, result));
		}
	}
	else
	{
		tagSym->entry = tags->Insert(mk_tag(tagSym->name, result));
    
//     std::cout << "struct/union/enum_tag_ref:"
//       "There is no tag entry:"
//               << "(" << tagSym->entry << ")" << tagSym->name  << "$";
//     tagSym->entry->scope->ShowScopeId(std::cout);
//     std::cout << " has been created" << endl;

	}

	return result;
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
Decl* ParseCtxt::Mk_direct_declarator_reentrance (Symbol* declSym, SymTbl* syms)
{
	if (! declSym)
		return NULL;
 
	if (curCtxt->possibleDuplication != NULL)
	{ 
		yyerr("Warning: Duplicate name: ", declSym->name);
	}

	Decl*	result= new Decl(declSym);
  
//   std::cout << "isTypedefCtxt=" << IsTypedefDeclCtxt()
//             << ", Var/Param=" << curCtxt->varParam
//             << ", isKnR=" << curCtxt->isKnR
//             << ", isFieldId=" << isFieldId
//             << ", Declarator=" << declSym->name
//             << ", Type Entry=" << (declSym->entry ? declSym->entry->type : -1)
//             << ", Entry level=" << (declSym->entry ? declSym->entry->scope->level : -1)
//             << ", Parsing level=" << syms->clevel
//             << std::endl;
  
	if (curCtxt->varParam == 0)
	{
		if (IsTypedefDeclCtxt())
		{
			assert(err_top_level || ! isFieldId);
 			if (declSym->entry && (declSym->entry->scope->level == syms->clevel))
				// There is a previous entry defined at the same level.
				yyerr ("Duplicate typedef name: ", declSym->name);
      
			declSym->entry = syms->Insert(mk_typedef(declSym->name, result));
		}
		else
		{
			if (declSym->entry && (declSym->entry->scope->level == syms->clevel))
			{
				if (! declSym->entry->IsFctDecl())
					yyerr ("Symbol name duplicated: ", declSym->name);
				else
					curCtxt->possibleDuplication=declSym->entry;
			}

			if (isFieldId)
				declSym->entry = syms->Insert(mk_component(declSym->name, result));
			else
				declSym->entry = syms->Insert(mk_varfctdecl(declSym->name, result));
		}
	}
	else
	{
		assert(err_top_level || ! IsTypedefDeclCtxt());

		if (curCtxt->isKnR && curCtxt->varParam == 1)
		{
			if (! (declSym->entry && (declSym->entry->scope->level == syms->clevel)))
			{
				// There is not a previous entry defined at the same level.
				yyerr ("Unknown parameter: ", declSym->name);
				declSym->entry = syms->Insert(mk_paramdecl(declSym->name, result));
			}
			else if (declSym->entry->uVarDecl->form)
			{
				assert(declSym->entry->IsParamDecl());
				yyerr("Duplicate type declaration: ", declSym->name);
			}
		}
		else if (declSym->entry && (declSym->entry->scope->level == syms->clevel))
		{
			// There is a previous entry defined at the same level.
			yyerr("Duplicate parameter name: ", declSym->name);
			declSym->entry = syms->Insert(mk_paramdecl(declSym->name, result));
		}
		else
			declSym->entry = syms->Insert(mk_paramdecl(declSym->name, result));
	}
    
//     std::cout << "isTypedefCtxt=" << IsTypedefDeclCtxt()
//               << ", Var/Param=" << curCtxt->varParam
//               << ", isKnR=" << curCtxt->isKnR
//               << ", isFieldId=" << isFieldId
//               << ", Declarator=" << declSym->name
//               << ", Type Entry=" << (declSym->entry ? declSym->entry->type : -1)
//               << ", Entry level=" << (declSym->entry ? declSym->entry->scope->level : -1)
//               << ", Parsing level=" << syms->clevel
//               << ", Duplication=" << (curCtxt->possibleDuplication ? curCtxt->possibleDuplication->name : "NO")
//               << std::endl;

	return result;
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void ParseCtxt::Mk_declarator(Decl* decl)
{
	if (! decl)
		return;
  
	Symbol* ident = decl->name;
     
	assert(! curCtxt->possibleDuplication || ident->entry);
  
	if (ident && ident->entry)
	{
		if (curCtxt->possibleDuplication)
		{
			assert(curCtxt->possibleDuplication->IsFctDecl());
			//assert(curCtxt->possibleDuplication->scope->level
			//		== gProject->Parse_TOS->transUnit->contxt.syms->clevel);
 
			if (! decl->form || (decl->form->type != TT_Function))
				yyerr ("Duplicate function name: ", ident->name);
			/*
			else
				yywarn ("TO DO: checking prototype consistency and eventually delete the new fct symbol");
			*/

			curCtxt->possibleDuplication = NULL;
		}  
                    
		if (ident->entry->type == VarFctDeclEntry)
		{
			if (decl->form && (decl->form->type == TT_Function))
				ident->entry->type = FctDeclEntry;
			else
				ident->entry->type = VarDeclEntry;
		}
	}
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
void ParseCtxt::Mk_func_declarator(Decl* decl)
{
	if (! decl)
		return;
  
	Symbol* ident = decl->name;
  
	assert(! curCtxt->possibleDuplication || ident->entry);
	assert(! err_top_level || decl->form->type == TT_Function);
  
	if (ident && ident->entry)
	{
		ident->entry->type = FctDeclEntry;
		if (curCtxt->possibleDuplication)
		{
			assert(curCtxt->possibleDuplication->IsFctDecl());
			assert(curCtxt->possibleDuplication->scope->level
					== gProject->Parse_TOS->transUnit->contxt.syms->clevel);

			if (curCtxt->possibleDuplication->u2FunctionDef)
				yyerr ("Duplicate function name: ", ident->name);
			/*
			else
				yywarn("TO DO: checking prototype consistency and eventually delete the new fct symbol");
			*/

			curCtxt->possibleDuplication = NULL;
		}
	}
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
Label* ParseCtxt::Mk_named_label (Symbol* labelSym, SymTbl* labels)
{
	if (! labelSym)
		return NULL;
        
	labelSym->entry = labels->LookupAt(labelSym->name, FUNCTION_SCOPE);
  
	Label* result = new Label(labelSym);
  
	if (labelSym->entry)
	{
		if (labelSym->entry->u2LabelPosition)
		{
			yyerr("Duplicate label: ", labelSym->name);
			labelSym->entry = mk_label(labelSym->name, result);
			labels->InsertAt(labelSym->entry, FUNCTION_SCOPE);
		}
	}
	else
	{
		labelSym->entry = mk_label(labelSym->name, result);
		labels->InsertAt(labelSym->entry, FUNCTION_SCOPE);
	}

	result->syment = labelSym->entry;
	return result;
}

/* o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
