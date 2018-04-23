
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
    o+     File:         symbol.cpp
    o+
    o+     Programmer:   Shaun Flisakowski
    o+     Date:         Aug 19, 1998
    o+
    o+     A symbol class.
    o+
    o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */

#include <cstdlib>
#include <cstring>
#include <cassert>

#include "symbol.h"
#include "stemnt.h"
#include "project.h"

//#define  PRINT_LEVEL

#ifdef PRINT_LEVEL

static int indentation = 0 ;

static void Indent(void)
{
  for (int i = indentation ; i ; i--) 
    std::cout << " " ;
}
 
static void IncrIndent(void)
{
  indentation++ ;
}
 
static void DecrIndent(void)
{
  indentation-- ;
}
 
#endif

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
static
uint
calcHash( const std::string& str )
{
    uint hsh = 0, c;
    const char* cstr = str.c_str();

    while(*cstr)
    {
        c = *cstr++;
        hsh = (hsh << 1) ^ (hsh >> 20) ^ c;
    }

    return hsh;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
SymEntry::SymEntry(SymEntryType _type)
{
    type = _type;
    uVarDecl = NULL;
    u2FunctionDef =NULL;
    next = NULL;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
SymEntry::SymEntry(SymEntryType _type, const std::string& sym, Label *labelDef)
{
    type = _type;
    name = sym;
    uLabelDef = labelDef;
    u2LabelPosition = NULL;
    next = NULL;
}

SymEntry::SymEntry(SymEntryType _type, const std::string& sym, Expression *enumVal)
{
    type = _type;
    name = sym;
    uEnumValue = enumVal;
    u2Container=NULL;
    next = NULL;
}

SymEntry::SymEntry(SymEntryType _type, const std::string& sym, Decl *varDecl)
{
    type = _type;
    name = sym;
    uVarDecl = varDecl;
    u2FunctionDef =NULL;
    next = NULL;
}

SymEntry::SymEntry(SymEntryType _type, const std::string& sym, BaseType *defn)
{
    type = _type;
    name = sym;
    uStructDef = defn;
    u2Container=NULL;
    next = NULL;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry::~SymEntry()
{
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
SymEntry::Show(std::ostream& out) const
{
    const char * typeEntry;

    switch (type)
    {
        default:
            typeEntry = "Unknown entry type";
            break;

        case TypedefEntry:
            typeEntry = "Typedef";
            break;

        case VarDeclEntry:
            typeEntry = "Variable";
            break;

        case FctDeclEntry:
            typeEntry = "Function";
            break;

        case ParamDeclEntry:
            typeEntry = "Parameter";
            break;

        case EnumConstEntry:
            typeEntry = "Enumeration constant";
            break;

        case LabelEntry:
            typeEntry = "Label";
            break;

        case TagEntry:
            typeEntry = "Tag";
            break;

        case ComponentEntry:
            typeEntry = "Component";
            break;
    }
    
    out << typeEntry << " ";

    out << "Name: " << name << ":\n";

    out << "Definition: ";

    switch (type)
    {
        default:
            out << "Unknown entry type!";
            break;

        case FctDeclEntry:
            if (uVarDecl != NULL)
            {
                uVarDecl->print(out,true,0);
                if (gProject->gDebug &&
                    uVarDecl->form)
                {
                    out << std::endl;
                    uVarDecl->form->printForm(out);
                }
            }
            else
                out << "NO definition for " << typeEntry << " entry!";
                
            if (u2FunctionDef != NULL)
            {
                out << std::endl << "Position: ";
                u2FunctionDef->location.printLocation(out);
            }
            else
                out << std::endl << "No Position.";
            break;
            
        case VarDeclEntry:
        case TypedefEntry:
        case ParamDeclEntry:
            if (uVarDecl != NULL)
            {
                uVarDecl->print(out,true,0);
                if (gProject->gDebug &&
                    uVarDecl->form)
                {
                    out << std::endl;
                    uVarDecl->form->printForm(out);
                }
            }
            else
                out << "NO definition for " << typeEntry << " entry!";
            break;

        case EnumConstEntry:
            if (uEnumValue != NULL)
                uEnumValue->print(out);
            else
                out << "NO definition for " << typeEntry << " entry!";
                
            if (u2EnumDef != NULL)
            {
                out << std::endl << "Container: ";
                u2EnumDef->print(out,NULL,0);
            }
            else
                out << std::endl << "No Container!";
            break;

        case LabelEntry:
            if (uLabelDef != NULL)
                uLabelDef->print(out,0);
            else
                out << "NO definition for " << typeEntry << " entry!";
                
            if (u2LabelPosition != NULL)
            {
                out << "Position: ";
                u2LabelPosition->location.printLocation(out);
            }
            else
                out << std::endl << "NO position!";
            break;

        case TagEntry:
            if (uStructDef != NULL)
                uStructDef->printType(out,NULL,true,0);
            else
                out << "NO definition for " << typeEntry << " entry!" ;
            break;

        case ComponentEntry:
            if (uComponent != NULL)
            {
                uComponent->print(out,true,0);
                if (gProject->gDebug &&
                    uComponent->form)
                {
                    out << std::endl;
                    uComponent->form->printForm(out);
                }
            }
            else
                out << "NO definition for " << typeEntry << " entry!" ;

            if (u2Container != NULL)
            {
                out << std::endl << "Container: ";
                u2Container->print(out,NULL,0);
            }
            else
                out << std::endl << "No Container;";
            break;
    }
    out << std::endl;
    
    out << "\n-------------\n";
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
mk_typedef(const std::string& sym, Decl *typeDef)
{
    SymEntry *ret = new SymEntry(TypedefEntry,sym,typeDef);
    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
mk_vardecl(const std::string& sym, Decl *vDecl)
{
    SymEntry *ret = new SymEntry(VarDeclEntry,sym,vDecl);
    return ret;
}

SymEntry*
mk_fctdecl(const std::string& sym, Decl *vDecl)
{
    SymEntry *ret = new SymEntry(FctDeclEntry,sym,vDecl);
    return ret;
}

SymEntry*
mk_varfctdecl(const std::string& sym, Decl *vDecl)
{
    SymEntry *ret = new SymEntry(VarFctDeclEntry,sym,vDecl);
    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
SymEntry*
mk_paramdecl(const std::string& sym, Decl *vDecl)
{
    SymEntry *ret = new SymEntry(ParamDeclEntry,sym,vDecl);
    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o */
SymEntry*
mk_component(const std::string& sym, Decl *compDef)
{
    SymEntry *ret = new SymEntry(ComponentEntry,sym,compDef);
    return ret;
}

SymEntry*
mk_component(const std::string& sym, Decl *compDef, StructDef* contain)
{
    SymEntry *ret = mk_component(sym,compDef);
    ret->u2Container = contain;
    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
mk_enum_const(const std::string& sym, Expression* enumVal)
{
    SymEntry *ret = new SymEntry(EnumConstEntry,sym,enumVal);
    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
mk_label(const std::string& sym, Label *labelDef)
{
    SymEntry *ret = new SymEntry(LabelEntry,sym,labelDef);
    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
mk_tag(const std::string& sym, BaseType *defn)
{
    SymEntry *ret = new SymEntry(TagEntry,sym,defn);
    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
HashTbl::HashTbl(int initSize /* = INIT_HASHTAB_SIZE */)
{
    nent = 0;
    tsize = initSize;
  
    tab = new SymEntry* [tsize];

    for (int j=0; j < tsize; j++)
        tab[j] = NULL;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
HashTbl::~HashTbl()
{
    for (int j=0; j < tsize; j++)
    {
        SymEntry *curr, *prev = NULL;

        for (curr=tab[j]; curr; curr = curr->next)
        {
            delete prev;
            prev = curr;
        }

        delete prev;
    }

    delete [] tab;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
HashTbl::Lookup( const std::string& sym )
{
    int j = calcHash(sym) % tsize;
 
    for (SymEntry *curr = tab[j]; curr; curr = curr->next)
        if (curr->name == sym) 
            return curr;

    return NULL;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
HashTbl::Insert( SymEntry *entry )
{
//   std::cout << "HashTbl::Insert: " << entry->name 
//             << " ->scope->level=" << entry->scope->level
//             << " ->type=" << entry->type 
//             << " Lookup(entry->name)=" << Lookup(entry->name)
//             << std::endl;

    int j = calcHash(entry->name) % tsize;
    entry->next = tab[j];
    tab[j] = entry;

    nent++;
    return entry;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
HashTbl::Delete( SymEntry *entry )
{
    int j = calcHash(entry->name) % tsize;

    SymEntry *prev = NULL;
    SymEntry *curr = tab[j];

    while ((curr != NULL) && (curr != entry))
    {
        prev = curr;
        curr = curr->next;
    }

    if (curr != NULL)
    {
        if (prev)
        {
            prev->next = entry->next;
        }
        else
        {
            tab[j] = entry->next;
        }

        entry->next = NULL;
        nent--;
    }

}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
HashTbl::Show(std::ostream& out) const
{
    out << "HashTab:  #entries: " << nent
        << "  size: " << tsize << std::endl;
  
    for (int j=0; j < tsize; j++)
    {
        SymEntry *list;

        list = tab[j];
        if (list)
        {
            out << "[" << j << "]: ";
            for (; list; list=list->next)
                list->Show(out);
            out << std::endl;
        }
    }
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
ScopeTbl::ScopeTbl( ScopeTbl *mom /* =NULL */,
                    int initSize /* = INIT_CHILD_SIZE */ )
{
    htab = NULL;
    nchild = 0;
    size = initSize;
    children = new ScopeTbl* [size];

    for (int j=0; j < size; j++)
        children[j] = NULL;

    nsyms = 0;
    parent = mom;

    if (mom)
        level = mom->level + 1;
    else
        level = EXTERN_SCOPE;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
ScopeTbl::~ScopeTbl()
{
    delete htab;

    /*
    for (int j=0; j < size; j++)
        delete children[j];
    */

    delete [] children;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
ScopeTbl::PostOrderTblDelete()
{
    if (this)
    {
        for (int j=0; j < nchild; j++)
            children[j]->PostOrderTblDelete();

        delete this;
    }
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
ScopeTbl::Lookup( const std::string& sym )
{
    SymEntry *ret = NULL;

    if (this && htab)
        ret = htab->Lookup(sym);

    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
ScopeTbl::Find( const std::string& sym )
{
    SymEntry *ret = NULL;
  
    if (this)
    {
#ifdef  PRINT_LEVEL
        IncrIndent();
        Indent();

        std::cout << "Looking for '" << sym << "' in level "
             << level << " (<entry>)\n";
#endif
  
        if (htab)
            ret = htab->Lookup(sym);
  
        if (!ret)
            ret = parent->Find(sym);
  
#ifdef  PRINT_LEVEL
        Indent();

        std::cout << "Looking for '" << sym << "' in level "
             << level << " (" << (ret ? "found" : "NOT found") << ")\n";

        DecrIndent();
#endif
    }
  
    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
ScopeTbl::Insert(SymEntry *entry)
{
#ifdef  PRINT_LEVEL
    Indent();
    std::cout << "Inserting '" << entry->name
         << "' in level " << level << " as ";

    switch (entry->type)
    { 
        case TypedefEntry:
            std::cout << "typedef";
            break;

        case VarFctDeclEntry:
            std::cout << "variable/function";
            break;

        case VarDeclEntry:
            std::cout << "variable";
            break;

        case FctDeclEntry:
            std::cout << "function";
            break;

        case ParamDeclEntry:
            std::cout << "parameter";
            break;

        case EnumConstEntry:
            std::cout << "enum const";
            break;

        case LabelEntry:
            std::cout << "label";
            break;
        case TagEntry:
            std::cout << "tag";
            break;

        case ComponentEntry:
            std::cout << "component";
            break;
    };
    std::cout << std::endl;
#endif

    if (!htab)
        htab = new HashTbl();

    nsyms++;
    entry->scope = this;
    return htab->Insert(entry);
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
bool
ScopeTbl::ChildInsert(ScopeTbl *kid)
{
    if (nchild >= size)
    {
        ScopeTbl **oldkids = children;
        size += size;
        children = new ScopeTbl* [size];
    
        if (!children)
            return false;
    
        for (int j=0; j < nchild; j++)
            children[j] = oldkids[j];
    }
  
    children[nchild++] = kid;
  
    return true;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
ScopeTbl::ShowScopeId(std::ostream& out) const
{
    if (parent)
    {
        parent->ShowScopeId(out);

        int j;
        for (j=0; parent->children[j] != this; j++)
        {
            assert(j < parent->nchild);
        }

        out << "_" << j + 1;
    }
    else
    {
        out << "1";
    }
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
ScopeTbl::Show(std::ostream& out) const
{
#ifdef DEBUG_SYMBOL_TBL
    out << "\nScopeTab (" << (void*) this
        << "): (parent: " << (void*) parent << ")\n";
#endif

    out << "level: " << level << "  nsyms: " << nsyms
        << "  nchild: " << nchild << std::endl;

    out << "Scope Id: ";
    ShowScopeId(out);
    out << std::endl;
  
    if (htab)
        htab->Show(out);
    else
        out << "HashTab: NULL\n";
  
    for (int j=0; j < nchild; j++)
        children[j]->Show(out);
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymTbl::SymTbl()
{
    root = new ScopeTbl();
    clevel = EXTERN_SCOPE;
    current = root;
    reEnter = NULL;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymTbl::~SymTbl()
{
    /* delete all ScopeTbl's (post-order traversal). */ 
    root->PostOrderTblDelete();
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
SymTbl::Lookup(const std::string& nme)
{
    SymEntry *ret = NULL;

    if (current)
        ret = current->Find(nme);

    return ret;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
SymTbl::LookupAt(const std::string& nme, int level)
{
    ScopeTbl *scp = current;

    while (scp && (scp->level < level))
        scp = scp->parent;

    return scp->Find(nme);
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
SymTbl::Insert(SymEntry *entry)
{
#ifdef PRINT_LEVEL
    Indent();
    std::cout << "SymTbl::Insert as new entry:" << entry->name << std::endl;
#endif

    if (!current)
        return (SymEntry*) NULL;

    while (clevel > current->level)
    {
        ScopeTbl *child = new ScopeTbl(current);
  
        if (!child || !current->ChildInsert(child))
            return NULL;
  
        current = child;
    }
  
    return current->Insert(entry);
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
SymEntry*
SymTbl::InsertUp(SymEntry *entry)
{
#ifdef PRINT_LEVEL
    Indent();
    std::cout << "SymTbl::InsertUp as new entry:" << entry->name << std::endl;
#endif

    InsertAt( entry, clevel - 1);
  
    return entry;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
bool
SymTbl::InsertAt(SymEntry *entry, int level)
{
    ScopeTbl *scp;

#ifdef PRINT_LEVEL
    Indent();
    std::cout << "SymTbl::InsertAt as new entry:" << entry->name << std::endl;
#endif

#ifdef  PRINT_LEVEL
    Indent();
    *gProject->Parse_TOS->yyerrstream
      << "scope level " << current->level << std::endl;
    Indent();
    *gProject->Parse_TOS->yyerrstream
      << "Current level " << clevel << std::endl;
    Indent();
    *gProject->Parse_TOS->yyerrstream
      << "request level " << level << std::endl;
#endif

    while ((clevel > current->level) && (clevel >= level))
    {
        ScopeTbl *child = new ScopeTbl(current);

        if (!child || !current->ChildInsert(child))
            return false;

        current = child;
    }

    scp = current;
    while (scp && (scp->level > level))
        scp = scp->parent;
  
    if (scp)
        return (scp->Insert(entry) != NULL);
    else
        return false;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
int
SymTbl::EnterScope()
{
    reEnter = NULL;
    return ++clevel;
}

int
SymTbl::ReEnterScope()
{
    if (reEnter)
        current = reEnter;

    return ++clevel;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
SymTbl::ExitScope( bool _reEnter /* =false */ )
{
    clevel--;

    assert(current != NULL);

    if (_reEnter)
        reEnter = current;

    if (current->level > clevel)
        current = current->parent;
    else
        reEnter = NULL;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
SymTbl::Show(std::ostream& out) const
{
    out << "\n-------------\nSymbol Table:\n";

#ifdef DEBUG_SYMBOL_TBL
    out << "current level: " << clevel
        << "  (" << (void*) current
        << ")  parent: (" << (void*) current->parent << ")\n";
#endif

    root->Show(out);

    out << "\n-------------\n";
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
Context::Context()
{
    labels = new SymTbl();
    tags   = new SymTbl();
    syms   = new SymTbl();

    // Gcc Extension, add __FUNCTION__ and __PRETTY_FUNCTION__
    // to syms, so they are considered defined and don't cause
    // warnings.  I'm adding them at external scope, even though
    // they really belong at function scope.
    std::string   gccExt("__FUNCTION__");
    syms->InsertAt(mk_vardecl(gccExt,NULL), EXTERN_SCOPE);

    gccExt = "__PRETTY_FUNCTION__";
    syms->InsertAt(mk_vardecl(gccExt,NULL), EXTERN_SCOPE);
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
Context::~Context()
{
    delete labels;
    delete tags;
    delete syms;
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
int
Context::EnterScope()
{
#ifdef  PRINT_LEVEL
    IncrIndent();
    Indent();
    std::cout << "Entering Scope: " << (syms->clevel + 1) << std::endl;
#endif
    labels->EnterScope();
    tags->EnterScope();
    return syms->EnterScope();
}

int
Context::ReEnterScope()
{
#ifdef  PRINT_LEVEL
    IncrIndent();
    Indent();
    std::cout << "ReEntering Scope: " << (syms->clevel + 1) << std::endl;
#endif
    labels->ReEnterScope();
    tags->ReEnterScope();
    return syms->ReEnterScope();
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
Context::ExitScope( bool reEnter /* =false */)
{
#ifdef  PRINT_LEVEL
    Indent();
    std::cout << "Exiting Scope" << (reEnter ? "(allow reEnter):" : ": ") << (syms->clevel) << std::endl;
    DecrIndent();
#endif
    labels->ExitScope(reEnter);
    tags->ExitScope(reEnter);
    syms->ExitScope(reEnter);
}

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
void
Context::ExitScopes(int newlev)
{
    if (newlev < EXTERN_SCOPE)
        newlev = EXTERN_SCOPE;

    if (syms->current)
    {
        while (newlev < syms->current->level)
            ExitScope();
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Symbol::Symbol()
{
    entry = NULL;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Symbol::~Symbol()
{
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
uint
Symbol::hash() const
{
    return calcHash(name);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Symbol*
Symbol::dup() const
{
    Symbol* ret = this ? dup0() : NULL;
    return ret;
}

Symbol*
Symbol::dup0() const
{
    Symbol* ret = new Symbol();

    ret->name  = name;
    ret->entry = entry;

    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
std::ostream&
operator<< ( std::ostream& out, const Symbol& sym )
{
    //static int inSymShow = 0;

    out << sym.name;

    if (sym.entry && Project::gDebug)
    {
        out << "$"; 
        // To print an identifier of the scope.
        sym.entry->scope->ShowScopeId (out);
        
        // To print an sub identifier inside the scope.
        const HashTbl * htab = sym.entry->scope->htab;
        int id1 = calcHash(sym.name) % htab->tsize;
        int id2 = 0;
        for (const SymEntry *curr = htab->tab[id1] ; curr != sym.entry; ++id2)
        {
          curr = curr->next;
          assert (curr); // the symbol have to be in this list! 
        }
        out << "$" << id1 << "_" << id2 << "$" ;

    }

    /*
    if (sym.entry && !inSymShow)
    {
        inSymShow = 1;
        out << "[";
        sym.entry->Show(out);
        out << "]";
        inSymShow = 0;
    }
    */

    return out;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

