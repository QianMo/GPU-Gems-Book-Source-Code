
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
    o+     File:         symbol.h
    o+
    o+     Programmer:   Shaun Flisakowski
    o+     Date:         Aug 15, 1998
    o+
    o+     A symbol class.
    o+
    o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */

#ifndef    SYMBOL_H
#define    SYMBOL_H

#include <cstdlib>
#include <string>
#include <iostream>

#include "utype.h"

/*  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o 

    SymTab:  A tree of scope tables,
             one for each scope that declares something, or has a
             child scope that declares something:

         Level
           1              external
                         /        \
           2          file        file 
                      scope       scope 
                                /       \
           3              prototype    function
                            scope        scope
                                        /     \
           4                         block    block
                                     scope    scope    
                                                 \
           5                                     block
                                                 scope    
                                                    \
                                                   (etc.)

    At any particular point you can see all the symbols 
    (variables, types, functions, etc) declared above you in the tree.

    The scope tables are recreated in a lazy fashion, so entering
    and exiting scopes that don't declare new symbols is cheap.

    o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */

extern bool    gShowScopeId;

//  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

// Different kinds of entries in the symbol table.

enum SymEntryType
{
    // These things are all in the same namespace.
    VarFctDeclEntry,       // temporary SymEntryType for variable/function.
    TypedefEntry,          // type definition.
    VarDeclEntry,          // variable declaration.
    FctDeclEntry,          // function declaration.
    ParamDeclEntry,        // parameter declaration.
    EnumConstEntry,        // enum constant.
    ComponentEntry,        // components of a struct/union.

    // These things are in seperate namespaces.
    LabelEntry,            // label definition.
    TagEntry               // struct/union/enum tags

};

// These might be convienent.
const int  CURRENT_SCOPE  = 0;    // the default
const int  EXTERN_SCOPE   = 1;
const int  FILE_SCOPE     = 2;
const int  FUNCTION_SCOPE = 3;
const int  BLOCK_SCOPE    = 4;    // really, 4 or more.

//  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

class FunctionDef;
class StructDef;
class EnumDef;
class Decl;
class Expression;
class Statement;
class ScopeTbl;
class Label;   
class BaseType;

class SymEntry
{
  public:
    SymEntry(SymEntryType _type);
    SymEntry(SymEntryType _type, const std::string& sym, Label *labelDef);
    SymEntry(SymEntryType _type, const std::string& sym, Expression *enumVal);
    SymEntry(SymEntryType _type, const std::string& sym, Decl *varDecl);
    SymEntry(SymEntryType _type, const std::string& sym, BaseType *defn);
   ~SymEntry();

    bool    IsTypeDef() const { return (type == TypedefEntry); }
    bool    IsVarDecl() const { return (type == VarDeclEntry); }
    bool    IsFctDecl() const { return (type == FctDeclEntry); }
    bool    IsVarFctDecl() const { return (type == VarFctDeclEntry); }
    bool    IsParamDecl() const { return (type == ParamDeclEntry); }
    bool    IsEnumConst() const { return (type == EnumConstEntry); }
    bool    IsComponentDecl() const { return (type == ComponentEntry); }

    bool    IsTagDecl() const { return (type == TagEntry); }
    bool    IsLabelDecl() const { return (type == LabelEntry); }
    
    SymEntryType    SymType() { return type; }

    void    Show(std::ostream& out) const;

  public:
    SymEntryType    type;    // what kind of entry this is.
    std::string     name;    // The name of the thing.

    
    // Pointer to it's definition/declaration.
    union
    {
        Decl*           uVarDecl;        // Also used by typedef
        Expression*     uEnumValue;

        Label*          uLabelDef;

        BaseType*       uStructDef;      // enum/struct/union

        Decl*           uComponent;
        
    };

    union
    {
        // For a label, this points to the its definition.
        Statement*      u2LabelPosition;
        
        // For a function, this points to its definition.
        FunctionDef*    u2FunctionDef;
        
        // For a struct/union component, this points to the definition
        // of the container.
        StructDef*      u2Container;
        
        // For a enum component, this points to the definition
        // of the container.
        EnumDef*        u2EnumDef;        
    };
    // The scope this symbol is defined at.
    ScopeTbl       *scope;


    // This would probably be a good place to add more attributes, 

    SymEntry       *next;
};

SymEntry  *mk_typedef   (const std::string& sym, Decl *);
SymEntry  *mk_vardecl   (const std::string& sym, Decl *);
SymEntry  *mk_fctdecl   (const std::string& sym, Decl *);
SymEntry  *mk_varfctdecl(const std::string& sym, Decl *);
SymEntry  *mk_paramdecl (const std::string& sym, Decl *);
SymEntry  *mk_enum_const(const std::string& sym, Expression* enumVal);
SymEntry  *mk_label     (const std::string& sym, Label *labelDef);
SymEntry  *mk_component (const std::string& sym, Decl *, StructDef*);
SymEntry  *mk_component (const std::string& sym, Decl *);
SymEntry  *mk_tag       (const std::string& sym, BaseType *);

//  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

#define    INIT_HASHTAB_SIZE    (5)

class HashTbl
{
  public:
    HashTbl(int initSize = INIT_HASHTAB_SIZE);
   ~HashTbl();

    SymEntry   *Lookup( const std::string& );
    SymEntry   *Insert( SymEntry* );
    void        Delete( SymEntry* );

    void Show(std::ostream& out) const;

  public:
    int            tsize;     // The current size of the table.
    int            nent;      // The number of entries being stored.
    SymEntry     **tab;       // The table.
};

//  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

#define    INIT_CHILD_SIZE     4

class ScopeTbl
{
  public:
    ScopeTbl( ScopeTbl *mom = NULL, int initSize = INIT_CHILD_SIZE );
   ~ScopeTbl();

    SymEntry   *Lookup( const std::string& );
    SymEntry   *Find( const std::string& );
    SymEntry   *Insert( SymEntry* );

    void        Show(std::ostream& out) const;
    void        ShowScopeId(std::ostream& out) const;

    void        PostOrderTblDelete();
    bool        ChildInsert(ScopeTbl *kid);

  public:
    int               nsyms; // The num of syms declared at this scope.
    int               level; // This ScopeTbl's scoping level.

    HashTbl          *htab;  // A hash table - to store the symbols.

    ScopeTbl         *parent; // The scope enclosing us, if any.

    // A doubling array of scopes we enclose.
    int               nchild;
    int               size;
    ScopeTbl        **children;
};

//  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

class SymTbl
{
  public:
    SymTbl();
   ~SymTbl();

    SymEntry* Lookup( const std::string& );
    SymEntry* LookupAt( const std::string& , int level );
    SymEntry* Insert( SymEntry* );
    SymEntry* InsertUp( SymEntry* );
    bool      InsertAt( SymEntry*, int level );

    bool      IsDefined( const std::string& sym )
                    { return (LookupAt(sym,clevel) != NULL); }
    bool      IsDefinedAt( const std::string& sym, int level ) 
                    { return (LookupAt(sym,level) != NULL); }
    bool      IsDefinedUp( const std::string& sym )
                    { return (LookupAt(sym,clevel-1) != NULL); }

    int       EnterScope();
    int       ReEnterScope();
    void      ExitScope( bool reEnter =false );

    void      Show(std::ostream& out) const;

  public:
    ScopeTbl        *root;    // The top scopetab - external scope.

    int              clevel;  // The current level.
    ScopeTbl        *current; // The current scopetab, or one of its
                              // ancestors, if it doesn't exist yet.
    ScopeTbl        *reEnter; // A scope we can reEnter.
};


//  o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o 

class Context
{
  public:
    Context();
   ~Context();

    int     EnterScope();
    int     ReEnterScope();
    void    ExitScope( bool reEnter =false );
    void    ExitScopes( int newlev );

  public:
    SymTbl     *labels;    // Statement labels.
    SymTbl     *tags;      // Struct/Union/Enum tags.
    SymTbl     *syms;      // Vars, Types, Functions, etc.
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class Symbol
{
  public:
    Symbol();
   ~Symbol();

    Symbol* dup0() const;
    Symbol* dup() const;
    uint    hash() const;

    std::string   name;
    SymEntry*     entry;

  friend std::ostream& operator<< ( std::ostream& out, const Symbol& sym );
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

#endif  /* SYMBOL_H */

