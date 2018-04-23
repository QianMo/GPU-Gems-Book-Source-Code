
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
    o+     File:         decl.cpp
    o+
    o+     Programmer:   Shaun Flisakowski
    o+     Date:         Aug 9, 1998
    o+
    o+     A high-level view of declarations.
    o+
    o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */
#ifdef _WIN32
#pragma warning(disable:4786)
#endif
#include <cassert>
#include <cstring>
#include <sstream>

#include "decl.h"
#include "express.h"
#include "stemnt.h"

#include "token.h"
#include "gram.h"
#include "project.h"
#include "splitting/splitting.h"

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
static
void
printStorage( std::ostream& out, StorageType storage )
{
    switch (storage)
    {
        case ST_None:
            break;

        case ST_Typedef:
            out << "typedef ";
            break;

        case ST_Auto:
            out << "auto ";
            break;

        case ST_Register:
            out << "register ";
            break;

        case ST_Static:
            out << "static ";
            break;

        case ST_Extern:
            out << "extern ";
            break;
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
static
void
printQual( std::ostream& out, TypeQual qualifier )
{
// TIM: complete HACK to make the iterator test case work
// I can't figure out how to remove qualifiers from
// printType, though
//    if ((qualifier & TQ_Iter)!=0)
//        out << "iter ";

    if ((qualifier & TQ_Out)!=0)
        out << "out ";

    if ((qualifier & TQ_Const)!=0)
        out << "const ";

    if ((qualifier & TQ_Volatile)!=0)
        out << "volatile ";
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type::Type(TypeType _type /* =TT_Base */)
{
    type      = _type;
    storage  = ST_None;

    // Add us into the global list for destruction later.
    link = gProject->typeList;
    gProject->typeList = this;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type::~Type()
{
    // assert(false);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
Type::DeleteTypeList(Type* typeList)
{
    Type    *prev = NULL;
    Type    *curr = typeList;

    while (curr != NULL)
    {
	if(prev!=NULL) delete prev;
        prev = curr;
        curr = curr->link;
    }

    if(prev!=NULL) delete prev;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
Type::printType( std::ostream& out, Symbol *name, bool showBase, int level, bool raw ) const
{
    if (showBase) {
       if (raw) {
          printRawBase(out,level);
       }else {
          printBase(out,level);
       }
       if (name != NULL) out << " ";
    }
    printBefore(out,name,level);
    printAfter(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BaseType::BaseType( BaseTypeSpec bt /* =BT_NoType */ )
        : Type(TT_Base)
{
    typemask = bt;
    qualifier = TQ_None;

    tag  = NULL;
    typeName = NULL;
    stDefn = NULL;
    enDefn = NULL;
}

BaseType::BaseType( StructDef *sd )
         : Type(TT_Base)
{
    typemask = (sd->isUnion()) ? BT_Union : BT_Struct;
    qualifier = TQ_None;

    tag  = sd->tag->dup();
    
    typeName = NULL;
    stDefn = sd;
    enDefn = NULL;
}

BaseType::BaseType( EnumDef *ed )
         : Type(TT_Base)
{
    typemask = BT_Enum;
    qualifier = TQ_None;

    tag  = ed->tag->dup();
    typeName = NULL;
    stDefn = NULL;
    enDefn = ed;
}

BaseType::~BaseType()
{
    delete tag;
    delete typeName;
    delete stDefn;
    delete enDefn;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type*
BaseType::dup0() const
{
    BaseType *ret = new BaseType();

    ret->storage = storage; 
    ret->qualifier = qualifier; 
    ret->typemask = typemask; 

    ret->tag = tag->dup();
    ret->typeName = typeName->dup();
    ret->stDefn = stDefn->dup();
    ret->enDefn = enDefn->dup();
    
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

bool BaseType::printStructureStreamHelperType( std::ostream& out, const std::string& name , bool raw) const
{
    printQual(out,qualifier);

    if (typemask & BT_UnSigned)
        out << "unsigned ";
    else if (typemask & BT_Signed)
        out << "signed ";

    if (typemask & BT_Void)
        out << "void ";
    else if (typemask & BT_Char)
        out << "char ";
    else if (typemask & BT_Short)
        out << "short ";
    else if ((typemask & BT_Float)||(raw==false&&((typemask &BT_Fixed)||(typemask&BT_Half))))
       out << "__BrtFloat1 ";
    else if ((typemask & BT_Float2)||(raw==false&&((typemask &BT_Fixed2)||(typemask&BT_Half2))))
       out << "__BrtFloat2 ";
    else if ((typemask & BT_Float3)||(raw==false&&((typemask &BT_Fixed2)||(typemask&BT_Half2))))
       out << "__BrtFloat3 ";
    else if ((typemask & BT_Float4)||(raw==false&&((typemask &BT_Fixed2)||(typemask&BT_Half2))))
       out << "__BrtFloat4 ";
    else if (typemask & BT_Fixed)
        out << "fixed ";
    else if (typemask & BT_Fixed2)
        out << "fixed2 ";
    else if (typemask & BT_Fixed3)
        out << "fixed3 ";
    else if (typemask & BT_Fixed4)
        out << "fixed4 ";
    else if (typemask & BT_Half)
        out << "half ";
    else if (typemask & BT_Half2)
        out << "half2 ";
    else if (typemask & BT_Half3)
        out << "half3 ";
    else if (typemask & BT_Half4)
        out << "half4 ";
    else if ((typemask & BT_Double) && (typemask & BT_Long))
        out << "long double ";
    else if (typemask & BT_Double)
        out << "__BrtDouble ";
    else if (typemask & BT_Double2)
        out << "__BrtDouble2 ";
    else if (typemask & BT_Ellipsis)
        out << "...";
    else if (typemask & BT_Long)
        out << "long ";
    else if (typemask & BT_Struct)
    {
        if (stDefn != NULL)
        {          
           if( !stDefn->printStructureStreamHelper(out,raw) )
              return false;
        }
        else
        {
            out << "struct ";

            if (tag)
               out << (raw?"__castablestruct_":"__cpustruct_") << *tag << " ";
        }
    }
    else if (typemask & BT_Union)
    {
      return false;
    }
    else if (typemask & BT_Enum)
    {
      return false;
    }
    else if (typemask & BT_UserType)
    {
        if (typeName)
           out << (raw?"__castablestruct_":"__cpustruct_") << *typeName << " ";
    }
    else
    {
        out << "int ";        // Default
    }
    out << name;
    return true;
}

static Symbol* findStructureTag( Type* inType )
{
	BaseType* base = inType->getBase();
	while(true)
	{
		BaseTypeSpec mask = base->typemask;
		if( mask & BT_UserType )
		{
			base = base->typeName->entry->uVarDecl->form->getBase();
		}
		else if( mask & BT_Struct )
			return base->tag;
		else break;
	}
	return NULL;
}

static StructDef* findStructureDef( Type* inType )
{
	Symbol* tag = findStructureTag( inType );
	if( tag == NULL ) return NULL;
	return tag->entry->uStructDef->stDefn;
}


bool BaseType::printStructureStreamShape( std::ostream& out )
{

    if (typemask & BT_Float)
        out << "__BRTFLOAT, ";
    else if (typemask & BT_Float2)
        out << "__BRTFLOAT2, ";
    else if (typemask & BT_Float3)
        out << "__BRTFLOAT3, ";
    else if (typemask & BT_Float4)
        out << "__BRTFLOAT4, ";
    else if (typemask & BT_Double)
        out << "__BRTDOUBLE, ";
    else if (typemask & BT_Double2)
        out << "__BRTDOUBLE2, ";
    else if (typemask & BT_Fixed)
        out << "__BRTFIXED, ";
    else if (typemask & BT_Fixed2)
        out << "__BRTFIXED2, ";
    else if (typemask & BT_Fixed3)
        out << "__BRTFIXED3, ";
    else if (typemask & BT_Fixed4)
        out << "__BRTFIXED4, ";
    else if (typemask & BT_Half)
        out << "__BRTHALF, ";
    else if (typemask & BT_Half2)
        out << "__BRTHALF2, ";
    else if (typemask & BT_Half3)
        out << "__BRTHALF3, ";
    else if (typemask & BT_Half4)
        out << "__BRTHALF4, ";
    else
    {
      StructDef* s = findStructureDef(this);
      if( s == NULL ) return false;
      s->printStructureStreamShape(out);
    }
    return true;
}

void
BaseType::printBase(std::ostream& out, int level) const
{
    printQual(out,qualifier);

    if (typemask & BT_UnSigned)
        out << "unsigned ";
    else if (typemask & BT_Signed)
        out << "signed ";

    if (typemask & BT_Void)
        out << "void ";
    else if (typemask & BT_Char)
        out << "char ";
    else if (typemask & BT_Short)
        out << "short ";
    else if ((typemask & BT_Float)||(0&typemask & BT_Fixed)||(0&typemask & BT_Half))
        out << "float ";
    else if ((typemask & BT_Float2)||(0&typemask & BT_Fixed2)||(0&typemask & BT_Half2))
        out << "float2 ";
    else if ((typemask & BT_Float3)||(0&typemask & BT_Fixed3)||(0&typemask & BT_Half3))
        out << "float3 ";
    else if ((typemask & BT_Float4)||(0&typemask & BT_Fixed4)||(0&typemask & BT_Half4))
        out << "float4 ";
    else if (typemask & BT_Fixed)
        out << "fixed ";
    else if (typemask & BT_Fixed2)
        out << "fixed2 ";
    else if (typemask & BT_Fixed3)
        out << "fixed3 ";
    else if (typemask & BT_Fixed4)
        out << "fixed4 ";
    else if (typemask & BT_Half)
        out << "half ";
    else if (typemask & BT_Half2)
        out << "half2 ";
    else if (typemask & BT_Half3)
        out << "half3 ";
    else if (typemask & BT_Half4)
        out << "half4 ";
    else if ((typemask & BT_Double) && (typemask & BT_Long))
        out << "long double ";
    else if (typemask & BT_Double)
        out << "double ";//FIXME?
    else if (typemask & BT_Double2)
        out << "double2 ";//FIXME?
    else if (typemask & BT_Ellipsis)
        out << "...";
    else if (typemask & BT_Long)
        out << "long ";
    else if (typemask & BT_Struct)
    {
        if (stDefn != NULL)
        {
            stDefn->print(out, NULL, level);
        }
        else
        {
            out << "struct ";

            if (tag)
                out << *tag << " ";
        }
    }
    else if (typemask & BT_Union)
    {
        if (stDefn != NULL)
        {
            stDefn->print(out, NULL, level);
        }
        else
        {
            out << "union ";

            if (tag)
                out << *tag << " ";
        }
    }
    else if (typemask & BT_Enum)
    {
        out << "enum ";
        if (enDefn != NULL)
        {
            enDefn->print(out, NULL, level);
        }
        else
        {
            if (tag)
                out << *tag << " ";
        }
    }
    else if (typemask & BT_UserType)
    {
        if (typeName)
            out << *typeName << " ";
    }
    else
    {
        out << "int ";        // Default
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BaseType::printBefore(std::ostream& out, Symbol *name, int level ) const
{
    if (name)
    {
        out << *name;
    }
}

void
BaseType::printAfter( std::ostream& out ) const
{
}

void
BaseType::printForm(std::ostream& out) const
{
    out << "-Base";
    if (qualifier != TQ_None)
    {
        out << ":";
        printQual(out,qualifier);
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BaseType::registerComponents()
{
    if ((typemask & BT_Struct) | (typemask & BT_Union))
    {
        stDefn->registerComponents();
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool
BaseType::lookup( Symbol* sym ) const
{
    if ((typemask & BT_Struct)
        || (typemask & BT_Union))
    {
        if (stDefn != NULL)
        {
            return stDefn->lookup(sym);
        }
    }
    else if (typemask & BT_UserType)
    {
        if (typeName)
        {
            SymEntry *typeEntry = typeName->entry;

            if (typeEntry)
            {
                return typeEntry->uVarDecl->lookup(sym);
            }
        }
    }

    return false;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
TypeQual
BaseType::getQualifiers( void ) const
{
   return qualifier;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BaseType *
BaseType::getBase( void )
{
   return this;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type*
PtrType::dup0() const
{
    PtrType *ret = new PtrType(qualifier);
    
    ret->subType = subType->dup();
    ret->storage = storage; 
    
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type *
PtrType::extend(Type *extension)
{
    if (subType)
        return subType->extend(extension);
        
    subType = extension;      
    return this ;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
PtrType::printBase( std::ostream& out, int level) const
{
    if (subType)
        subType->printBase(out,level);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
PtrType::printBefore( std::ostream& out, Symbol *name, int level ) const
{
    if (subType)
    {
        subType->printBefore(out,NULL,level);
        
        bool paren = ! (subType->isPointer() || subType->isBaseType());
        
        if (paren)
            out << "(" ;
            
        out << "*" ;
        printQual(out,qualifier);
        
    }

    if (name)
    {
        out << *name;
    }
}

void
PtrType::printAfter( std::ostream& out ) const
{
    if (subType)
    {
        bool paren = ! (subType->isPointer() || subType->isBaseType());
        
        if (paren)
            out << ")" ;

        subType->printAfter(out);
    }
}

void
PtrType::printForm(std::ostream& out) const
{
    out << "-Ptr";
    if (qualifier != TQ_None)
    {
        out << ":";
        printQual(out,qualifier);
    }
    if (subType)
        subType->printForm(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
PtrType::findExpr( fnExprCallback cb )
{
    if (subType)
        subType->findExpr(cb);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool
PtrType::lookup( Symbol* sym ) const
{
    if (subType)
        return subType->lookup(sym);
    else
        return false;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
TypeQual
PtrType::getQualifiers( void ) const
{
   return qualifier;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BaseType *
PtrType::getBase( void )
{
   return subType->getBase();
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
ArrayType::~ArrayType()
{
    // Handled by deleting the global type list
    // delete subType;
    delete size;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type*
ArrayType::dup0() const
{
    ArrayType *ret  = new ArrayType(type, size->dup());

    ret->subType = subType->dup();
    ret->storage = storage;
     
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type*
ArrayType::extend(Type *extension)
{
    if (subType)
        return subType->extend(extension);
    subType = extension;
    return this ;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
ArrayType::printBase( std::ostream& out, int level) const
{
    if (subType)
        subType->printBase(out,level);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
ArrayType::printBefore( std::ostream& out, Symbol *name, int level ) const
{
    if (subType)
        subType->printBefore(out,name,level);
}

void
ArrayType::printAfter( std::ostream& out ) const
{

  if (type != TT_BrtStream) {

    out << (type == TT_Array ? "[" : "<");

    if (size)
        size->print(out);

    out << (type == TT_Array ? "]" : ">");
  }
    
  if (subType)
    subType->printAfter(out);
}

void
ArrayType::printForm(std::ostream& out) const
{
    out << (type == TT_Array ? "-Array[" : "-Stream<");
    if (size)
        size->print(out);
    out << (type == TT_Array ? "]" : ">");
    if (subType)
        subType->printForm(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
ArrayType::findExpr( fnExprCallback cb )
{
    if (subType)
        subType->findExpr(cb);

    if (size) {
       size = (cb)(size);
       size->findExpr(cb);
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool
ArrayType::lookup( Symbol* sym ) const
{
    if (subType)
        return subType->lookup(sym);
    else
        return false;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
TypeQual
ArrayType::getQualifiers( void ) const
{
   return subType->getQualifiers();
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BaseType *
ArrayType::getBase( void )
{
   return subType->getBase();
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BitFieldType::~BitFieldType()
{
    // Handled by deleting the global type list
    // delete subType;
    delete size;
}

Type *
BitFieldType::extend(Type *extension)
{
    if (subType)
        return subType->extend(extension);
    
    subType = extension;
    return this;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type*
BitFieldType::dup0() const
{
    BitFieldType *ret = new BitFieldType(size->dup());
    ret->storage = storage; 
    
    ret->subType = subType->dup();
    
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BitFieldType::printBase( std::ostream& out, int level) const
{
    if (subType)
    {
        subType->printBase(out,level);
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BitFieldType::printBefore( std::ostream& out, Symbol *name, int level ) const
{
    if (subType)
    {
        subType->printBefore(out,NULL,level);
    }

    if (name)
    {
        out << *name;
    }

    out << ":";

    if (size)
    {
        size->print(out);
    }
}

void
BitFieldType::printAfter( std::ostream& out ) const
{
}

void
BitFieldType::printForm(std::ostream& out) const
{
    out << "-BitField";
    if (subType)
        subType->printForm(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
BitFieldType::findExpr( fnExprCallback cb )
{
   if (size) {
      size = (cb)(size);
      size->findExpr(cb);
   }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool
BitFieldType::lookup( Symbol* sym ) const
{
    if (subType)
        return subType->lookup(sym);
    else
        return false;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
TypeQual
BitFieldType::getQualifiers( void ) const
{
   return subType->getQualifiers();
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BaseType *
BitFieldType::getBase( void )
{
   return subType->getBase();
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
FunctionType::FunctionType(Decl *args_list)
  : Type(TT_Function), KnR_decl(false), nArgs(0), 
    size(0), args(NULL), subType(NULL)
{
   addArgs (args_list);
}
        
FunctionType::~FunctionType()
{
    for (unsigned int j=0; j < nArgs; j++)
    {
        delete args[j];
    }

    delete [] args;

    // Handled by deleting the global type list
    // delete subType;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type*
FunctionType::dup0() const
{
    FunctionType *ret = new FunctionType();
    ret->storage = storage; 
    ret->size    = size;
    ret->args = new Decl* [size];
    ret->KnR_decl = KnR_decl;

    for (unsigned int j=0; j < nArgs; j++)
        ret->addArg(args[j]->dup());

    ret->subType = subType->dup();
    
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type*
FunctionType::extend(Type *extension)
{
    if (subType)
        return subType->extend(extension);
    
    subType = extension;
    return this;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
FunctionType::printBase( std::ostream& out, int level) const
{
    if (subType)
    {
        subType->printBase(out,level);
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
FunctionType::printBefore( std::ostream& out, Symbol *name, int level ) const
{
    if (subType)
    {
        subType->printBefore(out,name,level);
    }
    else if (name)
    {
        out << *name;
    }
}

void
FunctionType::printAfter( std::ostream& out ) const
{
    if (KnR_decl)
    {
        out << "(";

        if (nArgs > 0)
        {
            out << *(args[0]->name);
            for (unsigned int j=1; j < nArgs; j++)
            {
                out << ", ";
                out << *(args[j]->name);
            }
        }
    
        out << ")\n";

        for (unsigned int j=0; j < nArgs; j++)
        {
            args[j]->print(out,true);
            out << ";\n";
        }
    }
    else
    {
        out << "(";
    
        if (nArgs > 0)
        {
            args[0]->print(out,true);
            for (unsigned int j=1; j < nArgs; j++)
            {
                out << ", ";
                args[j]->print(out,true);
            }
        }
    
        out << ")";
    }

    if (subType)
    {
        subType->printAfter(out);
    }
}

void
FunctionType::printForm(std::ostream& out) const
{
    out << "-Function";
    if (subType)
        subType->printForm(out);
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
FunctionType::addArg(Decl *arg)
{
    if (size == nArgs)
    {
        if (size == 0)
            size = 4;
        else
            size += size;

        Decl   **oldArgs = args;

        args = new Decl* [size];

        for (unsigned int j=0; j < nArgs; j++)
        {
            args[j] = oldArgs[j];
        }

        delete [] oldArgs;
    }

    args[nArgs] = arg;
    nArgs++;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
FunctionType::addArgs(Decl *args)
{
    Decl *arg = args;

    while (args != NULL)
    {
        args = args->next;
        arg->next = NULL;
        addArg(arg);
        arg = args;
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
FunctionType::findExpr( fnExprCallback cb )
{
    if (subType)
        subType->findExpr(cb);

    for (unsigned int j=0; j < nArgs; j++)
    {
        args[j]->findExpr(cb);
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool
FunctionType::lookup( Symbol* sym ) const
{
    if (subType)
        return subType->lookup(sym);
    else
        return false;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
TypeQual
FunctionType::getQualifiers( void ) const
{
   return subType->getQualifiers();
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
BaseType *
FunctionType::getBase( void )
{
   return subType->getBase();
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
StructDef::StructDef( bool _is_union /* =false */ )
{
    _isUnion = _is_union;
    tag = NULL;
    size = 0;
    nComponents = 0;

    components = NULL;
}

StructDef::~StructDef()
{
    delete tag;

    for (int j=0; j < nComponents; j++)
    {
        delete components[j];
    }

    delete [] components;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Decl*
StructDef::lastDecl()
{
    return components[nComponents-1];
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
StructDef*
StructDef::dup() const
{
    StructDef *ret = this ? dup0() : NULL;
    return ret;
}

StructDef*
StructDef::dup0() const
{
    StructDef *ret = new StructDef();
    ret->size    = size;
    ret->_isUnion = _isUnion;
    ret->components = new Decl* [size];

    for (int j=0; j < nComponents; j++)
        ret->addComponent(components[j]->dup());

    ret->tag = tag->dup();
    
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
StructDef::print(std::ostream& out, Symbol *name, int level) const
{
    if (isUnion())
        out << "union ";
    else
        out << "struct ";

    if (tag)
        out << *tag << " ";

    out << "{\n"; 

    for (int j=0; j < nComponents; j++)
    {
        indent(out,level+1);
        components[j]->print(out,true,level+1);

        Decl *decl = components[j]->next;
        while (decl != NULL)
        {
            out << ", ";
            decl->print(out,false,level+1);
            decl = decl->next;
        }

        out << ";\n";
    }

    indent(out,level);
    out << "}"; 

    if (name)
        out << " " << *name;
}

bool StructDef::printStructureStreamHelper(std::ostream& out, bool raw) const
{
    if (isUnion())
        out << "union ";
    else
        out << "struct ";

    if (tag)
       out << (raw?"__castablestruct_":"__cpustruct_") << *tag << " ";

    out << "{\n"; 

    for (int j=0; j < nComponents; j++)
    {
       if(!components[j]->printStructureStreamInternals(out,raw))
          return false;

        Decl *decl = components[j]->next;
        while (decl != NULL)
        {
            out << ", ";
            if(!decl->printStructureStreamInternals(out,raw))
              return false;
            decl = decl->next;
        }

        out << ";\n";
    }

    out << "}\n";
    return true;
}

bool StructDef::printStructureStreamShape(std::ostream& out)
{
    for (int j=0; j < nComponents; j++)
    {
        if(!components[j]->printStructureStreamShapeInternals(out))
          return false;

        Decl *decl = components[j]->next;
        while (decl != NULL)
        {
            if(!decl->printStructureStreamShapeInternals(out))
              return false;
            decl = decl->next;
        }
    }
    return true;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
StructDef::findExpr( fnExprCallback cb )
{
    for (int j=0; j < nComponents; j++)
    {
        components[j]->findExpr(cb);
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
StructDef::addComponent(Decl *comp)
{
    if (size == nComponents)
    {
        if (size == 0)
            size = 4;
        else
            size += size;

        Decl **oldComps = components;

        components = new Decl* [size];

        for (int j=0; j < nComponents; j++)
        {
            components[j] = oldComps[j];
        }

        delete [] oldComps;
    }

    components[nComponents] = comp;
    nComponents++;

    do
    {
        // Hook this component's symtable entry back to here.
        if ((comp->name != NULL) && (comp->name->entry != NULL))
        {
            comp->name->entry->uComponent = comp;
            comp->name->entry->u2Container = this;
            // The entry was inserted by gram.y as a ComponentEntry.
            assert (comp->name->entry->IsComponentDecl()) ;
        }
        comp = comp->next ;
    } while (comp) ;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
StructDef::registerComponents()
{
    int    j;

    for (j=0; j < nComponents; j++)
    {
        Decl    *decl = components[j];

        while (decl != NULL)
        {
            Symbol *ident = decl->name;

            if (ident != NULL)
            {
               ident->entry = gProject->Parse_TOS->transUnit->contxt.syms
                    ->Insert(mk_component(ident->name,decl,this));
            }

            // Register any sub-components also.
            decl->form->registerComponents();

            decl = decl->next;
        }
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool
StructDef::lookup( Symbol* sym ) const
{
    int    j;

    for (j=0; j < nComponents; j++)
    {
        Decl    *decl = components[j];

        while (decl != NULL)
        {
            Symbol *ident = decl->name;

            if (ident->name == sym->name)
            {
                sym->entry = ident->entry;

                return true;
            }

            decl = decl->next;
        }
    }

    return false;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
EnumDef::EnumDef()
{
    tag = NULL;
    size = 0;
    nElements = 0;

    names = NULL;
    values = NULL;
}

EnumDef::~EnumDef()
{
    delete tag;

    for (int j=0; j < nElements; j++)
    {
        delete names[j];
        delete values[j];
    }

    delete [] names;
    delete [] values;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
EnumDef::addElement(Symbol *nme, Expression *val /* =NULL */ )
{
    if (size == nElements)
    {
        if (size == 0)
            size = 4;
        else
            size += size;

        Symbol     **oldNames = names;
        Expression **oldVals = values;

        names  = new Symbol* [size];
        values = new Expression* [size];

        for (int j=0; j < nElements; j++)
        {
            names[j]  = oldNames[j];
            values[j] = oldVals[j];
        }

        delete [] oldNames;
        delete [] oldVals;
    }

    names[nElements]  = nme;
    values[nElements] = val;
    nElements++;

    if ((nme->entry != NULL) && (nme->entry->type == EnumConstEntry))
    {
        assert(nme->entry->type == EnumConstEntry);
        nme->entry->uEnumValue = val;
        nme->entry->u2EnumDef = this;
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
EnumDef::addElement( EnumConstant* ec )
{
    addElement(ec->name, ec->value);

    ec->name = NULL;
    ec->value = NULL;

    delete ec;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
EnumDef*
EnumDef::dup() const
{
    EnumDef *ret = this ? dup0() : NULL;
    return ret;
}

EnumDef*
EnumDef::dup0() const
{
    EnumDef *ret = new EnumDef();
    ret->size  = size;
    ret->names = new Symbol* [size];
    ret->values = new Expression* [size];

    for (int j=0; j < nElements; j++)
    {
        Expression *val = values[j]->dup();
        ret->addElement(names[j]->dup(),val);
    }

    ret->tag = tag->dup();
    
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
EnumDef::print(std::ostream& out, Symbol*, int level) const
{
    if (tag)
	{
        out << *tag << " ";
	}

   	out << "{ "; 

    if (nElements > 0)
    {
        out << *names[0];

        if (values[0])
        {
            out << " = ";
            values[0]->print(out);
        }

        for (int j=1; j < nElements; j++)
        {
            out << ", ";
            out << *names[j];

            if (values[j])
            {
                out << " = ";
                values[j]->print(out);
            }
        }
    }

    out << " }"; 
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
EnumDef::findExpr( fnExprCallback cb )
{
    for (int j=0; j < nElements; j++)
    {
       if (values[j] != NULL) {
          values[j] = (cb)(values[j]);
          values[j]->findExpr(cb);
       }
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
GccAttrib::GccAttrib( GccAttribType gccType )
{
    type = gccType;
    value = 0;
    mode = NULL;
}

GccAttrib::~GccAttrib()
{
    delete mode;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
GccAttrib::print( std::ostream& out ) const
{
    out << " __attribute__ ((";

    switch (type)
    {
        case GCC_Aligned:
            out << "aligned (" << value << ")";
            break;

        case GCC_Packed:
            out << "packed";
            break;

        case GCC_CDecl:
            out << "__cdecl__";
            break;

        case GCC_Mode:
            out << "__mode__ (" << *mode << ")";
            break;
   
        case GCC_Format:
            out << "format (" << *mode << "," << strIdx << "," << first << ")";
            break;

        case GCC_Const:
            out << "__const__";
            break;

        case GCC_NoReturn:
            out << "__noreturn__";
            break;

        case GCC_Unsupported:
        default:
            out << "<unsupported gcc attribute>";
            break;
    }

    out << "))";
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
GccAttrib*
GccAttrib::dup() const
{
    GccAttrib *ret = this ? dup0() : NULL;
    return ret;
}

GccAttrib*
GccAttrib::dup0() const
{
    GccAttrib    *ret = new GccAttrib(type);
    ret->value = value;
    ret->mode = mode->dup();
      
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Decl::Decl( Symbol* sym /* =NULL */ )
{
    clear();

    name = sym;
}

Decl::Decl( Type* type )
{
    clear();

    form = type;
    storage = type->storage;
    name = NULL;
}
BaseType& BaseType::operator = (const BaseType& b) {
  this->typemask=b.typemask;
  this->qualifier = b.qualifier;
  this->tag=b.tag?b.tag->dup():NULL;
  this->typeName=b.typeName?b.typeName->dup():NULL;
  this->stDefn=b.stDefn?b.stDefn->dup():NULL;
  this->enDefn=b.enDefn?b.enDefn->dup():NULL;
  return *this;
}
// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Decl::~Decl()
{
    // Handled by deleting the global type list
    // delete form;
    delete attrib;
    delete initializer;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
Decl::clear()
{
    storage    = ST_None;

    name        = NULL;
    form        = NULL;
    attrib      = NULL;
    initializer = NULL;
    next        = NULL;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Type*
Decl::extend( Type* type )
{
    if (storage == ST_None)
        storage = type->storage;
    
    if (form != NULL)
        return form->extend(type);
    
    form = type;
    return NULL;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Decl*
Decl::dup() const
{
    Decl *ret = this ? dup0() : NULL;
    return ret;
}

Decl*
Decl::dup0() const
{
    Decl *ret = new Decl();
    ret->storage    = storage;
    ret->form = form->dup();//do copy so that cpu implementation can be above

    ret->name     = name->dup();
    ret->attrib = attrib->dup();
    ret->initializer = initializer->dup();
    ret->next = next->dup(); 
    
    return ret;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
Decl::copy(const Decl& decl)
{
    storage     = decl.storage;
    name        = decl.name;
    form        = decl.form;
    attrib      = decl.attrib;
    initializer = decl.initializer;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
Decl::print(std::ostream& out, bool showBase, int level) const
{
    assert(this != NULL);

    if (showBase)
    {
        printStorage(out,storage);

        // Hack to fix K&R non-declarations
        if (form == NULL)
        {
            out << "int ";
        }
    }

    if (form)
    {
       form->printType(out,name,showBase,level,false);
    }
    else if (name)
    {
        out << *name;
    }

    if (attrib)
    {
        attrib->print(out);
    }

    if (initializer)
    {
        out << " = ";

        initializer->print(out);
    }

    /*
    if (!form->isBaseType())
    {
        out << "  [FORM: ";
        form->printForm(out);
        out << " ]";
    }
    */
}
static std::string PrintCastToBody(StructDef *str) {
   if (!str) {
      return "    return *this;";
   }
   std::string ret="    T ret;\n"; 
   for (int i=0;i<str->nComponents;++i) {
      ret+="    ret."+str->components[i]->name->name+" = this->"+str->components[i]->name->name+".castToArg(ret."+str->components[i]->name->name+");\n";
   }
   return ret+="    return ret;";
}
void Decl::printStructureStreamHelpers( std::ostream& out ) const
{
    assert(this != NULL);

    if(!isTypedef()) return;


    for (int i=0;i<2;++i) {
       std::ostringstream stringout;
       stringout << "\ntypedef ";
       if(!form->printStructureStreamHelperType( stringout, (i==0?std::string("__cpustruct_"):std::string("__castablestruct_")) + name->name,i!=0 ))
          return;
       stringout << ";\n";
       std::string tmp= stringout.str();
       std::string::value_type where= tmp.find("{");
       if (where!=std::string::npos) {
          tmp = tmp.substr(0,where+1)+"\n  template <typename T> T castToArg(const T& dummy)const{\n"+PrintCastToBody(form->getBase()->stDefn)+"\n  }\n"+tmp.substr(where+1);
       }
       out << tmp;
    }
}

bool Decl::printStructureStreamInternals( std::ostream& out, bool raw ) const
{
    assert(this != NULL);

    if (true)
    {
        printStorage(out,storage);

        // Hack to fix K&R non-declarations
        if (form == NULL)
        {
            out << "int ";
        }
    }

    if (form)
    {
       if(!form->printStructureStreamHelperType(out,name->name,raw))
          return false;
    }
    else if (name)
    {
        out << *name;
    }

    if (attrib)
    {
        attrib->print(out);
    }

    if (initializer)
    {
        out << " = ";

        initializer->print(out);
    }

    return true;
}

bool Decl::printStructureStreamShape(std::ostream& out)
{
  assert(this != NULL);
  if(!isTypedef()) return false;

  std::ostringstream stringout;

  stringout << "\nnamespace brook {\n";
  stringout << "\ttemplate<> const StreamType* getStreamType(";
  stringout << name->name << "*) {\n";
  stringout << "\t\tstatic const StreamType result[] = {";
  if(!form->printStructureStreamShape( stringout ))
    return false;
  stringout << "__BRTNONE};\n";
  stringout << "\t\treturn result;\n";
  stringout << "\t}\n}\n";
  out << stringout.str();
  return true;
}

bool Decl::printStructureStreamShapeInternals(std::ostream& out)
{
  if(isTypedef()) return true;
  if((storage & ST_Static) != 0) return true;
  
  if (form)
  {
    if(!form->printStructureStreamShape(out))
      return false;
  }
  return true;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
void
Decl::findExpr( fnExprCallback cb )
{
    if (form)
        form->findExpr(cb);

    if (initializer != NULL) {
       initializer = (cb)(initializer);
       initializer->findExpr(cb);
    }
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
// TIM: adding DAG-building for kernel splitting support
void Decl::buildSplitTree( SplitTreeBuilder& ioBuilder )
{
  std::string nameString = name->name;

  SplitNode* value = NULL;
  if( initializer )
    value = initializer->buildSplitTree( ioBuilder );

  if( form )
    ioBuilder.addVariable( nameString, form );

  if( value )
    ioBuilder.assign( nameString, value->getValueNode() );
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
bool
Decl::lookup( Symbol* sym ) const
{
    if (form)
        return form->lookup(sym);
    else
        return false;
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
Decl*
ReverseList( Decl* dList )
{
    Decl*    head = NULL;

    while (dList != NULL)
    {
        Decl*    dcl = dList;

        dList = dList->next;

        dcl->next = head;
        head = dcl;
    }

    return head; 
}

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
