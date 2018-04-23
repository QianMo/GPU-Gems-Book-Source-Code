
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
    o+     File:         decl.h
    o+
    o+     Programmer:   Shaun Flisakowski
    o+     Date:         Aug 9, 1998
    o+
    o+     A high-level view of types / declarations.
    o+
    o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o  */

#ifndef    DECL_H
#define    DECL_H

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cassert>

#include "symbol.h"
#include "callback.h"
#include "location.h"

#include "dup.h"

class Constant;
class Expression;
class EnumConstant;
class SplitTreeBuilder;

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

typedef unsigned long BaseTypeSpec;

const BaseTypeSpec BT_NoType       = 0x00000000;  // no type provided
const BaseTypeSpec BT_Void         = 0x00000001;  // explicitly no type
const BaseTypeSpec BT_Char         = 0x00000002;
const BaseTypeSpec BT_Short        = 0x00000004;
const BaseTypeSpec BT_Int          = 0x00000008;
const BaseTypeSpec BT_Double       = 0x00000010;
const BaseTypeSpec BT_Double2      = 0x00000020;
const BaseTypeSpec BT_Long         = 0x00000040;
const BaseTypeSpec BT_Ellipsis     = 0x00000080;

/*
 * These have to be adjacent and have these values for the math that parses
 * float2, float3, float4 in lexer.l to work.
 */
const BaseTypeSpec BT_Float        = 0x00000100;
const BaseTypeSpec BT_Float2       = 0x00000200;
const BaseTypeSpec BT_Float3       = 0x00000400;
const BaseTypeSpec BT_Float4       = 0x00000800;

const BaseTypeSpec BT_Struct       = 0x00001000;
const BaseTypeSpec BT_Union        = 0x00002000;
const BaseTypeSpec BT_Enum         = 0x00004000;
const BaseTypeSpec BT_UserType     = 0x00008000;

const BaseTypeSpec BT_Fixed        = 0x00010000;
const BaseTypeSpec BT_Fixed2       = 0x00020000;
const BaseTypeSpec BT_Fixed3       = 0x00040000;
const BaseTypeSpec BT_Fixed4       = 0x00080000;

const BaseTypeSpec BT_Half        = 0x00100000;
const BaseTypeSpec BT_Half2       = 0x00200000;
const BaseTypeSpec BT_Half3       = 0x00400000;
const BaseTypeSpec BT_Half4       = 0x00800000;

const BaseTypeSpec BT_BaseMask     = 0x00FFFFFF;


// Sign indicator
const BaseTypeSpec BT_Signed       = 0x01000000;
const BaseTypeSpec BT_UnSigned     = 0x02000000;
const BaseTypeSpec BT_SignMask     = 0x03000000;

const BaseTypeSpec BT_TypeError    = 0x10000000;

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
struct TypeQual{
   unsigned short tq;
   Expression * vout;
   bool operator == (unsigned int x)const {return x==tq;}
   bool operator == (const TypeQual &x)const {return x.tq==tq;}
   bool operator !=(const TypeQual &x)const {return tq!=x.tq;}
   bool operator !=(const unsigned int x) const {return tq!=x;}

   void init (unsigned short t){tq=t;vout=NULL;}
   void init (unsigned short t,Expression * e){tq=t;vout=e;}
   TypeQual operator &(unsigned int x)const {TypeQual ret(*this);ret&=x;return ret;}
   TypeQual operator |(unsigned int x)const {TypeQual ret(*this);ret|=x;return ret;}
   TypeQual operator &(const TypeQual &x)const {TypeQual ret(*this);ret&=x;return ret;}
   TypeQual operator |(const TypeQual &x)const {TypeQual ret(*this);ret|=x;return ret;}
   TypeQual& operator |=(unsigned int x){tq|=x;return *this;}
   TypeQual& operator &=(unsigned int x){tq&=x;return *this;}
   TypeQual operator ~()const {
      TypeQual ret(*this);
      ret.tq=~ret.tq;
      return ret;
   }
   TypeQual& operator |=(const TypeQual &x){
      tq|=x.tq;
      if (x.vout&&!vout) {
         vout=x.vout;
      }
      return *this;
   }
   TypeQual& operator &=(const TypeQual &x){
      tq&=x.tq;
      return *this;
   }

   static TypeQual ReturnNone() {TypeQual ret;ret.init(0x0000);return ret;}
   static TypeQual ReturnConst() {TypeQual ret;ret.init(0x0001);return ret;}
   static TypeQual ReturnVolatile() {TypeQual ret;ret.init(0x0002);return ret;}
   static TypeQual ReturnOut() {TypeQual ret;ret.init(0x0004);return ret;}
   static TypeQual ReturnReduce() {TypeQual ret;ret.init(0x0008);return ret;}
   static TypeQual ReturnVout() {TypeQual ret;ret.init(0x0010);return ret;}
   static TypeQual ReturnIter() {TypeQual ret;ret.init(0x0020);return ret;}
   static TypeQual ReturnKernel() {TypeQual ret;ret.init(0x0040);return ret;}
};

const TypeQual TQ_None=TypeQual::ReturnNone();
const TypeQual TQ_Const=TypeQual::ReturnConst();
const TypeQual TQ_Volatile=TypeQual::ReturnVolatile();
const TypeQual TQ_Out=TypeQual::ReturnOut();
const TypeQual TQ_Reduce=TypeQual::ReturnReduce();
const TypeQual TQ_Vout=TypeQual::ReturnVout();
const TypeQual TQ_Iter=TypeQual::ReturnIter();
const TypeQual TQ_Kernel=TypeQual::ReturnKernel();

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
enum StorageType
{
    ST_None  = 0,
    ST_Auto,
    ST_Extern,
    ST_Register,
    ST_Static,
    ST_Typedef,
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
enum TypeKind
{
    TK_Base = 0,
    TK_TypeDef,
    TK_UserType
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
enum TypeType
{
    TT_Base,        // a simple base type, T
    TT_Pointer,     // pointer to T
    TT_Array,       // an array of T
    TT_Stream,      // a stream / iter of T
    TT_BitField,    // a bitfield
    TT_Function,    // <args> -> <result>

    TT_BrtStream,   // Converted stream declaration
    TT_BrtIter,     // Converted iter declaration


    /*    These are now considered TT_Base
    TT_Struct,
    TT_Union,
    TT_Enum
    */
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
// For gcc extension __attribute__
enum GccAttribType
{
    GCC_Unsupported = 0,    // Not supported by cTool

    GCC_Aligned,
    GCC_Packed,
    GCC_CDecl,
    GCC_Mode,
    GCC_Format,
   
    GCC_Const,
    GCC_NoReturn
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

class Decl;

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class StructDef
{
  public:
    StructDef( bool _is_union = false );
    ~StructDef();

    bool    isUnion() const { return _isUnion; }

    StructDef*   dup0() const;
    StructDef*   dup() const;    // deep-copy

    void    print(std::ostream& out, Symbol *name, int level) const;
    bool printStructureStreamHelper(std::ostream& out, bool raw) const;
    bool printStructureStreamShape(std::ostream& out);

    void    findExpr( fnExprCallback cb );

    void    addComponent(Decl *comp);

    void    registerComponents();

    // Lookup this symbol in this struct/union and set its entry
    // if its a component of it.
    bool    lookup( Symbol* sym ) const;

    bool            _isUnion;
    Symbol         *tag;

    int             size;          // size of the array.
    int             nComponents;
    Decl          **components;

  private:
    Decl   *lastDecl();
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class EnumDef
{
  public:
    EnumDef();
    ~EnumDef();

    EnumDef*   dup0() const;
    EnumDef*   dup() const;    // deep-copy

    void print(std::ostream& out, Symbol *name, int level) const;
    void findExpr( fnExprCallback cb );

    void addElement( Symbol *nme, Expression *val = NULL );

    void addElement( EnumConstant* ec );

    Symbol         *tag;

    int             size;       // size of the arrays.
    int             nElements;
    Symbol        **names;
    Expression    **values;
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

class Type;
typedef Dup<Type> DupableType;

class Type : public DupableType
{
  public:
    Type(TypeType _type=TT_Base);
    virtual ~Type();
    virtual Type ** getSubType() {return NULL;}
    virtual int     precedence() const { return 16; }
    virtual Type*   dup0() const =0;    // deep-copy

    virtual Type*   extend(Type *extension) { assert(0); return NULL; }

    // This function handles the complexity of printing a type.
    virtual void printType( std::ostream& out, Symbol *name,
			    bool showBase, int level, bool rawType=false/*prevents cpu-specific transforms*/ ) const;

    virtual bool printStructureStreamHelperType( std::ostream& out, const std::string& name, bool raw ) const = 0;
    virtual bool printStructureStreamShape( std::ostream& out ) { return false; }

    virtual void printBase( std::ostream& out, int level ) const {}
    virtual void printRawBase( std::ostream& out, int level ) const {printBase(out,level);}
    virtual void printBefore( std::ostream& out, Symbol *name, int level) const {}
    virtual void printAfter( std::ostream& out ) const {}

    virtual void printForm(std::ostream& out) const { out << "-[ABT]"; }

    virtual void registerComponents() {}

    virtual void findExpr( fnExprCallback cb ) {}

    virtual bool lookup( Symbol* sym ) const { return false; }

    virtual TypeQual getQualifiers( void ) const = 0;
    virtual BaseType *getBase( void ) = 0;

    bool    isBaseType() const { return (type == TT_Base); }
    bool    isPointer() const { return (type == TT_Pointer); }
    bool    isFunction() const { return (type == TT_Function); }
    bool    isArray() const { return (type == TT_Array); }
    bool    isStream() const { return (type == TT_Stream); }
    bool    isReduce() const { return (getQualifiers() & TQ_Reduce) != 0; }
    bool    isKernel() const {
       return ((getQualifiers() & (TQ_Reduce|TQ_Kernel)) != 0);
    }

    // Delete all types stored in this list.
    static void    DeleteTypeList(Type* typelist);


    TypeType        type;

    // Temporary - is moved into the declaration (Decl).
    StorageType     storage;

  private:
    Type*           link;    // For linking all type classes togather.
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class BaseType : public Type
{
  public:
    BaseType( BaseTypeSpec = BT_NoType );
    BaseType( StructDef *sd );
    BaseType( EnumDef *ed );
    ~BaseType();

    Type* dup0() const;    // deep-copy

    Type* extend(Type *extension) { assert(0); return NULL; }

    bool printStructureStreamHelperType( std::ostream& out, const std::string& name, bool raw ) const;
    bool printStructureStreamShape( std::ostream& out );

    void printBase( std::ostream& out, int level ) const;
    void printBefore( std::ostream& out, Symbol *name, int level) const;
    void printAfter( std::ostream& out ) const;

    void printForm(std::ostream& out) const;

    void registerComponents();

    bool lookup( Symbol* sym ) const;

    TypeQual getQualifiers( void ) const;
    BaseType *getBase( void );
    BaseType& operator = (const BaseType &b);

    BaseTypeSpec    typemask;

    TypeQual        qualifier;

    Symbol          *tag;        // tag for struct/union/enum
    Symbol          *typeName;   // typedef name for a UserType

    StructDef       *stDefn;     // optional definition of struct/union
    EnumDef         *enDefn;     // optional definition of enum 

  
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class PtrType : public Type
{
  public:
    PtrType(TypeQual qual = TQ_None)
        : Type(TT_Pointer), qualifier(qual), subType(NULL){};
    ~PtrType(){};
    virtual Type ** getSubType() {return &subType;}
    int    precedence() const { return 15; }

    Type* dup0() const;    // deep-copy

    Type* extend(Type *extension);

    bool printStructureStreamHelperType( std::ostream& out, const std::string& name, bool raw ) const {
      return false;
    }

    void printBase( std::ostream& out, int level ) const;
    void printBefore( std::ostream& out, Symbol *name, int level) const;
    void printAfter( std::ostream& out ) const;

    void printForm(std::ostream& out) const;

    void registerComponents() { if (subType) subType->registerComponents(); }

    void findExpr( fnExprCallback cb );

    bool lookup( Symbol* sym ) const;

    TypeQual getQualifiers( void ) const;
    BaseType *getBase( void );

    TypeQual        qualifier;
    Type           *subType;
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class ArrayType : public Type
{
  public:
    ArrayType(TypeType t, Expression *s)
          : Type(t), subType(NULL), size(s)
          { assert(t == TT_Stream || t == TT_Array); };

    ~ArrayType();

    Type* dup0() const;    // deep-copy

    Type* extend(Type *extension);
    virtual Type ** getSubType() {return &subType;}
    bool printStructureStreamHelperType( std::ostream& out, const std::string& name, bool raw ) const {
      return false;
    }

    void printBase( std::ostream& out, int level ) const;
    void printBefore( std::ostream& out, Symbol *name, int level) const;
    void printAfter( std::ostream& out ) const;

    void printForm(std::ostream& out) const;

    void registerComponents() { if (subType) subType->registerComponents(); }

    void findExpr( fnExprCallback cb );

    bool lookup( Symbol* sym ) const;

    TypeQual getQualifiers( void ) const;
    BaseType *getBase( void );

    Type           *subType;

    Expression     *size;
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class BitFieldType : public Type
{
  public:
    BitFieldType(Expression * s = NULL)
          : Type(TT_BitField),size(s),subType(NULL) {};
    ~BitFieldType();

    Type* dup0() const;    // deep-copy
    virtual Type ** getSubType() {return &subType;}
    Type* extend(Type *extension);

    bool printStructureStreamHelperType( std::ostream& out, const std::string& name, bool raw ) const {
      return false;
    }

    void printBase( std::ostream& out, int level ) const;
    void printBefore( std::ostream& out, Symbol *name, int level) const;
    void printAfter( std::ostream& out ) const;

    void printForm(std::ostream& out) const;

    void registerComponents() { if (subType) subType->registerComponents(); }

    void findExpr( fnExprCallback cb );

    bool lookup( Symbol* sym ) const;

    TypeQual getQualifiers( void ) const;
    BaseType *getBase( void );

    Expression     *size;

    Type           *subType;
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class FunctionType : public Type
{
  public:
    FunctionType(Decl *args_list = NULL);
    ~FunctionType();

    Type* dup0() const;    // deep-copy

    Type* extend(Type *extension);
    virtual Type ** getSubType() {return &subType;}

    bool printStructureStreamHelperType( std::ostream& out, 
                                         const std::string& name,
                                         bool raw) const {
      return false;
    }

    void printBase( std::ostream& out, int level ) const;
    void printBefore( std::ostream& out, Symbol *name, int level) const;
    void printAfter( std::ostream& out ) const;

    void printForm(std::ostream& out) const;

    void registerComponents() { if (subType) subType->registerComponents(); }

    void addArg(Decl *arg);
    void addArgs(Decl *args);

    void findExpr( fnExprCallback cb );

    bool lookup( Symbol* sym ) const;

    TypeQual getQualifiers( void ) const;
    BaseType *getBase( void );

    bool            KnR_decl;    // old-style function declaration?
    unsigned int    nArgs;
    unsigned int    size;
    Decl          **args;

    Type           *subType;    // The return type
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class StreamType : public Type
{
  public:
    StreamType(Expression *s)
          : Type(TT_Stream), subType(NULL), size(s) {};

    ~StreamType();
    virtual Type ** getSubType() {return &subType;}
    Type* dup0() const;    // deep-copy

    Type* extend(Type *extension);

    void printBase( std::ostream& out, int level ) const;
    void printBefore( std::ostream& out, Symbol *name, int level) const;
    void printAfter( std::ostream& out ) const;

    void printForm(std::ostream& out) const;

    void registerComponents() { if (subType) subType->registerComponents(); }

    void findExpr( fnExprCallback cb );

    bool lookup( Symbol* sym ) const;

    TypeQual getQualifiers( void ) const;
    BaseType *getBase( void );

    Type           *subType;

    Expression     *size;
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class GccAttrib
{
  public:
    GccAttrib( GccAttribType gccType );
    ~GccAttrib();

    GccAttrib* dup0() const;
    GccAttrib* dup() const;

    void    print( std::ostream& out ) const;

    GccAttribType    type;

    uint             value;    // For use with GCC_Aligned
    Symbol          *mode;     // For use with GCC_Mode, GCC_Format

    uint             strIdx;   // For use with GCC_Format
    uint             first;    // For use with GCC_Format
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
class Decl
{
  public:
    Decl( Symbol* sym = (Symbol*) NULL );
    Decl( Type* type );
    virtual ~Decl();

    Type*   extend( Type* type );

    bool    isTypedef() const { return (storage == ST_Typedef); }
    bool    isStatic() const { return (storage == ST_Static); }
    bool    isReduce() const { return (form->getQualifiers()&TQ_Reduce)!=0; }
    bool    isKernel() const { return form->isKernel(); }
    bool    isStream() const {
       return (form->type == TT_Stream || form->type == TT_BrtStream); }
    bool    isArray() const {
       return (form->type == TT_Array); }

    void    clear();

    Decl*   dup0() const;
    Decl*   dup() const;        // deep-copy

    void    copy(const Decl& decl);    // shallow copy

    void    print(std::ostream& out, bool showBase, int level=0) const;
    void printStructureStreamHelpers( std::ostream& out ) const;
    bool printStructureStreamInternals( std::ostream& out, bool raw ) const;
    bool printStructureStreamShape(std::ostream& out);
    bool printStructureStreamShapeInternals(std::ostream& out);
    void    printBase(std::ostream& out, Symbol *name,
                       bool showBase, int level) const;

    void    findExpr( fnExprCallback cb );

    // TIM: adding DAG-building for kernel splitting support
    virtual void buildSplitTree( SplitTreeBuilder& ioBuilder );

    bool lookup( Symbol* sym ) const;

    StorageType     storage;

    Type            *form;    // i.e., int *x[5] 

    Symbol          *name;    // The symbol being declared.

    GccAttrib       *attrib;  // optional gcc attribute

    Expression      *initializer;

    Decl            *next;    // For linking into lists
};

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

typedef    std::vector<Decl*>    DeclVector;

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

Decl*	ReverseList( Decl* dList );

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o


/*
 * FloatDimension --
 *
 *      Simple helper function to tell how many elements a FloatN type has
 *      fiven its TypeSpec.  Only annoying because type specs are bitmasks,
 *      not linear.
 */

static inline int
FloatDimension(BaseTypeSpec bt)
{
   if ((bt & BT_Float)||(bt&BT_Fixed)||(bt&BT_Half)||(bt&BT_Double)) return 1;
   else if ((bt & BT_Float2)||(bt&BT_Fixed2)||(bt&BT_Half2)||(bt&BT_Double2)) return 2;
   else if ((bt & BT_Float3)||(bt&BT_Fixed3)||(bt&BT_Half3)) return 3;
   else if ((bt & BT_Float4)||(bt&BT_Fixed4)||(bt&BT_Half4)) return 4;
   else return 0;
}
static inline int FloatGPUDimension(BaseTypeSpec bt)
{
    if (bt&BT_Double) return 2;
    if (bt&BT_Double2) return 4;
    return FloatDimension(bt);
}
// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o

#endif  /* DECL_H */

