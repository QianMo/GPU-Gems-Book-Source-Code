#ifndef SHEVAL_HPP 
#define SHEVAL_HPP

#include <vector>
#include <map>
#include "ShHashMap.hpp"
#include "ShStatement.hpp"
#include "ShVariant.hpp"
#include "ShOperation.hpp"
#include "ShRefCount.hpp"
#include "ShInterval.hpp"
#include "ShHalf.hpp"

namespace SH {

/** 
 * Note: Everything here is meant for internal use.
 *
 * Type specific definitions for the internal instructions
 * as they are used in host computations.
 *
 * Each ShEvalOp can evaluate a subset of operations
 * for specific input types (i.e. arithmetic ops, with T <- T x T)
 *
 * Most normal ops fall into the dest & src typse all = T category,
 * but some (like interval lo/hi accessors) have different dest 
 * type from src type.
 *
 * Initially, each evalOp registers all the ops it can handle
 * with the ShEval (the (opcode X src types) must be unique for now).
 * Then ShInstructions passes ShEval a bunch of stuff and ShEval
 * will look up the appropriate evalOp for an operation.
 *
 * Extend later to handle n-ary operations if necessary (there are a few in the
 * shref, but none implemented as IR operations yet...)
 */


// forward declarations
//class ShVariant;
class ShEvalOp;
//typedef ShPointer<ShVariant> ShVariant*;
//typedef ShPointer<const ShVariant> const ShVariant*;

struct 
SH_DLLEXPORT
ShEvalOpInfo: public ShStatementInfo {
  ShOperation m_op;

  const ShEvalOp* m_evalOp;

  // type indices of the destination and sources
  // These are set to a valid value type except when the src/dest is not
  // used for m_op 
  ShValueType m_dest;
  ShValueType m_src[3]; 

  ShEvalOpInfo(ShOperation op, const ShEvalOp* evalOp, ShValueType dest, 
      ShValueType src0, ShValueType src1, ShValueType src2);

  ShStatementInfo* clone() const;

  std::string encode() const;
};

class 
SH_DLLEXPORT
ShEval {
  public:
    /** Decides which evaluation evalOp to use and calls it up.
     * If an op is unary, leave b, c = 0.
     * If an op is binary leave c = 0.
     * If an op has no dest, leave dest = 0.
     *
     * TODO (should really break this out into separate functions.  EvalOps can
     * have a single function in th einterface)
     */
    void operator()(ShOperation op, ShVariant* dest,
        const ShVariant* a, const ShVariant* b, const ShVariant* c) const;

    /** Registers a evalOp for a certain operation/source type index combination */ 
    void addOp(ShOperation op, const ShEvalOp* evalOp, ShValueType dest, 
        ShValueType src0, ShValueType src1 = SH_VALUETYPE_END, 
        ShValueType src2 = SH_VALUETYPE_END); 

    /** Returns a new op info representing the types that arguments
     * should be cast into for an operation.
     * Caches the result.
     */
    const ShEvalOpInfo* getEvalOpInfo(ShOperation op, ShValueType dest,
        ShValueType src0, ShValueType src1 = SH_VALUETYPE_END, 
        ShValueType src2 = SH_VALUETYPE_END) const;

    /** debugging function */ 
    std::string availableOps() const;

    static ShEval* instance();

  private:
    ShEval();


    typedef std::list<ShEvalOpInfo> OpInfoList;
    typedef OpInfoList OpInfoMap[SH_OPERATION_END];
    OpInfoMap m_evalOpMap; 

    typedef ShPairPairHashMap<ShOperation, ShValueType, ShValueType, ShValueType, const ShEvalOpInfo*> EvalOpCache;
    mutable EvalOpCache  m_evalOpCache;

    static ShEval* m_instance;
};

class 
SH_DLLEXPORT
ShEvalOp {
  public:
    virtual ~ShEvalOp();

    // Wraps an operation where at least dest and a are non-null.
    virtual void operator()(ShVariant* dest, const ShVariant* a, 
        const ShVariant* b, const ShVariant* c) const = 0;
};

// The strategy for defining ops is to use separate classes to hold
// and register operations for different categories of types.
//
// Each of these "concrete" evaluation evalOps must implement
// 1) the virtual void ShEvalOp::operator()(...)
//    This is used to evaluate ShVariables
//
// 2) Functions 
//    template<ShOperation S>
//    static void unaryOp(ShDataVariant<T1> &dest, const ShDataVariant<T2> &src);
//
//    and similarly for binary, ternary ops 
//    (for most ops, only T1 = T2 is supported directly,
//    but for range arithmetic ops, some args may be in a range arithmetic
//    type and other args might not)
//
//    These can be specialized to implement specific ops,
//    and thus used directly for computation on data from
//    ShGenerics without going through ANY layers of virtual 
//    function calls.

/** A ShRegularOp is one where all the arguments and the destination
 * are variants of type V (data type SH_HOST).
 */
template<ShOperation S, typename T>
struct ShRegularOp: public ShEvalOp {
  void operator()(ShVariant* dest, const ShVariant* a, 
      const ShVariant* b, const ShVariant* c) const; 
};

// If functions could be partially specialized, then wouldn't need
// to wrap this in a class.

// TODO might want to break this down into a few more subclasses to handle
// 1) special float/double cmath functions (for C built in types)
// 2) other special functions (sgn, rcp, rsq, etc.) that C types don't have
// OR, use helper functions 
template<ShOperation S, typename T>
struct ShConcreteRegularOp {
  typedef ShDataVariant<T, SH_HOST> Variant;
  typedef Variant* DataPtr; 
  typedef const Variant* DataCPtr; 

  static void doop(DataPtr dest, DataCPtr a, DataCPtr b = 0, DataCPtr c = 0);
};

/// evalOp that uses cmath functions and
// has little code snippets for functions like sgn, rcp, rsq, etc.
// that are not built-in cmath functions.
//
//TODO - not all the functions make sense on integer types...may
//want to not declare the ones that don't make sense...
template<ShOperation S, typename T>
struct ShConcreteCTypeOp {
  typedef ShDataVariant<T, SH_HOST> Variant;
  typedef Variant* DataPtr; 
  typedef const Variant* DataCPtr; 

  static void doop(DataPtr dest, DataCPtr a, DataCPtr b = 0, DataCPtr c = 0);
};

template<ShOperation S, typename T>
struct ShRegularOpChooser {
  typedef ShConcreteRegularOp<S, T> Op;
};

#define SHOPC_CTYPE_OP(T)\
template<ShOperation S>\
struct ShRegularOpChooser<S, T> {\
  typedef ShConcreteCTypeOp<S, T> Op;\
};

SHOPC_CTYPE_OP(double);
SHOPC_CTYPE_OP(float);
SHOPC_CTYPE_OP(ShHalf);
SHOPC_CTYPE_OP(int);

/** A ShIntervalOP is one where one argument is an interval type,
 * and the other argument must be its corresponding bound type.
 */
template<ShOperation S, typename T1, typename T2>
struct ShIntervalOp: public ShEvalOp {
  void operator()(ShVariant* dest, const ShVariant* a, 
      const ShVariant* b, const ShVariant* c) const; 
};

template<ShOperation S, typename T1, typename T2>
struct ShConcreteIntervalOp{
  static void doop(ShDataVariant<T1, SH_HOST> &dest, 
      const ShDataVariant<T2, SH_HOST> &a);
      
};


// initializes the regular Ops for a floating point type T
// with ShOpEvalOp<OP, T> objects
template<typename T>
void _shInitFloatOps();

// initializes the regular Ops for an integer type T
// (a subset of the floating point ones)
template<typename T>
void _shInitIntOps();

// initializes the interval ops for a type T and ShInterval<T>
template<typename T, typename IntervalT>
void _shInitIntervalOps();


}

#include "ShEvalImpl.hpp"
#include "ShConcreteRegularOpImpl.hpp"
#include "ShConcreteCTypeOpImpl.hpp"
#include "ShConcreteIntervalOpImpl.hpp"
//#include "ShIntervalEvalImpl.hpp"

#endif
