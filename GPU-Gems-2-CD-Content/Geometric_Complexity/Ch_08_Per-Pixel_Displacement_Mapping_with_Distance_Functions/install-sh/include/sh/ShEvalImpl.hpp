#ifndef SHEVALIMPL_HPP 
#define SHEVALIMPL_HPP

#include <numeric>
#include "ShEval.hpp"
#include "ShVariant.hpp"
#include "ShDebug.hpp"
#include "ShError.hpp"

namespace SH {

template<ShOperation OP, typename T>
void ShRegularOp<OP, T>::operator()( 
    ShVariant* dest, const ShVariant* a, const ShVariant* b, const ShVariant* c) const
{
  ShDataVariant<T, SH_HOST>* destVec;
  const ShDataVariant<T, SH_HOST>  *aVec, *bVec, *cVec;

  SH_DEBUG_ASSERT(dest && a);
  destVec = variant_cast<T, SH_HOST>(dest);
  aVec = variant_cast<T, SH_HOST>(a);

  if(b) bVec = variant_cast<T, SH_HOST>(b);
  else bVec = 0;

  if(c) cVec = variant_cast<T, SH_HOST>(c);
  else cVec = 0;

  ShRegularOpChooser<OP, T>::Op::doop(destVec, aVec, bVec, cVec);
}

template<ShOperation OP, typename T1, typename T2>
void ShIntervalOp<OP, T1, T2>::operator()( 
    ShVariant* dest, const ShVariant* a, const ShVariant* b, const ShVariant* c) const
{

  SH_DEBUG_ASSERT(dest && a);

  ShDataVariant<T1, SH_HOST>* destVec = variant_cast<T1, SH_HOST>(dest);

  const ShDataVariant<T2, SH_HOST>* aVec = variant_cast<T2, SH_HOST>(a);

  ShConcreteIntervalOp<OP, T1, T2>::doop(*destVec, *aVec);
}

template<typename T>
void _shInitFloatOps() {
  ShEval* eval = ShEval::instance();
  const ShValueType V = ShStorageTypeInfo<T>::value_type;

  eval->addOp(SH_OP_ABS, new ShRegularOp<SH_OP_ABS, T>(), V, V);
  eval->addOp(SH_OP_ACOS, new ShRegularOp<SH_OP_ACOS, T>(), V, V);
  eval->addOp(SH_OP_ASIN, new ShRegularOp<SH_OP_ASIN, T>(), V, V);
  eval->addOp(SH_OP_ASN, new ShRegularOp<SH_OP_ASN, T>(), V, V);
  eval->addOp(SH_OP_ATAN, new ShRegularOp<SH_OP_ATAN, T>(), V, V);
  eval->addOp(SH_OP_CBRT, new ShRegularOp<SH_OP_CBRT, T>(), V, V);
  eval->addOp(SH_OP_CEIL, new ShRegularOp<SH_OP_CEIL, T>(), V, V);
  eval->addOp(SH_OP_COS, new ShRegularOp<SH_OP_COS, T>(), V, V);
  eval->addOp(SH_OP_CSUM, new ShRegularOp<SH_OP_CSUM, T>(), V, V);
  eval->addOp(SH_OP_CMUL, new ShRegularOp<SH_OP_CMUL, T>(), V, V);
  eval->addOp(SH_OP_EXP, new ShRegularOp<SH_OP_EXP, T>(), V, V);
  eval->addOp(SH_OP_EXP2, new ShRegularOp<SH_OP_EXP2, T>(), V, V);
  eval->addOp(SH_OP_EXP10, new ShRegularOp<SH_OP_EXP10, T>(), V, V);
  eval->addOp(SH_OP_FLR, new ShRegularOp<SH_OP_FLR, T>(), V, V);
  eval->addOp(SH_OP_FRAC, new ShRegularOp<SH_OP_FRAC, T>(), V, V);
  eval->addOp(SH_OP_LOG, new ShRegularOp<SH_OP_LOG, T>(), V, V);
  eval->addOp(SH_OP_LOG2, new ShRegularOp<SH_OP_LOG2, T>(), V, V);
  eval->addOp(SH_OP_LOG10, new ShRegularOp<SH_OP_LOG10, T>(), V, V);
  //eval->addOp(SH_OP_NEG, new ShRegularOp<SH_OP_NEG, T>());
  eval->addOp(SH_OP_NORM, new ShRegularOp<SH_OP_NORM, T>(), V, V);
  eval->addOp(SH_OP_RCP, new ShRegularOp<SH_OP_RCP, T>(), V, V);
  eval->addOp(SH_OP_RND, new ShRegularOp<SH_OP_RND, T>(), V, V);
  eval->addOp(SH_OP_RSQ, new ShRegularOp<SH_OP_RSQ, T>(), V, V);
  eval->addOp(SH_OP_SIN, new ShRegularOp<SH_OP_SIN, T>(), V, V);
  eval->addOp(SH_OP_SGN, new ShRegularOp<SH_OP_SGN, T>(), V, V);
  eval->addOp(SH_OP_SQRT, new ShRegularOp<SH_OP_SQRT, T>(), V, V);
  eval->addOp(SH_OP_TAN, new ShRegularOp<SH_OP_TAN, T>(), V, V);

  eval->addOp(SH_OP_ADD, new ShRegularOp<SH_OP_ADD, T>(), V, V, V);
  eval->addOp(SH_OP_ATAN2, new ShRegularOp<SH_OP_ATAN2, T>(), V, V, V);
  eval->addOp(SH_OP_DIV, new ShRegularOp<SH_OP_DIV, T>(), V, V, V);
  eval->addOp(SH_OP_DOT, new ShRegularOp<SH_OP_DOT, T>(), V, V, V);
  eval->addOp(SH_OP_MAX, new ShRegularOp<SH_OP_MAX, T>(), V, V, V);
  eval->addOp(SH_OP_MIN, new ShRegularOp<SH_OP_MIN, T>(), V, V, V);
  eval->addOp(SH_OP_MOD, new ShRegularOp<SH_OP_MOD, T>(), V, V, V);
  eval->addOp(SH_OP_MUL, new ShRegularOp<SH_OP_MUL, T>(), V, V, V);
  eval->addOp(SH_OP_POW, new ShRegularOp<SH_OP_POW, T>(), V, V, V);
  eval->addOp(SH_OP_SEQ, new ShRegularOp<SH_OP_SEQ, T>(), V, V, V);
  eval->addOp(SH_OP_SGE, new ShRegularOp<SH_OP_SGE, T>(), V, V, V);
  eval->addOp(SH_OP_SGT, new ShRegularOp<SH_OP_SGT, T>(), V, V, V);
  eval->addOp(SH_OP_SLE, new ShRegularOp<SH_OP_SLE, T>(), V, V, V);
  eval->addOp(SH_OP_SLT, new ShRegularOp<SH_OP_SLT, T>(), V, V, V);
  eval->addOp(SH_OP_SNE, new ShRegularOp<SH_OP_SNE, T>(), V, V, V);
  eval->addOp(SH_OP_XPD, new ShRegularOp<SH_OP_XPD, T>(), V, V, V);

  eval->addOp(SH_OP_LRP, new ShRegularOp<SH_OP_LRP, T>(), V, V, V, V);
  eval->addOp(SH_OP_MAD, new ShRegularOp<SH_OP_MAD, T>(), V, V, V, V);
  eval->addOp(SH_OP_COND, new ShRegularOp<SH_OP_COND, T>(), V, V, V, V);
}

template<typename T>
void _shInitIntOps() {
  ShEval* eval = ShEval::instance();
  const ShValueType V = ShStorageTypeInfo<T>::value_type;

  // guaranteed int result ops (ignore overflow)
  eval->addOp(SH_OP_ABS, new ShRegularOp<SH_OP_ABS, T>(), V, V);
  eval->addOp(SH_OP_ASN, new ShRegularOp<SH_OP_ASN, T>(), V, V);
  //eval->addOp(SH_OP_NEG, new ShRegularOp<SH_OP_NEG, T>());
  eval->addOp(SH_OP_CSUM, new ShRegularOp<SH_OP_CSUM, T>(), V, V);
  eval->addOp(SH_OP_CMUL, new ShRegularOp<SH_OP_CMUL, T>(), V, V);
  eval->addOp(SH_OP_SGN, new ShRegularOp<SH_OP_SGN, T>(), V, V);

  eval->addOp(SH_OP_ADD, new ShRegularOp<SH_OP_ADD, T>(), V, V, V);
  eval->addOp(SH_OP_DIV, new ShRegularOp<SH_OP_DIV, T>(), V, V, V);
  eval->addOp(SH_OP_DOT, new ShRegularOp<SH_OP_DOT, T>(), V, V, V);
  eval->addOp(SH_OP_MAX, new ShRegularOp<SH_OP_MAX, T>(), V, V, V);
  eval->addOp(SH_OP_MIN, new ShRegularOp<SH_OP_MIN, T>(), V, V, V);
  eval->addOp(SH_OP_MOD, new ShRegularOp<SH_OP_MOD, T>(), V, V, V);
  eval->addOp(SH_OP_MUL, new ShRegularOp<SH_OP_MUL, T>(), V, V, V);
  eval->addOp(SH_OP_POW, new ShRegularOp<SH_OP_POW, T>(), V, V, V);
  eval->addOp(SH_OP_SEQ, new ShRegularOp<SH_OP_SEQ, T>(), V, V, V);
  eval->addOp(SH_OP_SGE, new ShRegularOp<SH_OP_SGE, T>(), V, V, V);
  eval->addOp(SH_OP_SGT, new ShRegularOp<SH_OP_SGT, T>(), V, V, V);
  eval->addOp(SH_OP_SLE, new ShRegularOp<SH_OP_SLE, T>(), V, V, V);
  eval->addOp(SH_OP_SLT, new ShRegularOp<SH_OP_SLT, T>(), V, V, V);
  eval->addOp(SH_OP_SNE, new ShRegularOp<SH_OP_SNE, T>(), V, V, V);
  eval->addOp(SH_OP_XPD, new ShRegularOp<SH_OP_XPD, T>(), V, V, V);

  eval->addOp(SH_OP_LRP, new ShRegularOp<SH_OP_LRP, T>(), V, V, V, V);
  eval->addOp(SH_OP_MAD, new ShRegularOp<SH_OP_MAD, T>(), V, V, V, V);
  eval->addOp(SH_OP_COND, new ShRegularOp<SH_OP_COND, T>(), V, V, V, V);

  // unary ASN equivalents for ints 
  eval->addOp(SH_OP_CEIL, new ShRegularOp<SH_OP_ASN, T>(), V, V);
  eval->addOp(SH_OP_FLR, new ShRegularOp<SH_OP_ASN, T>(), V, V);
  eval->addOp(SH_OP_FRAC, new ShRegularOp<SH_OP_ASN, T>(), V, V);
  eval->addOp(SH_OP_RND, new ShRegularOp<SH_OP_ASN, T>(), V, V);

  // all other ops must use float (since they can potentially give float
  // results)
}

template<typename T, typename IntervalT>
void _shInitIntervalOps() {
  ShEval* eval = ShEval::instance();
  const ShValueType V = ShStorageTypeInfo<T>::value_type;
  const ShValueType IntervalV = ShStorageTypeInfo<IntervalT>::value_type; 

  eval->addOp(SH_OP_LO, new ShIntervalOp<SH_OP_LO, T, IntervalT>(), 
      V, IntervalV);
  eval->addOp(SH_OP_HI, new ShIntervalOp<SH_OP_HI, T, IntervalT>(), 
      V, IntervalV);

  eval->addOp(SH_OP_SETLO, new ShIntervalOp<SH_OP_SETLO, IntervalT, T>(), 
      IntervalV, V);
  eval->addOp(SH_OP_SETHI, new ShIntervalOp<SH_OP_SETHI, IntervalT, T>(), 
      IntervalV, V);
}

}

#endif
