#ifndef BRTSCATTER_HPP
#define BRTSCATTER_HPP
#include "brtvector.hpp"
#include "brtarray.hpp"
#include "brtscatterintrinsic.hpp"
#define STREAM_SCATTER_FLOAT_MUL STREAM_SCATTER_MUL
#define STREAM_SCATTER_FLOAT_ADD STREAM_SCATTER_ADD
#define STREAM_SCATTER_INTEGER_MUL STREAM_SCATTER_MUL
#define STREAM_SCATTER_INTEGER_ADD STREAM_SCATTER_ADD
template <class T, class Functor> void scatterOpHelper (const T* s, 
                                                        float *index,
                                                        T* out,
                                                        unsigned int size,
                                                        const Functor & op) {
   for (unsigned int i=0;i<size;++i) {
      unsigned int tmp = (unsigned int)index[i];
      if (tmp<size)
        op(out[tmp],s[i]);
   }
}

template <class T, class U, class V, class Functor> 
void scatterOpArrayHelper (const T* s, 
                      V *index,
                      U &out,
                      unsigned int size,
                      const Functor & op) {
   for (unsigned int i=0;i<size;++i) {
      op(out[index[i]],s[i]);
   }
}

template <class Functor, class T> 
void streamScatterOpIndexDet(const T* data,
                             brook::StreamInterface * index,
                             brook::StreamInterface * out,
                             unsigned int bounds,
                             const Functor &op) {
   unsigned int dim = out->getDimension();
   assert(index->getFieldCount() == 1);
   switch (index->getIndexedFieldType(0)) {
   case __BRTFLOAT4:
      if (dim!=4)dim=1;
      break;
   case __BRTFLOAT3:
      if (dim!=3)dim=1;
      break;
   case __BRTFLOAT2:
      if (dim!=2)dim=1;
      break;
   default:
      dim=1;
   }
   switch (dim) {
   case 4:{
      __BrtArray <T,4,false> o((T*)out->getData(brook::Stream::WRITE),
                               out->getExtents());
      scatterOpArrayHelper(data,
                           (__BrtFloat4 *)index->getData(brook::Stream::READ),
                           o,
                           bounds,
                           op);      
      break;
   }
   case 3:{
      __BrtArray<T,3,false> o((T*)out->getData(brook::Stream::WRITE),
                              out->getExtents());
      scatterOpArrayHelper(data,
                           (__BrtFloat3 *)index->getData(brook::Stream::READ),
                           o,
                           bounds,
                           op);      
      
      break;
   }
   case 2:{
      __BrtArray<T,2,false> o((T*)out->getData(brook::Stream::WRITE),
                              out->getExtents());
      scatterOpArrayHelper(data,
                      (__BrtFloat2 *)index->getData(brook::Stream::READ),
                      o,
                      bounds,
                      op);      
      break;
   }
   default:
      scatterOpHelper(data,
                      (float *)index->getData(brook::Stream::READ),
                      (T*)out->getData(brook::Stream::WRITE),
                      bounds,
                      op);
   }
}


template <class Functor> 
void streamScatterOp4 (brook::StreamInterface *s, 
                       brook::StreamInterface *index,
                       brook::StreamInterface *array, 
                       const Functor&op) {
   unsigned int bounds = s->getTotalSize();
   const __BrtFloat4* data = 
      (const __BrtFloat4 *) s->getData(brook::Stream::READ);
   streamScatterOpIndexDet(data,index,array,bounds,op);
   s->releaseData(brook::Stream::READ); 
   array->releaseData(brook::Stream::WRITE);      
   index->releaseData(brook::Stream::READ);
}   
template <class Functor> void streamScatterOp3 (brook::StreamInterface *s, 
                                                brook::StreamInterface *index,
                                                brook::StreamInterface *array, 
                                                const Functor&op) {
   unsigned int bounds = s->getTotalSize();
   const __BrtFloat3* data = 
      (const __BrtFloat3 *) s->getData(brook::Stream::READ);
   streamScatterOpIndexDet(data,index,array,bounds,op);
   s->releaseData(brook::Stream::READ); 
   array->releaseData(brook::Stream::WRITE);      
   index->releaseData(brook::Stream::READ);
}   
template <class Functor> void streamScatterOp2 (brook::StreamInterface *s, 
                                                brook::StreamInterface *index,
                                                brook::StreamInterface *array, 
                                                const Functor&op) {
   unsigned int bounds = s->getTotalSize();
   const __BrtFloat2* data = 
      (const __BrtFloat2 *) s->getData(brook::Stream::READ);
   streamScatterOpIndexDet(data,index,array,bounds,op);
   s->releaseData(brook::Stream::READ); 
   array->releaseData(brook::Stream::WRITE);      
   index->releaseData(brook::Stream::READ);
}   
template <class Functor> void streamScatterOp1 (brook::StreamInterface *s, 
                                                brook::StreamInterface *index,
                                                brook::StreamInterface *array, 
                                                const Functor&op) {
   unsigned int bounds = s->getTotalSize();
   const __BrtFloat1* data = 
      (const __BrtFloat1 *) s->getData(brook::Stream::READ);
   streamScatterOpIndexDet(data,index,array,bounds,op);
   s->releaseData(brook::Stream::READ); 
   array->releaseData(brook::Stream::WRITE);      
   index->releaseData(brook::Stream::READ);
}   
#endif
