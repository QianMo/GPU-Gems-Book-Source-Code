#ifndef BRTGATHER_HPP
#define BRTGATHER_HPP
#include "brtvector.hpp"
#include "brtarray.hpp"
#include "brtscatterintrinsic.hpp"
#define STREAM_GATHER_FLOAT_INC STREAM_GATHER_INC
#define STREAM_GATHER_INTEGER_INC STREAM_GATHER_INC
template <class T, class Functor> void gatherOpHelper (T* s, 
                                                        float *index,
                                                        T* array,
                                                        unsigned int size,
                                                        const Functor & op) {
   for (unsigned int i=0;i<size;++i) {
      unsigned int ind = (unsigned int)index[i];
      memcpy (s+i,array+ind,sizeof(T));
      op(array[ind],s[i]);
   }
}

template <class T, class U, class V, class Functor> 
void gatherOpArrayHelper (T* s, 
                      V *index,
                      U &array,
                      unsigned int size,
                      const Functor & op) {
   for (unsigned int i=0;i<size;++i) {
      T* temp = &array[index[i]];
      memcpy (s+i,temp,sizeof(T));
      op(*temp,s[i]);
   }
}

template <class Functor, class T> 
void streamGatherOpIndexDet(T* data,
                             brook::StreamInterface * index,
                             brook::StreamInterface * out,
                             unsigned int bounds,
                             const Functor &op) {
   unsigned int dim = out->getDimension();
   assert(index->getFieldCount()==1);
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
      __BrtArray <T,4,false> o((T*)out->getData(brook::Stream::READ 
                                                | brook::Stream::WRITE),
                               out->getExtents());
      gatherOpArrayHelper(data,
                           (__BrtFloat4 *)index->getData(brook::Stream::READ),
                           o,
                           bounds,
                           op);      
      break;
   }
   case 3:{
      __BrtArray<T,3,false> o((T*)out->getData(brook::Stream::READ
                                               | brook::Stream::WRITE),
                              out->getExtents());
      gatherOpArrayHelper(data,
                           (__BrtFloat3 *)index->getData(brook::Stream::READ),
                           o,
                           bounds,
                           op);      
      
      break;
   }
   case 2:{
      __BrtArray<T,2,false> o((T*)out->getData(brook::Stream::READ
                                               |brook::Stream::WRITE),
                              out->getExtents());
      gatherOpArrayHelper(data,
                      (__BrtFloat2 *)index->getData(brook::Stream::READ),
                      o,
                      bounds,
                      op);      
      break;
   }
   default:
      gatherOpHelper(data,
                      (float *)index->getData(brook::Stream::READ),
                      (T*)out->getData(brook::Stream::READ 
                                       | brook::Stream::WRITE),
                      bounds,
                      op);
   }
}


template <class Functor> 
void streamGatherOp4 (brook::StreamInterface *s, 
                      brook::StreamInterface *index,
                      brook::StreamInterface *array, 
                      const Functor&op) {
   unsigned int bounds = s->getTotalSize();
   __BrtFloat4* data = 
      (__BrtFloat4 *) s->getData(brook::Stream::WRITE);
   streamGatherOpIndexDet(data,index,array,bounds,op);
   s->releaseData(brook::Stream::WRITE); 
   array->releaseData(brook::Stream::READ|brook::Stream::WRITE);      
   index->releaseData(brook::Stream::READ);
}   
template <class Functor> void streamGatherOp3 (brook::StreamInterface *s, 
                                               brook::StreamInterface *index,
                                               brook::StreamInterface *array, 
                                               const Functor&op) {
   unsigned int bounds = s->getTotalSize();
   __BrtFloat3* data = 
      (__BrtFloat3 *) s->getData(brook::Stream::WRITE);
   streamGatherOpIndexDet(data,index,array,bounds,op);
   s->releaseData(brook::Stream::WRITE); 
   array->releaseData(brook::Stream::READ|brook::Stream::WRITE);      
   index->releaseData(brook::Stream::READ);
}   
template <class Functor> void streamGatherOp2 (brook::StreamInterface *s, 
                                               brook::StreamInterface *index,
                                               brook::StreamInterface *array, 
                                               const Functor&op) {
   unsigned int bounds = s->getTotalSize();
   __BrtFloat2* data = 
      (__BrtFloat2 *) s->getData(brook::Stream::WRITE);
   streamGatherOpIndexDet(data,index,array,bounds,op);
   s->releaseData(brook::Stream::WRITE); 
   array->releaseData(brook::Stream::READ|brook::Stream::WRITE);      
   index->releaseData(brook::Stream::READ);
}   
template <class Functor> void streamGatherOp1 (brook::StreamInterface *s, 
                                               brook::StreamInterface *index,
                                               brook::StreamInterface *array, 
                                               const Functor&op) {
   unsigned int bounds = s->getTotalSize();
   __BrtFloat1* data = 
      (__BrtFloat1 *) s->getData(brook::Stream::WRITE);
   streamGatherOpIndexDet(data,index,array,bounds,op);
   s->releaseData(brook::Stream::WRITE); 
   array->releaseData(brook::Stream::READ|brook::Stream::WRITE);      
   index->releaseData(brook::Stream::READ);
}   
#endif
