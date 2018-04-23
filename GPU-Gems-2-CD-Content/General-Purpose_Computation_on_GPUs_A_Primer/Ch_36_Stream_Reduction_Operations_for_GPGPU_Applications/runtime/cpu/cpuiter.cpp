#include <stdio.h>
#include <stdlib.h>
#include "cpu.hpp"

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
static float lerp (unsigned int i, unsigned int end,float lower,float upper) {
   float frac=end>0?((float)i)/(float)end:(float)upper;
   return (1-frac)*lower+frac*upper;
}

namespace brook {

   // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
   // This will create a the local CPU stream with the appropriate data
   // that fulfills the ranges specified by the program.
   void CPUIter::allocateStream(unsigned int dims, 
                                const unsigned int extents[],
                                const float ranges[]) {
      float * data = (float*)malloc (stream.getTotalSize()*sizeof(float)*type);
      float epsilon = 1.0f/1000000.0f;
      
      if (dims<2) {
         for (unsigned int i=0;i<extents[0];++i) {
            for (unsigned int j=0;j<(unsigned int) type;++j) {
               data[i*type+j]=lerp(i,extents[0],ranges[j],ranges[j+type]) +
                 epsilon;
            }
         }
      }else if (dims==2){
         //now we know dims == data type;
         unsigned int i[2]={0,0};
         for (i[0]=0;i[0]<extents[0];++i[0]) {
            for (i[1]=0;i[1]<extents[1];++i[1]) {
               for (unsigned int k=0;k<2;++k) {
                  float f= lerp (i[k],extents[k],ranges[1-k],ranges[3-k]);
                  data[(i[0]*extents[1]+i[1])*2+1-k]=f + epsilon;
               }
            }
         }
      }else {
         // we do not handle 3d iterators.
         // This is not a planned feature as video card texture interpolants
         // go in 2d.
         fprintf(stderr,"Nd Iterator not allowed where N>2\n");; 
         assert(0);
      }
      streamRead(&stream,data);
      free(data);
   }

}
