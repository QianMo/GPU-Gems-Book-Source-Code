#ifdef _WIN32
#pragma warning(disable:4786)
#endif
#include <brook/brook.hpp>
#include <map>
#include <stdio.h>
class DualInt {
public:
   int k;
   int N;
   DualInt (int k, int N) {
      this->k=k;
      this->N=N;
   }
   bool operator < (const DualInt &other) const{
      if (N<other.N)
         return true;
      if (N>other.N)
         return false;
      return k<other.k;
   }
};
static int TwoPowerX(int nNumber) {
  // Achieve the multiplication by left-shifting 
  return (1<<nNumber);
}

static int BitReverse(int nNumberToReverse, int nNumberOfBits) {
  int nBitIndex;
  int nReversedNumber = 0;
  for (nBitIndex = nNumberOfBits-1; nBitIndex >= 0; --nBitIndex) {
    if ((1 == nNumberToReverse >> nBitIndex)) {         
      nReversedNumber  += TwoPowerX(nNumberOfBits-1-nBitIndex);    
      nNumberToReverse -= TwoPowerX(nBitIndex);                      
    }
  }
  return(nReversedNumber);
}
static std::map<int, float2 *> rawWstream;
static std::map <DualInt,brook::stream *> Ws;
static std::map <DualInt,brook::stream *> WWs;
int myabs (int i) {
   return i>=0?i:-i;
}
brook::stream &getWreverse(int k, int logN, bool vertical) {
    int N=(1<<logN);
  const brook::StreamType* flawt2 = brook::getStreamType((float2*)0);
   std::map<DualInt,brook::stream*>::iterator iter = WWs.find(DualInt(k,vertical?myabs(N):-myabs(N)));
   if (iter!=WWs.end()) {
      return *(*iter).second;
   }
   brook::stream *ret;
   std::map<int,float2 *>::iterator rawW = rawWstream.find(N);
   float2 * rW=0;
   if (rawW==rawWstream.end()) {
      rW = new float2 [N/2];
      for (int i=0;i<N/2;++i) {
         float theta = (float)(2*3.1415926536*i/N);
         rW[i].x=cos(theta);
         rW[i].y=sin(theta);
      }
      rawWstream[N]=rW;
   }else {
      rW = (*rawW).second;
   }
   if (0) {
      if (vertical) {
         ret = new brook::stream (flawt2,1,N/2,-1);
      }else {
         ret = new brook::stream (flawt2,N/2,1,-1);
      }
      if (k!=0) {
         //else we stay the same
         float2 *stridedW = new float2 [N/2];
         for (int i=0;i<N/2;++i) {
            stridedW[i]=rW[i-(i%(1<<k))];
         }
         streamRead(*ret,stridedW);
         delete []stridedW;
      }else {
         streamRead(*ret,rW);
      }
   }else {
      int thissize = (N/2)/(1<<k);
      if (vertical) {
         ret = new brook::stream (flawt2,1,thissize,-1);
      }else {
         ret = new brook::stream (flawt2,thissize,1,-1);
      }
      if (k!=0) {
         //else we stay the same
         float2 *stridedW = new float2 [thissize];
         for (int i=0;i<thissize;++i) {
            stridedW[i]=rW[i*(1<<k)];
         }
         streamRead(*ret,stridedW);
         delete []stridedW;
      }else {
         streamRead(*ret,rW);
      }
   }
   WWs[DualInt(k,vertical?myabs(N):-myabs(N))]=ret;   
   return *ret;
}
brook::stream &getWforward(int k, int logN, bool vertical) {
  int N = (1<<logN);
  int Stride=(N/2)/(1<<k);
  const brook::StreamType* flawt2 = brook::getStreamType((float2*)NULL);
  std::map<DualInt,brook::stream*>::iterator iter = Ws.find(DualInt(k,vertical?myabs(N):-myabs(N)));
  if (iter!=Ws.end()) {
    return *(*iter).second;
  }
  brook::stream *ret;
  if (vertical) {
    ret = new brook::stream (flawt2,1,N/2,-1);
  }else {
    ret = new brook::stream (flawt2,N/2,1,-1);
  }
  float2 *W = new float2 [N/2];
  for (int i=0;i<N/2;++i) {
      int aindex=i*2;
      int bindex=BitReverse(aindex,logN);
      int index=(bindex-(bindex%Stride));
      float ang=(float)(index*3.1415926536/(N/2));
      W[i].x=cos(ang);
      W[i].y=sin(ang);
  }
  streamRead(*ret,W);
  delete []W;
  Ws[DualInt(k,vertical?myabs(N):-myabs(N))]=ret;   
  return *ret;
}
brook::stream &getW(int k, int N, bool vertical, bool reverse) {
    return reverse?getWreverse(k,N,vertical):getWforward(k,N,vertical);
}
void freeWs () {
   for (std::map<DualInt,brook::stream *>::iterator i=Ws.begin();
        i!=Ws.end();
        ++i) {
      delete (*i).second;      
   }
   for (std::map<DualInt,brook::stream *>::iterator i=WWs.begin();
        i!=WWs.end();
        ++i) {
      delete (*i).second;      
   }
}
