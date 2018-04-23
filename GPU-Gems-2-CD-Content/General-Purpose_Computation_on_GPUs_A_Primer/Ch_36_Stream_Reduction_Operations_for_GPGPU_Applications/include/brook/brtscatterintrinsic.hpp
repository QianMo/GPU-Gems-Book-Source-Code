#ifndef GATHERINTRINSIC_HPP
#define GATHERINTRINSIC_HPP
class __StreamScatterAssign {public:
   template <class T> void operator () (T& out, const T& in)const{ 
      out=in;
   }
};
class __StreamScatterAdd {public:
   template <class T> void operator () (T& out, const T& in)const{ 
      out+=in;
   }
};
class __StreamScatterMul {public:
   template <class T> void operator () (T& out, const T& in)const{ 
      out*=in;
   }
};
class __StreamGatherInc {public:
   template <class T> void operator () (T&out, const T&in) const {
      out+=1;
   }
};
class __StreamGatherFetch {public:
   template <class T> void operator () (T&out, const T&in) const {}
};
extern  __StreamScatterAssign STREAM_SCATTER_ASSIGN;
extern  __StreamScatterAdd STREAM_SCATTER_ADD;
extern  __StreamScatterMul STREAM_SCATTER_MUL;
extern __StreamGatherInc STREAM_GATHER_INC;
extern __StreamGatherFetch STREAM_GATHER_FETCH;

#endif
