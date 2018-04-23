#include <iostream>
#include <assert.h>
#include "cpu.hpp"
#include <stdio.h>

#include <brook/brtarray.hpp>



static brook::CPUKernel *current_cpu_kernel = NULL; 

// o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
//          The indexof function

__BrtFloat4 __indexof(const void *ptr) {
  unsigned int i;
  
  assert(current_cpu_kernel);
  
  brook::CPUKernel *k = current_cpu_kernel;
  
  for (i=0; i<k->input_args.size(); i++) {
    brook::CPUStream *s = (brook::CPUStream *) k->input_args[i];
    unsigned char *p = (unsigned char *) ptr;
    unsigned char *p_begin = ((unsigned char *) 
                              s->getData(brook::StreamInterface::READ));
    unsigned char *p_end = p_begin + s->malloced_size;
    
    if (p >= p_begin && p < p_end)
      return s->computeIndexOf((p-p_begin) / s->stride);
  }
  
  for (i=0; i<k->output_args.size(); i++) {
    brook::CPUStream *s = (brook::CPUStream *) k->output_args[i];
    unsigned char *p = (unsigned char *) ptr;
    unsigned char *p_begin = ((unsigned char *) 
                              s->getData(brook::StreamInterface::READ));
    unsigned char *p_end = p_begin + s->malloced_size;
    
    if (p >= p_begin && p < p_end)
      return s->computeIndexOf((p-p_begin) / s->stride);
  }
  
  fprintf (stderr, "Indexof called on bogus address");
  assert (0);
  
  return __BrtFloat4();
}

namespace brook {

  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  CPUKernel::CPUKernel(const void * source[]) {
    const char ** src= (const char**)(source);
    
    func = 0;
    
    for (unsigned int i=0;;i+=2) {
      if (src[i]==NULL){
        if (!func){
          func=NULL;
          std::cerr<<"CPUKernel failure - ";
          std::cerr<<"no CPU program string found.";
          std::cerr <<std::endl;
        }
        break;
      }
      if (strcmp(src[i],"cpu")==0){
        func = (callable*)source[i+1];
      }
    }

    curpos = NULL;
    extents = NULL;
    minpos = NULL;
    maxpos = NULL;
    dims = 0;

    Cleanup();
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushStreamInterface(StreamInterface * s) {
    if (!s->isCPU()) {
      s = new CPUStreamShadow(s, ::brook::Stream::READ);
      freeme.push_back(s);
    }
    
    args.push_back(s);
    input_args.push_back(s);
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushIter(Iter * i) {
    PushStreamInterface(i);
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushStream(Stream *s){
    PushStreamInterface(s);
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushConstant(const float &val){
    args.push_back(const_cast<float*>(&val));
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushConstant(const float2 &val){
    args.push_back(const_cast<float2*>(&val));
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushConstant(const float3 &val){
    args.push_back(const_cast<float3*>(&val));
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushConstant(const float4 &val){
    args.push_back(const_cast<float4*>(&val));
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushGatherStream(Stream *s){
    if (!s->isCPU()) {
      s = new CPUStreamShadow(s, ::brook::Stream::READ);
      freeme.push_back(s);
    }

    // Template only determins return type
    // This can be recasted to any other type
    __BrtArray<unsigned char> *array = new __BrtArray<unsigned char>(s);

    args.push_back(array);
    freeme_array.push_back(array);
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushReduce(void * data, StreamType type) {
    if (type==__BRTSTREAM) {
      StreamInterface *s = (StreamInterface *) 
        (* (const brook::stream *) data);

      if (!s->isCPU()) {
        s = new CPUStreamShadow(s, brook::Stream::WRITE);
        freeme.push_back(s);
      }
      args.push_back(s);
      output_args.push_back(s);
      reduce_is_scalar = false;
      reduce_arg = (void *) s;

    } else {
      args.push_back(data);
      reduce_is_scalar = true;
      reduce_arg = data;
    }

  }

  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::PushOutput(Stream *s){
    if (!s->isCPU()) {
      s = new CPUStreamShadow(s, ::brook::Stream::WRITE);
      freeme.push_back(s);
    }
    
    args.push_back(s);
    output_args.push_back(s);
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::Cleanup() {
    args.clear();
    output_args.clear();
    input_args.clear();

    if (curpos)  free(curpos);
    if (extents) free(extents);
    if (minpos)  free(minpos);
    if (maxpos)  free(maxpos);

    curpos = NULL;
    extents = NULL;
    minpos = NULL;
    maxpos = NULL;
    reduce_arg = NULL;
    reduce_value = NULL;

    dims = 0;
    reduce_is_scalar = false;
    is_reduce = false;
    
    for (unsigned int i=0; i<freeme.size(); i++) 
      freeme[i]->releaseReference();

    for (unsigned int i=0; i<freeme_array.size(); i++) 
      delete freeme_array[i];

    freeme.clear();
    freeme_array.clear();
  }
  

  void CPUKernel::Map() {
    unsigned int i,j;

    /* First check to see if all of the outputs 
    ** are of the same size and get the dim */
    assert (output_args.size() > 0);

    const unsigned int *dmin = output_args[0]->getDomainMin();
    const unsigned int *dmax = output_args[0]->getDomainMax();
    dims = output_args[0]->getDimension();

    for (i=1; i<output_args.size(); i++) {
      assert (output_args[i]->getDimension() == dims);
      for (j=0; j<dims; j++) 
        assert (output_args[i]->getDomainMax()[j] - 
                output_args[i]->getDomainMin()[j] ==
                dmax[j] - dmin[j]);
    }

    /* Initialize the position vector */
    curpos  = (unsigned int *) malloc (dims * sizeof (unsigned int));
    extents = (unsigned int *) malloc (dims * sizeof (unsigned int));
    minpos  = (unsigned int *) malloc (dims * sizeof (unsigned int));
    maxpos  = (unsigned int *) malloc (dims * sizeof (unsigned int));

    for (i=0; i<dims; i++) {
      curpos[i] = 0;
      minpos[i] = 0;
    }

    /* Create the extents vector */
    for (i=0; i<dims; i++) {
      int delta = dmax[i] - dmin[i];
      assert (delta >= 0);

      if (delta == 0) {
        // If delta is 0 than the 
        // output stream has a dimension of 0 size.
        Cleanup();
        return;
      }
      else
        extents[i] = delta;
    }

    for (i=0; i<dims; i++)
      maxpos[i] = extents[i];

    // Save the CPU kernel
    current_cpu_kernel = this;
    
    // Call the kernel
    func(this, args);
    
    // Free the resources
    Cleanup();
  }

  void CPUKernel::Reduce() {
    unsigned int i;
    StreamInterface *rstream;
    StreamInterface *instream;

    assert (output_args.size() == 1 ||
            (output_args.size() == 0 && reduce_is_scalar));
    
    assert (input_args.size() == 1);
    
    // These are the reduction arguments
    instream = input_args[0];

    // Set the is_reduce flag
    is_reduce = true;

    // Create the counter arrays
    dims    = instream->getDimension();
    curpos  = (unsigned int *) malloc (dims * sizeof (unsigned int));
    extents = (unsigned int *) malloc (dims * sizeof (unsigned int));
    minpos = (unsigned int *) malloc (dims * sizeof (unsigned int));
    maxpos = (unsigned int *) malloc (dims * sizeof (unsigned int));

    /* Create the extents vector */
    const unsigned int *dmin = instream->getDomainMin();
    const unsigned int *dmax = instream->getDomainMax();
    for (i=0; i<dims; i++) {
      assert (dmax[i] >= dmax[i]);
      // XXX: I have no idea what will happen if delta == 0
      unsigned int delta = dmax[i] - dmin[i];
      extents[i] = delta;
    }

    // Save the CPU kernel
    current_cpu_kernel = this;

    if (reduce_is_scalar) {
      reduce_value = (void *)reduce_arg;
      
      /* Initialize counter */      
      for (i=0; i<dims; i++) {
        curpos[i] = 0;
        minpos[i] = 0;
        maxpos[i] = extents[i];
      }

      // Fetch the initial value
      void *initial = 
        instream->fetchElem(curpos, extents, dims);
      
      // Set the initial value
      memcpy (reduce_value, initial, instream->getElementSize());
      
      // Perform the reduce
      if (Continue())
        func(this, args);
      
      Cleanup();
      return;
    }

    assert( !reduce_is_scalar);
    
    /*  Reduce to Stream   
    **  This is a bit more complicated.
    **  Here we call the reduction kernel on
    **  sub-blocks of the reduction regions.
    */
    
    /* Create the reduction variable vectors */
    rstream = output_args[0];

    // Hopefully we are reduce streams of the same type
    assert (rstream->getElementSize() == 
            instream->getElementSize());
    
    unsigned int reduce_dims   = rstream->getDimension();
    unsigned int *reduce_curpos = (unsigned int *) 
      malloc (reduce_dims * sizeof (unsigned int));
    unsigned int *reduce_extents =  (unsigned int *) 
      malloc (reduce_dims * sizeof (unsigned int));

    // Need to implement!!!
    if (dims != reduce_dims) {
      fprintf (stderr, "Reductions of streams with different "
               "dimensionality not implemented yet, sorry.\n");
      assert (dims == reduce_dims);
      exit(1);
    }
      
    /* Initialize the reduction counters */
    dmin = rstream->getDomainMin();
    dmax = rstream->getDomainMax();
    for (i=0; i<reduce_dims; i++) {
      reduce_curpos[i] = 0;
      assert (dmax[i] > dmin[i]);
      reduce_extents[i] = dmax[i] - dmin[i];
    }

    do {
      bool exit_cond = false;

      // This is the value we are going 
      // to reduce into
      reduce_value = rstream->fetchElem(reduce_curpos,
                                        reduce_extents,
                                        reduce_dims);

      // Compute the curpos which corresponds
      // to this reduction element
      for (i=0; i<dims; i++) {
        if  (extents[i] % reduce_extents[i] != 0) {
          fprintf (stderr, "CPU: Error reduction stream "
                   "not a multiple size of input stream.\n");
          assert (extents[i] % reduce_extents[i] == 0);
          exit(1);
        }
          
        int factor = extents[i] / reduce_extents[i];
        
        curpos[i] = factor * reduce_curpos[i];
        minpos[i] = curpos[i];
        maxpos[i] = factor * (reduce_curpos[i] + 1);
      }
      
      // Fetch the initial value
      void *initial = 
        instream->fetchElem(curpos, extents, dims);
      
      // Set the initial value
      memcpy (reduce_value, initial, rstream->getElementSize());
      
      // Perform the reduce on the sub-block
      if (Continue())
        func(this, args);
      
      // Increment the reduction_curpos
      for (i=0; i<reduce_dims; i++) {
        reduce_curpos[i]++;
        if (reduce_curpos[i] == reduce_extents[i]) {
          if (i == reduce_dims - 1) {
            exit_cond = true;
            break;
          }
          reduce_curpos[i] = 0;
        } else
          break;
      }

      // Exit condition
      if (exit_cond)
        break;

    } while (1);

    // Free the resources
    Cleanup();

    free(reduce_curpos);
    free(reduce_extents);
  }


  bool CPUKernel::Continue() {
    unsigned int i;

    // Increment the curpos
    for (i=dims-1; i>=0; i--) {
      curpos[i]++;
      if (curpos[i] == maxpos[i]) {
        if (i == 0)
          return false;
        curpos[i] = minpos[i];
      } else
        break;
      if (i==0) break;
    }
    return true;
  }

  void * CPUKernel::FetchElem(StreamInterface *s) {
    if (is_reduce &&
        s == (StreamInterface *) reduce_arg) 
      return reduce_value;
   
    // Let the stream do the fetch
    return s->fetchElem(curpos, extents, dims);
  }

  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  void CPUKernel::Release() {
    delete this;
  }
  
  // o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o+o
  CPUKernel::~CPUKernel() {
    // Do nothing
  }
}


