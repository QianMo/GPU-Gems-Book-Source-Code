#ifdef USE_RCSID
static const char RCSid_CUDASolver[] = "$Id$";
#endif

/*----------------------------------------------------------------------
|
|
| $Log$
|
|
+---------------------------------------------------------------------*/

#include "CUDASolver.hh"

#ifdef OUTLINE
#include "CUDASolver.in"
#endif 

CUDASolver::~CUDASolver()
{
    cudaShutdown();
    deallocate();
    debugmsg("solver shutdown complete");
}

CUDASolver::CUDASolver ()
{
}

void
CUDASolver::configure(int maxNumVertices, int maxNumFaces, int maxNumObjects)
{   
    maxNumEquations_ = 2*maxNumFaces + 2*VEC_DIMENSION;
    maxNumObjects_ = maxNumObjects;
    maxNumCollisionPairs_ = (maxNumObjects-1)*(maxNumObjects-1); // cannot collide with self

    // assert solver multithreading requirements
    assert(maxNumVertices+1 <= (int)CUDA_MAX_LOCAL_ARRAY_SIZE);   // add one for #vertices
    assert(maxNumFaces+1 <= (int)CUDA_MAX_LOCAL_ARRAY_SIZE);      // add one for #indices
    assert(maxNumEquations_+1 <= (int)CUDA_MAX_LOCAL_ARRAY_SIZE); // add one for zero variable

    assert(maxNumVertices <= (int)CUDA_MAX_NUM_VERTICES);
    assert(maxNumVertices <= maxNumEquations_);
    assert(maxNumFaces <= maxNumEquations_);

    initCuda();
    allocate();
    cudaSetup(maxNumEquations_,maxNumObjects_,maxNumCollisionPairs_);

    debugmsg("solver setup complete: maxNumEquations="<<maxNumEquations_<<" maxNumObjects="<<maxNumObjects_<<" maxNumPairs="<<maxNumCollisionPairs_);
}

void
CUDASolver::solve(const CollisionPair* pairList, unsigned int numPairs,
				    GbVec3<float>* contactPoints, 
				    StatusCode* statusList)
{
    if (numPairs < 1) return;

    cudaUploadPairList((float*)pairList, numPairs);

    CUDAStatusCode iReturn = cudaSolve(numPairs, (float*)contactPoints, (CUDAStatusCode*)statusList);

    if (iReturn != CUDA_SC_FOUND_SOLUTION)
	errormsg("solution error");

#ifdef DEBUG
    debugmsg("status after cudaSolve ("<<numPairs<<" pairs)");
    for (unsigned int p=0; p<numPairs; ++p)
    {
	if (statusList[p] == SC_FOUND_SOLUTION)
	    std::cerr<<"S";
	else if (statusList[p] == SC_CANNOT_REMOVE_COMPLEMENTARY)
	    std::cerr<<"C";
	else if (statusList[p] == SC_FOUND_TRIVIAL_SOLUTION)
	    std::cerr<<"T";
	else if (statusList[p] == SC_INVALID)
	    std::cerr<<"I";
	else if (statusList[p] == SC_EXCEEDED_MAX_RETRIES)
	    std::cerr<<"X";
	else 
	    std::cerr<<"-";
    }
    std::cerr<<std::endl;
#endif
}

