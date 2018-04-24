/*----------------------------------------------------------------------
|
| $Id$
|
+---------------------------------------------------------------------*/

#ifndef  CUDASOLVER_HH
#define  CUDASOLVER_HH

#include "GbDefines.hh"

/*----------------------------------------------------------------------
|       includes
+---------------------------------------------------------------------*/

#include "GbTypes.hh"
#include "GbVec3.hh"
#include "GbVec3i.hh"

#include "solver.h"
#include <vector>

/*----------------------------------------------------------------------
|       declaration
+---------------------------------------------------------------------*/

class CUDASolver
{
public:
    // CUDA implementation of a solver for linear complementarity problems (LCP)
    // this is the C++ interface
    
    // published for GpuGems3 May 2007
    // author Peter Kipfer <peter@kipfer.de>

    // translate status codes
    typedef enum
    {
        SC_FOUND_SOLUTION = CUDA_SC_FOUND_SOLUTION, // solution
        SC_FOUND_TRIVIAL_SOLUTION = CUDA_SC_FOUND_TRIVIAL_SOLUTION, // solution (z = 0, w = q)
        SC_CANNOT_REMOVE_COMPLEMENTARY = CUDA_SC_CANNOT_REMOVE_COMPLEMENTARY, // no solution
        SC_EXCEEDED_MAX_RETRIES = CUDA_SC_EXCEEDED_MAX_ITERATIONS, 
	SC_INVALID = CUDA_SC_INVALID  // no solution (round-off problems?)
    } StatusCode;

    class CollisionPair
    {
    public:
	CollisionPair(unsigned int aa, unsigned int bb, float ra=1.0f, float rb=1.0f) : a(aa), b(bb), reduce_a(ra), reduce_b(rb) {}

	unsigned int a; // index of first object
	unsigned int b; // index of second object
	float reduce_a; // centric scale for first object
	float reduce_b; // centric scale for second object
    };

    CUDASolver();

    // configure is separate from constructor so solver object can be global static
    void configure(int maxNumVertices, int maxNumFaces, int maxNumObjects);

    ~CUDASolver();

    // provide fastest memory available
    // use these methods to allocate space for the pair list, contact points and 
    // status array to allow fast transfer in solve method
    INLINE void* allocSystemMem(unsigned int size);
    INLINE void freeSystemMem(void* mem);

    // store the given data (no GPU communication)
    // can be called any time
    INLINE void storeVertices(int index, int numPoints, const GbVec3<float>* points);
    INLINE void storeIndices(int index, int numIndices, const GbVec3i<int>* indices);

    // download the stored data to the GPU
    // call only when not running GPU computation
    INLINE void commitVertices() const;
    INLINE void commitIndices() const;

    // for the given list of collision pairs, solve for the distance of the
    // objects using LCP. The closest points are returned in the contactPoints
    // array which must therefore have size numPairs
    // Upon return, statusList holds the solution state for every pair.
    void solve(const CollisionPair* pairList, unsigned int numPairs, 
	       GbVec3<float>* contactPoints, 
	       StatusCode* statusList);

private:
    // also use fast mem for own purposes
    INLINE void allocate();
    INLINE void deallocate();

    // these limits must hold for all collision pairs
    int maxNumEquations_;
    int maxNumObjects_;
    int maxNumCollisionPairs_;
    
    // internal storage to allow async CPU/GPU processing
    GbVec3<float>* vertices_;
    GbVec3i<int>* indices_;
};

#ifndef OUTLINE
#include "CUDASolver.in"
#endif

#endif  // CUDASOLVER_HH
/*----------------------------------------------------------------------
|
| $Log$
|
+---------------------------------------------------------------------*/
