// -*- C -*- automatisch in C-mode wechseln (emacs)

// includes, system
#ifdef WIN32
#include "_windows.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "Profiler.hh"
#include "cudaDefines.h"

#include "solver.h"


// shared memory debugging for bank conflicts
// flip the comments in emu mode
// the checks are very slow so you might want to enable one at a time

/* #define CHECK_BANK_BUILD( array, index) CUT_BANK_CHECKER(array,index)   */
#define CHECK_BANK_BUILD( array, index)  array[index]  

/* #define CHECK_BANK_SOLVE( array, index) CUT_BANK_CHECKER(array, index)      */
#define CHECK_BANK_SOLVE( array, index)  array[index]       

/* #define CHECK_BANK_SELECT( array, index) CUT_BANK_CHECKER(array, index)      */
#define CHECK_BANK_SELECT( array, index)  array[index]       


// the solver uses the MSB for signalling whether the
// variable is z or its complement w
#define VARIABLE_SET_Z0(v)      ((v)=0x80000000)
#define VARIABLE_IS_Z0(v)       ((v)==0x80000000)
#define VARIABLE_IS_Z(v)        ((v)&0x80000000)
#define VARIABLE_INDEX(v)       ((v)&0x7fffffff)
#define VARIABLE_BUILD(idx,isz) ((isz) ? ((idx)|0x80000000) : (idx))
#define VARIABLE_BUILD_COMPLEMENT(idx,isz) ((isz) ? (idx) : ((idx)|0x80000000))


// some handy types to make reading easier
#define uint unsigned int
typedef uint CudaEquationVar;
typedef struct __align__(8) 
{ 
    CudaEquationVar Var; 
    int             Equ; 
} CudaNonBasicVariable; 


// host variables
int hostMaxNumEquations = 0;
int hostMaxNumIterations = 100;
int hostMaxNumObjects = 0;
int hostMaxNumPairs = 0;
int* hostStatus = NULL;
float* hostZ = NULL;
bool hostUseFastMem = false;


// device variables
__device__ __constant__ uint deviceMaxNumEquations;
__device__ __constant__ int deviceMaxNumIterations;
int* deviceStatus = NULL;
CudaEquationVar* deviceEquationVar = NULL;
float* deviceEquationC = NULL;
float* deviceEquationW = NULL;
float* deviceEquationZ = NULL;
float* deviceZ = NULL;
float* deviceSolution = NULL;
CudaNonBasicVariable* deviceNonBasicVariable = NULL;
CudaEquationVar* deviceDepartingVariable = NULL;
float* deviceVertices = NULL;
uint* deviceIndices = NULL;
float* devicePairList = NULL;


// local cache
extern __shared__ float arrayShared[];


// better names for CUDA stuff to make reading easier
#define pairId     blockIdx.x
#define pairOffset __umul24(pairId, deviceMaxNumEquations)
#define pairData   __umul24(pairOffset, deviceMaxNumEquations + 1U)
#define eqId     threadIdx.x
#define eqOffset (pairOffset + eqId) 
#define eqData(entry)   (pairData + __umul24(entry, deviceMaxNumEquations) + eqId)
#define isZeroThread (threadIdx.x == 0U)
#define numEquationsPlusOne (numEquations + 1U)



// compute the halfspace constraints of the convex object and put them
// into the matrix M and right-hand side vector Q
// M and Q reside in shared mem and thus can be updated concurrently
__device__ void 
cudaComputeHalfspaces(int numPoints, float* P, float reduce,
		      int numFaces, uint* F, 
		      float* M, float* Q,
		      int numEquations, int rc0, int rc1)
{
    uint eqId3 = __umul24(eqId,3U); 
    // we know there are always more threads than points or faces
    // (asserted in host setup routine)
    float* tempShared = &M[CUDA_MAX_LOCAL_ARRAY_SIZE*CUDA_MAX_LOCAL_ARRAY_SIZE];

    float AvgPtx = 0.0f;
    float AvgPty = 0.0f;
    float AvgPtz = 0.0f;

    // read the vertices to shared mem in parallel
    CHECK_BANK_BUILD(tempShared,eqId3  ) = P[eqId3];   // read from dev mem
    CHECK_BANK_BUILD(tempShared,eqId3+1) = P[eqId3+1U]; // read from dev mem
    CHECK_BANK_BUILD(tempShared,eqId3+2) = P[eqId3+2U]; // read from dev mem

    __syncthreads(); // guard for tempShared

    // compute centroid in shared mem
    for (int j=0; j<numPoints; ++j)
    {
	uint j3 = __umul24(j,3U); 
	AvgPtx += CHECK_BANK_BUILD(tempShared,j3  );
	AvgPty += CHECK_BANK_BUILD(tempShared,j3+1U);
	AvgPtz += CHECK_BANK_BUILD(tempShared,j3+2U);
    }
    float invNumPoints = __fdividef(1.0f,float(numPoints));
    AvgPtx *= invNumPoints;
    AvgPty *= invNumPoints;
    AvgPtz *= invNumPoints;

    // reposition the vertices
    CHECK_BANK_BUILD(tempShared,eqId3   ) = AvgPtx + (CHECK_BANK_BUILD(tempShared,eqId3   ) - AvgPtx) * reduce;
    CHECK_BANK_BUILD(tempShared,eqId3+1U) = AvgPty + (CHECK_BANK_BUILD(tempShared,eqId3+1U) - AvgPty) * reduce;
    CHECK_BANK_BUILD(tempShared,eqId3+2U) = AvgPtz + (CHECK_BANK_BUILD(tempShared,eqId3+2U) - AvgPtz) * reduce;

    __syncthreads(); // guard for tempShared

    // read the faces to shared mem and build the halfspace representation in parallel
    if (eqId < numFaces)
    {
	uint face_a = F[eqId3   ]; // read from device mem
	uint face_b = F[eqId3+1U]; // read from device mem
	uint face_c = F[eqId3+2U]; // read from device mem
	uint face_a3 = __umul24(face_a,3U); 
	uint face_b3 = __umul24(face_b,3U); 
	uint face_c3 = __umul24(face_c,3U); 

	float P0x = CHECK_BANK_BUILD(tempShared,face_a3   );
	float P0y = CHECK_BANK_BUILD(tempShared,face_a3+1U);
	float P0z = CHECK_BANK_BUILD(tempShared,face_a3+2U);

	float P1x = CHECK_BANK_BUILD(tempShared,face_b3   );
	float P1y = CHECK_BANK_BUILD(tempShared,face_b3+1U);
	float P1z = CHECK_BANK_BUILD(tempShared,face_b3+2U);

	float P2x = CHECK_BANK_BUILD(tempShared,face_c3   );
	float P2y = CHECK_BANK_BUILD(tempShared,face_c3+1U);
	float P2z = CHECK_BANK_BUILD(tempShared,face_c3+2U);

	float D1x = P1x - P0x; 
	float D1y = P1y - P0y; 
	float D1z = P1z - P0z; 

	float D2x = P2x - P0x; 
	float D2y = P2y - P0y; 
	float D2z = P2z - P0z; 

	// D1.cross(D2)
	float Nx = D1y * D2z - D1z * D2y;
	float Ny = D1z * D2x - D1x * D2z;
	float Nz = D1x * D2y - D1y * D2x;

	// N.dot(P1)
	float RHS = P1x * Nx + P1y * Ny + P1z * Nz;

	// N.dot(AvgPt)
	float mult = AvgPtx * Nx + AvgPty * Ny + AvgPtz * Nz;
        mult = ((mult <= RHS) ? 1.0f : -1.0f);

	float Ax = mult * Nx;
	float Ay = mult * Ny;
	float Az = mult * Nz;

	// store right-hand side Q
        CHECK_BANK_BUILD(Q,rc0) = mult * RHS;

	// store constraints in matrix M
	uint rc0numEq = __umul24(rc0,numEquations) + rc1; 
	CHECK_BANK_BUILD(M,rc0numEq+0U)                       = -Ax;       // -A
	CHECK_BANK_BUILD(M,__umul24(rc1+0U,numEquations)+rc0) =  Ax;       //  A transpose
	CHECK_BANK_BUILD(M,rc0numEq+1U)                       = -Ay;       // -A
	CHECK_BANK_BUILD(M,__umul24(rc1+1U,numEquations)+rc0) =  Ay;       //  A transpose
	CHECK_BANK_BUILD(M,rc0numEq+2U)                       = -Az;       // -A
	CHECK_BANK_BUILD(M,__umul24(rc1+2U,numEquations)+rc0) =  Az;       //  A transpose
    }
}


// build all equations for a collision pair
// calculate the constraints and initialize the LCP state
__global__ void
cudaBuildEquations(float* pairList, float* vertices, uint* indices,
		   CudaEquationVar* eqVar, float* eqC, float* eqW, float* eqZ, 
		   float* Z, int* status) 
{
    // get the collision pair
    float* pair = pairList + __umul24(pairId,4U);

    // get the two participating objects
    int object1 = __float_as_int(pair[0]) * CUDA_MAX_LOCAL_ARRAY_SIZE * VEC_DIMENSION; // read from device mem
    int object2 = __float_as_int(pair[1]) * CUDA_MAX_LOCAL_ARRAY_SIZE * VEC_DIMENSION; // read from device mem
    float reduce1 = pair[2]; // read from device mem
    float reduce2 = pair[3]; // read from device mem

    // get data for first object
    float *vertices1 = vertices + object1;
    uint *indices1 =  indices + object1;
    int numVertices1 = __float_as_int(vertices1[0]); // read from device mem
    int numIndices1 = indices1[0];                   // read from device mem

    // get data for second object
    float *vertices2 = vertices + object2;
    uint *indices2 = indices + object2;
    int numVertices2 = __float_as_int(vertices2[0]); // read from device mem
    int numIndices2 = indices2[0];                   // read from device mem

    // configure the LCP of this collision pair
    int numEquations = numIndices1 + numIndices2 + TWO_VEC_DIMENSION;

    // split up the shared space for matrix M and vector Q
    float* sharedQ = arrayShared;
    float* sharedM = &arrayShared[CUDA_MAX_LOCAL_ARRAY_SIZE];

    // zero the part of M and Q that we will use in parallel
    CHECK_BANK_BUILD(sharedQ,eqId) = 0.0f;
    for (uint i=0; i<numEquations; ++i)
    {
	CHECK_BANK_BUILD(sharedM,__umul24(eqId,numEquations)+i) = 0.0f;
    }

    __syncthreads(); // guard for sharedM and sharedQ

    // set the matrix constants single threaded
    if (isZeroThread)
    {
	for (uint i = 0; i < TWO_VEC_DIMENSION; i++) 
	{
	    CHECK_BANK_BUILD(sharedM,__umul24(i,numEquations)+i) = 2.0f;
	    if (i < VEC_DIMENSION) 
	    {
		CHECK_BANK_BUILD(sharedM,__umul24(i,numEquations)+i+VEC_DIMENSION) = -2.0f;
		CHECK_BANK_BUILD(sharedM,__umul24(i+VEC_DIMENSION,numEquations)+i) = -2.0f;
	    }
	}
    }	

    // add constraints of first object
    cudaComputeHalfspaces(numVertices1, vertices1 + VEC_DIMENSION, reduce1, 
			  numIndices1, indices1 + VEC_DIMENSION, 
			  sharedM, sharedQ,
			  numEquations, eqId + TWO_VEC_DIMENSION, 0);
    
    __syncthreads(); // guard for sharedM and sharedQ 

    // add constraints of second object
    cudaComputeHalfspaces(numVertices2, vertices2 + VEC_DIMENSION, reduce2, 
			  numIndices2, indices2 + VEC_DIMENSION, 
			  sharedM, sharedQ,
			  numEquations, eqId + TWO_VEC_DIMENSION + numIndices1, VEC_DIMENSION);

    __syncthreads(); // guard for sharedM and sharedQ 

    // calculate minimum Q on all threads
    float QMin = 1.0f;
    for (uint j=0; j<(numIndices1+numIndices2); ++j)
    {
	uint adr = j + TWO_VEC_DIMENSION;
	if (CHECK_BANK_BUILD(sharedQ,adr) < QMin) 
	{
	    QMin = CHECK_BANK_BUILD(sharedQ,adr);
	}
    }

    // clear solution vector in parallel
    Z[eqOffset] = 0.0f; // write to device mem

    // check if all the constant terms are nonnegative.  If so, the solution
    // is trivially z = 0 and w = constant terms
    if (QMin >= 0.0f) 
    {
	// as this has been collected over all equations
	// all threads of the block will run into here in case
	// no need to update other data in device mem, so we can return early
	status[eqOffset] = CUDA_SC_FOUND_TRIVIAL_SOLUTION; // write to device mem
	return;
    }

    // write all valid equations in parallel
    if (eqId < numEquations)
    {
	// mark as valid equation of this collision pair
	status[eqOffset] = CUDA_SC_VALID + numEquations; // write to device mem

	// initially w's are basic, z's are non-basic
	// w indices run from 1 to numEquations
	// the "extra" variable in the equations is z[0]
	// the z flag is the high-bit so we don't need to set it here
	eqVar[eqOffset] = eqId+1U; // write to device mem

	// normalize the equations
	// Find the max abs value of the coefficients on each row and divide
	// each row by that max abs value.
	float temp = fabsf(CHECK_BANK_BUILD(sharedQ,eqId));          // c[0]
	float maxAbs = (temp > 1.0f) ? temp : 1.0f; // c[1..n] are either 0 or 1

	// set the z[0] of all equations to 0.0 in parallel for any matrix row in which all values are 0.0
	float rowOfZeros = 0.0f;
	for (uint j = 0; j < numEquations; j++) 
	{
	    // matrix in shared mem is row major
	    temp = CHECK_BANK_BUILD(sharedM,__umul24(eqId, numEquations) + j);
	    if (temp != 0.0f) 
	    {
		rowOfZeros = 1.0f;
	    }
	    temp = fabsf(temp);
	    maxAbs = (temp > maxAbs) ? temp : maxAbs;
	}

	float invMaxAbs = __fdividef(1.0f, maxAbs); // at least the const coeff of eqId is 1 so no div-by-zero possible

	// now assemble the constant term arrays in parallel
	eqC[eqData(0U)] = CHECK_BANK_BUILD(sharedQ,eqId) * invMaxAbs; // c[0] // write to device mem
	for (uint j = 1; j < numEquationsPlusOne; j++) 
	{
	    eqC[eqData(j)] = 0.0f; // write to device mem
	}
	// this is the equation this thread is responsible for
	eqC[eqData(eqId + 1U)] = invMaxAbs; // write to device mem  

	// now assemble the z term arrays in parallel
	eqZ[eqData(0U)] = rowOfZeros * invMaxAbs; // z[0] // write to device mem
	for (uint j = 0; j < numEquations; j++) 
	{
	    // matrix in shared mem is row major
	    eqZ[eqData(j + 1U)] = CHECK_BANK_BUILD(sharedM,__umul24(eqId, numEquations) + j) * invMaxAbs; // write to device mem
	} 

	// now clear the w term arrays in parallel
	for (uint j = 0; j < numEquationsPlusOne; j++) 
	{
	    eqW[eqData(j)] = 0.0f; // write to device mem
	}

    } // if valid equation
    else
    {
	// mark excess threads as invalid
	status[eqOffset] = CUDA_SC_INVALID; // write to device mem
    }
}


// solve the LCP of the collision pair for the given non-basic variable in parallel
#ifdef SEQUENTIAL_SOLVER_STEPPING
__global__
#else
__device__
#endif
void 
cudaSolveEquation(CudaEquationVar* eqVar, float* eqC, float* eqW, float* eqZ, 
		  CudaNonBasicVariable* solveNonBasicVariable,
		  float* Z, int* status)
{
    // see how many equations this LCP has
    int eqStatusFlag = status[eqOffset]; // read from device mem 

    // temporary space for variables
    // this is explicitly using shared mem to reduce register pressure
    // the two pointers alias the same memory location to allow type casts
    float* forceSharedFloatVars = &arrayShared[CUDA_MAX_LOCAL_ARRAY_SIZE]; 
    int* forceSharedIntVars = (int*)&arrayShared[CUDA_MAX_LOCAL_ARRAY_SIZE]; 

    // force variables into shared memory to reduce register pressure
#define numEquations CHECK_BANK_SOLVE(forceSharedIntVars,0U)
#define invDenom CHECK_BANK_SOLVE(forceSharedFloatVars,1U) 
#define basicVar CHECK_BANK_SOLVE(forceSharedIntVars,2U) 
#define nBVarVar CHECK_BANK_SOLVE(forceSharedIntVars,3U) 
#define nBVarEqu CHECK_BANK_SOLVE(forceSharedIntVars,4U) 
#define thisEquationVar CHECK_BANK_SOLVE(forceSharedIntVars,5U+eqId) 
#define coeff CHECK_BANK_SOLVE(forceSharedFloatVars,5U+CUDA_MAX_LOCAL_ARRAY_SIZE+eqId) 

    // convenient names
#define equToSolve (nBVarEqu - 1U)
#define equToSolveOffset (pairOffset + equToSolve)
#define equToSolveData(entry) (pairData + __umul24(entry, deviceMaxNumEquations) + equToSolve)
#define nonBasicVarIsZ VARIABLE_IS_Z(nBVarVar)
#define nonBasicVarIndex VARIABLE_INDEX(nBVarVar)
#define basicVarIsZ VARIABLE_IS_Z(basicVar)
#define basicVarIndex VARIABLE_INDEX(basicVar)
#define replacementEquationVarIsZ nonBasicVarIsZ
#define replacementEquationVarIndex nonBasicVarIndex

    // setup variables shared by all threads
    if (isZeroThread)   
    {
	// configure the block
	numEquations = eqStatusFlag;

	nBVarVar = solveNonBasicVariable[pairId].Var; // read from device mem  
	nBVarEqu = solveNonBasicVariable[pairId].Equ; // read from device mem  
	if (nonBasicVarIsZ) 
	{
	    invDenom = -eqZ[equToSolveData(nonBasicVarIndex)]; // read from device mem
	}
	else 
	{
	    invDenom = -eqW[equToSolveData(nonBasicVarIndex)]; // read from device mem
	}
 	invDenom = __fdividef(1.0f, invDenom); 

	basicVar = eqVar[equToSolveOffset]; // read from device mem 
    }

    __syncthreads(); // guard for variable setup

    if (numEquations < CUDA_SC_VALID) return;

    // setup per thread variables
    if (eqStatusFlag > CUDA_SC_VALID)
    {
	if (nonBasicVarIsZ) 
	{
	    coeff = eqZ[eqData(nonBasicVarIndex)]; // read from device mem
	}
	else 
	{
	    coeff = eqW[eqData(nonBasicVarIndex)]; // read from device mem
	}
	
	thisEquationVar = eqVar[eqOffset]; // read from device mem 
    }


    //
    // Z terms
    //
    if (eqId < numEquationsPlusOne) 
    {
	CHECK_BANK_SOLVE(arrayShared,eqId) = eqZ[equToSolveData(eqId)] * invDenom; // read from device mem 
    }

    __syncthreads(); // guard for arrayShared
	
    if (isZeroThread) // configure
    {
	if (nonBasicVarIsZ) CHECK_BANK_SOLVE(arrayShared,nonBasicVarIndex) = 0.0f; 
	if (basicVarIsZ)    CHECK_BANK_SOLVE(arrayShared,basicVarIndex) = -invDenom; 
    }

    __syncthreads(); // guard for arrayShared

    if (eqStatusFlag > CUDA_SC_VALID)
    {
	if (eqId != equToSolve) 
	{
	    if (coeff != 0.0f) 
	    {
		for (uint j = 0; j < numEquationsPlusOne; j++) 
		{
		    float eqLocal     = eqZ[eqData(j)]; // read from device mem

		    eqLocal += coeff * CHECK_BANK_SOLVE(arrayShared,j);
		    if (replacementEquationVarIsZ && (j==replacementEquationVarIndex)) eqLocal = 0.0f;
		    eqZ[eqData(j)] = eqLocal; // write to device mem  
		}
	    }
	}
	else 
	{ 
	    for (uint j = 0; j < numEquationsPlusOne; j++) 
	    {
 		eqZ[eqData(j)] = CHECK_BANK_SOLVE(arrayShared,j);   // write to device mem  
	    } 
	    thisEquationVar = nBVarVar;
	}
    }

    __syncthreads(); // guard for arrayShared

    //
    // W terms
    //
    if (eqId < numEquationsPlusOne) 
    {  
	CHECK_BANK_SOLVE(arrayShared,eqId) = eqW[equToSolveData(eqId)] * invDenom; // read from device mem  
    }

    __syncthreads(); // guard for arrayShared

    if (isZeroThread) // configure
    { 
	if (!nonBasicVarIsZ) CHECK_BANK_SOLVE(arrayShared,nonBasicVarIndex) = 0.0f; 
	if (!basicVarIsZ)    CHECK_BANK_SOLVE(arrayShared,basicVarIndex)  = -invDenom; 
    }

    __syncthreads();  // guard for arrayShared

    if (eqStatusFlag > CUDA_SC_VALID)
    { 
	if (eqId != equToSolve) 
	{
	    if (coeff != 0.0f) 
	    {
		for (uint j = 0; j < numEquationsPlusOne; j++)  
		{ 
		    float eqLocal     = eqW[eqData(j)]; // read from device mem
		    eqLocal += coeff * CHECK_BANK_SOLVE(arrayShared,j); 
		    if ((!replacementEquationVarIsZ) && (j==replacementEquationVarIndex)) eqLocal = 0.0f;
		    eqW[eqData(j)] = eqLocal; // write to device mem 

		}
	    }
	}
	else 
	{ 
	    for (uint j = 0; j < numEquationsPlusOne; j++)  
	    { 
		eqW[eqData(j)] = CHECK_BANK_SOLVE(arrayShared,j);    // write to device mem 
	    } 

	}
	
    }

    __syncthreads(); // guard for arrayShared

    // 
    // constant terms 
    // 
    if (eqId < numEquationsPlusOne) 
    { 
	CHECK_BANK_SOLVE(arrayShared,eqId) = eqC[equToSolveData(eqId)] * invDenom; // read from device mem 
    }
	
    __syncthreads(); // guard for arrayShared 

    float cZero = 0.0f; // the potential solution
    if (eqStatusFlag > CUDA_SC_VALID)
    {
	if (eqId != equToSolve) 
	{
	    float eqLocal     = eqC[eqData(0U)]; // read from device mem
	    
	    if (coeff != 0.0f) 
	    {
		eqLocal += coeff * CHECK_BANK_SOLVE(arrayShared,0);
	    }
	    eqC[eqData(0U)] = eqLocal; // write to device mem 

	    cZero = eqLocal;

	    if (coeff != 0.0f) 
	    {
		for (uint j = 1; j < numEquationsPlusOne; j++) 
		{
		    eqLocal     = eqC[eqData(j)]; // read from device mem
		    eqLocal += coeff * CHECK_BANK_SOLVE(arrayShared,j);
		    eqC[eqData(j)] = eqLocal; // write to device mem 
		}
	    }
	}
	else 
	{ 
	    for (uint j = 0; j < numEquationsPlusOne; j++) 
	    {
		eqC[eqData(j)] = CHECK_BANK_SOLVE(arrayShared,j);   // write to device mem 
	    } 
	    cZero = CHECK_BANK_SOLVE(arrayShared,0); 
	}

    } // if valid equation
    
    __syncthreads(); // guard for arrayShared reuse

    // reuse shared mem and change type
    bool* Z0Basic = (bool*)arrayShared;

    // check for solution: determine if z[0] is a basic variable in parallel
    if (VARIABLE_IS_Z0(thisEquationVar))
    {
	CHECK_BANK_SOLVE(Z0Basic,eqId) = true;
    }
    else
    {
	CHECK_BANK_SOLVE(Z0Basic,eqId) = false;
    }
    
    __syncthreads(); // guard for Z0Basic

    // only valid equations may do this
    if (eqStatusFlag > CUDA_SC_VALID) 
    {
	// solution found when z[0] is removed from the basic set
	bool isBasic = false;
	for (int j=0; j<numEquations; ++j)
	{
	    if (CHECK_BANK_SOLVE(Z0Basic,j) == true)
	    {
		isBasic = true;
	    }
	}

	if (!isBasic) 
	{
	    if (VARIABLE_IS_Z(thisEquationVar)) 
	    {
		// const term processing calculated c[0]
		Z[pairOffset + VARIABLE_INDEX(thisEquationVar) - 1U] = cZero;
	    }

	    // pair fully processed
	    status[eqOffset] = CUDA_SC_FOUND_SOLUTION;  // write to device mem
	}

	// store the equation we just solved
	eqVar[eqOffset] = thisEquationVar; // write to device mem

	// don't change status

    } // if valid equation

// undo all the convenient defines to prevent chaos
#undef equToSolve
#undef equToSolveOffset
#undef equToSolveData
#undef nonBasicVarIsZ
#undef nonBasicVarIndex
#undef basicVarIsZ
#undef basicVarIndex
#undef replacementEquationVarIsZ
#undef replacementEquationVarIndex
#undef numEquations
#undef invDenom
#undef basicVar
#undef nBVarVar
#undef nBVarEqu
#undef thisEquationVar
#undef coeff
}


// select the next equation to solve for in parallel
#ifdef SEQUENTIAL_SOLVER_STEPPING
__global__
#else
__device__ 
#endif
void 
cudaSelectEquation(CudaEquationVar* equation, float* eqC, float* eqW, float* eqZ,
		   CudaNonBasicVariable* nonBVar, CudaEquationVar* depVar,
		   int* status)
{
    int numEquations = status[eqOffset]; // read from device mem

    // the variable leaving the dictionary and the one to enter
    CudaEquationVar departingVariable,nonBasicVariable; 
    int nonBasicVariableEqu; 

    // type change on temp space offset
    bool* Z0Basic = (bool*)&arrayShared[CUDA_MAX_LOCAL_ARRAY_SIZE];

    if (numEquations > CUDA_SC_VALID)
    {
	departingVariable = depVar[pairId];  // read from device mem
#define departingVariableIsZ VARIABLE_IS_Z(departingVariable)
#define departingVariableIndex VARIABLE_INDEX(departingVariable)

	nonBasicVariable = nonBVar[pairId].Var; // read from device mem
	nonBasicVariableEqu = nonBVar[pairId].Equ; // read from device mem
#define nonBasicVariableIsZ VARIABLE_IS_Z(nonBasicVariable)
#define nonBasicVariableIndex VARIABLE_INDEX(nonBasicVariable)

	// determine if z[0] is a basic variable in parallel
	CudaEquationVar v = equation[eqOffset]; // read from device mem
	if (VARIABLE_IS_Z0(v))
	{
	    CHECK_BANK_SELECT(Z0Basic,eqId) = true;
	}
	else
	{
	    CHECK_BANK_SELECT(Z0Basic,eqId) = false;
	}
    } // if valid equation

    __syncthreads(); // guard for Z0Basic

    // see which case to take: check for z[0] being basic
    if (numEquations > CUDA_SC_VALID)
    {
	bool isBasic = false;
	for (int j=0; j<numEquations; ++j)
	{
	    if (CHECK_BANK_SELECT(Z0Basic,j) == true)
	    {
		isBasic = true;
	    }
	}

	if (!isBasic) 
	{
	    // z[0] is not basic, find the equation with the smallest (negative)
	    // constant term and solve that equation for z[0]
	    departingVariable = VARIABLE_IS_Z(departingVariable);
	    VARIABLE_SET_Z0(nonBasicVariable);
	}
	else 
	{   
	    // z[0] is basic
	    // Since the departing variable left the dictionary, solve for the complementary variable
	    nonBasicVariable = VARIABLE_BUILD_COMPLEMENT(VARIABLE_INDEX(nonBasicVariable),departingVariableIsZ);
	}

	if (departingVariableIndex != 0) 
	{
	    // record equations with negative coefficients for selected index in parallel
	    uint adr = eqData(departingVariableIndex); 
	    if (nonBasicVariableIsZ) 
	    {
		CHECK_BANK_SELECT(arrayShared,eqId) = eqZ[adr]; // read from device mem
	    }
	    else 
	    {
		CHECK_BANK_SELECT(arrayShared,eqId) = eqW[adr]; // read from device mem
	    }
	}
	else
	{
	    // Special case for nonbasic z[0]
	    // record the valid ratios
 	    CHECK_BANK_SELECT(arrayShared,eqId) = 0.0f; 

	    // get ratios in parallel
	    uint adr = eqData(0U);
	    float zZero = eqZ[adr]; // read from device mem
  	    if (zZero != 0.0f)   
	    {
		CHECK_BANK_SELECT(arrayShared,eqId) = __fdividef(eqC[adr], zZero); // read from device mem
	    }
	}

	if (CHECK_BANK_SELECT(arrayShared,eqId) > -0.00001f) 
	{
	    CHECK_BANK_SELECT(arrayShared,eqId) = CUDA_HUGE_VAL;
	}

    } // valid equation

    __syncthreads(); // guard for arrayShared

    if (numEquations > CUDA_SC_VALID)
    {

	bool eqFound = false;

	if (departingVariableIndex != 0) 
	{
	    // find the limiting equation for variables other than z[0].  The
	    // coefficient of the variable must be negative. We look for the ratio of the
	    // constant polynomial to the negative of the smallest coefficient
	    // of the variable. The constant polynomial must be evaluated to compute this ratio.

	    // find smallest ratio by looping through all equations with valid ratios
	    // and promote the smallest ones to lower power terms until unique
	    int smallest = -1;
	    float minRatio = -CUDA_HUGE_VAL;
	    uint column = 0U;
	    bool needToCheckNextPower = true;

	    while (needToCheckNextPower && (column<numEquationsPlusOne))
	    {
		needToCheckNextPower = false;
		uint row = 0U;
		while (row<numEquations)
		{
		    if (CHECK_BANK_SELECT(arrayShared,row) < 0.0f)
		    {
 			uint adr = pairData + __umul24(column, deviceMaxNumEquations) + row; 
			float ratio = __fdividef(eqC[adr], CHECK_BANK_SELECT(arrayShared,row)); // read from device mem
			float dTemp = ratio - minRatio;
		    
			if (dTemp < -ZERO_TOLERANCE) 
			{
			    // the current min is smaller
			    // invalidate equation
			    CHECK_BANK_SELECT(arrayShared,row) = CUDA_HUGE_VAL;
			}
			else if (dTemp > ZERO_TOLERANCE) 
			{
			    // The new equation has the smallest ratio
			    // make this equation comparison standard
			    // and invalidate other equation
			    if (smallest >=0) 
			    { 
				CHECK_BANK_SELECT(arrayShared,smallest) = CUDA_HUGE_VAL;
			    }
			    smallest = row;
			    minRatio = ratio;
			    // no need to check further
			    needToCheckNextPower = false;
			}
			else 
			{   // the ratios are the same - we found two equal ones
			    needToCheckNextPower = true;
			}
		    }
		    row++;
		}
	    
		column++;
	    }
	    if (smallest >= 0)
	    {
		// unique smallest ratio found
		eqFound = true;
		nonBasicVariableEqu = smallest + 1;
	    }
	}
	else
	{
	    // case for nonbasic z[0]; the coefficients are 1.  Find the
	    // limiting equation when solving for z[0].  At least one c[0] must be
	    // negative initially or we start with a solution.  If all of the
	    // negative constant terms are different, pick the equation with the
	    // smallest (negative) ratio of constant term to the coefficient of
	    // z[0] (least-index rule).
	    float quotMin = CHECK_BANK_SELECT(arrayShared,0);
	    nonBasicVariableEqu = 1;
	    for (int i=1; i<numEquations; ++i)
	    {
		if (CHECK_BANK_SELECT(arrayShared,i) <= quotMin) 
		{
		    quotMin = CHECK_BANK_SELECT(arrayShared,i);
		    nonBasicVariableEqu = i+1;
		}
	    }
	    eqFound = (quotMin < 0.0f);
	}

	if ( eqFound )
	{
	    nonBasicVariable = VARIABLE_BUILD(departingVariableIndex,nonBasicVariableIsZ);
	    departingVariable = equation[pairOffset + nonBasicVariableEqu - 1U]; // read from device mem
	}
	else
	{
	    // unable to remove complementary variable
	    // exclude pair from further processing
	    status[eqOffset] = CUDA_SC_CANNOT_REMOVE_COMPLEMENTARY; // write to device mem
	}

	if (isZeroThread)
	{
	    // only one may update the LCP state
	    nonBVar[pairId].Var = nonBasicVariable;  // write to device mem 
	    nonBVar[pairId].Equ = nonBasicVariableEqu; // write to device mem
	    depVar[pairId]      = departingVariable; // write to device mem 
	}

    } // valid equation

#undef departingVariableIsZ
#undef departingVariableIndex
#undef nonBasicVariableIsZ
#undef nonBasicVariableIndex
}


// check if we need to iterate further
__device__ bool
checkDeviceStatusForDone(int* status)
{
    for (uint i=0; i<deviceMaxNumEquations; ++i)
    {
	if (status[pairOffset + i] > CUDA_SC_VALID) // read from device mem
	    return false;
    }
    return true;
}

// solve all LCPs completely
#ifndef SEQUENTIAL_SOLVER_STEPPING
__global__ void 
cudaProcessEquations(CudaEquationVar* equation, float* eqC, float* eqW, float* eqZ,
		     CudaNonBasicVariable* nonBVar, CudaEquationVar* depVar,
		     float* Z, float* solution, int* status)
{
    for (int iter=0; iter < deviceMaxNumIterations; ++iter) 
    {
	cudaSelectEquation(equation,eqC,eqW,eqZ,
			   nonBVar, depVar,
			   status);

	__syncthreads(); // guard for equation, nonBVar, depVar and status

	cudaSolveEquation(equation,eqC,eqW,eqZ,
			  nonBVar,
			  Z, status);

	__syncthreads(); // guard for eqC, eqW, eqZ, Z and status

	// end threads all or none
	bool done = checkDeviceStatusForDone(status);

	__syncthreads(); // guard for done

	if (done)
	{
	    // store the closest points in packed vector
	    if (eqId < TWO_VEC_DIMENSION)
	    {
		solution[__mul24(pairId, TWO_VEC_DIMENSION) + eqId] = Z[eqOffset]; // read/write device mem
	    }

	    return;
	}
    }

    status[eqOffset] = CUDA_SC_EXCEEDED_MAX_ITERATIONS; // write to device mem

}
#endif


// see if we can get CUDA resources
void initCuda()
{
    CUT_CHECK_DEVICE();

#if __DEVICE_EMULATION__
#ifdef _DEBUG
	printf("CUDA solver using pageable memory\n");
#endif
#else
    // see whether we can use fast memory
    CUresult err = cuInit();

    if (err == CUDA_SUCCESS)
    {
#ifdef _DEBUG
	printf("CUDA solver using pinned memory\n");
#endif
	hostUseFastMem = true;
	
	int count = 0;
	CU_SAFE_CALL( cuDeviceGetCount(&count) );
	if( count <= 0 )
	{
	    printf("Error: CUDA driver cannot find device\n");
	    exit(3);
	}
	
	CUdevice dev;
	//get the compute device
	CU_SAFE_CALL( cuDeviceGet(&dev, 0) );
	
	//create the context
	CU_SAFE_CALL( cuCtxCreate(dev) );
    }
#ifdef _DEBUG
    else
	printf("CUDA solver using pageable memory\n");
#endif
#endif
}

template <class T> T* allocBuffer(unsigned int s)
{
    T *b;
    CUDA_SAFE_CALL(cudaMalloc((void**) &b, s*sizeof(T)));
    return b;
}

void freeBuffer(void *b)
{
    CUDA_SAFE_CALL(cudaFree(b));
}

template <class T> void uploadBuffer(T* device, T* host, unsigned int s)
{
    CUDA_SAFE_CALL(cudaMemcpy(device, host, s*sizeof(T), cudaMemcpyHostToDevice) );
}

template <class T> void downloadBuffer(T* host, T* device, unsigned int s)
{
    CUDA_SAFE_CALL(cudaMemcpy(host, device, s*sizeof(T), cudaMemcpyDeviceToHost) );
}

// set limits on host and device
void setMaxNumEquations(int m)
{
    assert((m+1)<CUDA_MAX_LOCAL_ARRAY_SIZE);
    hostMaxNumEquations = m;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceMaxNumEquations,&hostMaxNumEquations,sizeof(int)));
}

void setMaxNumIterations(int m)
{
    hostMaxNumIterations = m;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceMaxNumIterations,&hostMaxNumIterations,sizeof(int)));
}


// configure the solver
void 
cudaSetup(int maxNumEquations, int maxNumObjects, int maxNumPairs, int maxIter)
{
    if (maxIter < 0) maxIter = 3 * maxNumEquations;
    setMaxNumIterations(maxIter);

    hostMaxNumPairs = maxNumPairs;

    setMaxNumEquations(maxNumEquations);

    hostMaxNumObjects = maxNumObjects;

    deviceStatus = allocBuffer<int>(maxNumEquations*maxNumPairs);
    hostStatus = (int*)malloc(maxNumEquations*maxNumPairs*sizeof(int));

    deviceEquationVar = allocBuffer<CudaEquationVar>(maxNumEquations * maxNumPairs);
    deviceEquationC   = allocBuffer<float>(maxNumEquations * (maxNumEquations+1) * maxNumPairs);
    deviceEquationW   = allocBuffer<float>(maxNumEquations * (maxNumEquations+1) * maxNumPairs);
    deviceEquationZ   = allocBuffer<float>(maxNumEquations * (maxNumEquations+1) * maxNumPairs);

#ifdef SEQUENTIAL_SOLVER_STEPPING
    hostZ = (float*)malloc(maxNumPairs * maxNumEquations * sizeof(float));
#else
    deviceSolution = allocBuffer<float>(TWO_VEC_DIMENSION * maxNumPairs);
#endif
    deviceZ = allocBuffer<float>(maxNumEquations * maxNumPairs);

    deviceNonBasicVariable = allocBuffer<CudaNonBasicVariable>(maxNumPairs);
    deviceDepartingVariable = allocBuffer<CudaEquationVar>(maxNumPairs);

    deviceVertices = allocBuffer<float>(maxNumObjects * CUDA_MAX_LOCAL_ARRAY_SIZE * VEC_DIMENSION);
    deviceIndices  = allocBuffer<unsigned int>(maxNumObjects * CUDA_MAX_LOCAL_ARRAY_SIZE * VEC_DIMENSION); 

    devicePairList = allocBuffer<float>(maxNumPairs * 4); // pair has 4 components
}

void
cudaUploadVertices(float* vertices)
{
    uploadBuffer<float>(deviceVertices,vertices,hostMaxNumObjects * CUDA_MAX_LOCAL_ARRAY_SIZE * VEC_DIMENSION);
}

void
cudaUploadIndices(unsigned int* indices)
{
    uploadBuffer<unsigned int>(deviceIndices,indices,hostMaxNumObjects * CUDA_MAX_LOCAL_ARRAY_SIZE * VEC_DIMENSION);
}


// free GPU resources
void
cudaShutdown()
{
    hostMaxNumObjects = 0;
    hostMaxNumPairs = 0;

    freeBuffer(deviceStatus); deviceStatus = NULL;
    free(hostStatus); hostStatus = NULL;

    freeBuffer(deviceEquationVar); deviceEquationVar = NULL;
    freeBuffer(deviceEquationC);   deviceEquationC   = NULL;
    freeBuffer(deviceEquationW);   deviceEquationW   = NULL;
    freeBuffer(deviceEquationZ);   deviceEquationZ   = NULL;

    freeBuffer(deviceZ); deviceZ = NULL;
#ifdef SEQUENTIAL_SOLVER_STEPPING
    free(hostZ); hostZ = NULL;
#else
    freeBuffer(deviceSolution); deviceSolution = NULL;
#endif

    freeBuffer(deviceNonBasicVariable); deviceNonBasicVariable = NULL;
    freeBuffer(deviceDepartingVariable); deviceDepartingVariable = NULL;

    freeBuffer(deviceVertices); deviceVertices = NULL;
    freeBuffer(deviceIndices); deviceIndices = NULL;

    freeBuffer(devicePairList); devicePairList = NULL;

    if (hostUseFastMem)
    {
	//detach from the context
	CU_SAFE_CALL( cuCtxDetach() );
    }
}


// little helper routine for sequential mode
bool
checkStatusForDone(unsigned int numPairs)
{
    downloadBuffer<int>(hostStatus,deviceStatus,numPairs*hostMaxNumEquations);
    for (unsigned int i=0; i<numPairs; ++i)
    {
	if (hostStatus[i*hostMaxNumEquations] >= 0)
	    return false;
    }
    return true;
}


// nice debug output
void
printEquationStatus(unsigned int numPairs, const char* text)
{
    printf("%s ",text);
    downloadBuffer<int>(hostStatus,deviceStatus,numPairs*hostMaxNumEquations);
    for (int i=0; i<numPairs; ++i)
    {
	CUDAStatusCode s = (CUDAStatusCode)hostStatus[i*hostMaxNumEquations];
	if (s == CUDA_SC_FOUND_SOLUTION)
	    printf("S");
	else if (s == CUDA_SC_CANNOT_REMOVE_COMPLEMENTARY)
	    printf("C");
	else if (s == CUDA_SC_FOUND_TRIVIAL_SOLUTION)
	    printf("T");
	else if (s == CUDA_SC_INVALID)
	    printf("I");
	else if (s == CUDA_SC_EXCEEDED_MAX_ITERATIONS)
	    printf("X");
	else if (int(s) > 0)
	    printf(".");
	else
	    printf("-");
    }
    printf("\n");
}


// the solver frontend call
CUDAStatusCode
cudaSolve(unsigned int numPairs, float* contactPointsList, CUDAStatusCode* statusList)
{
    CUDAStatusCode rStatus = CUDA_SC_FOUND_SOLUTION; 
    dim3  eqThreads(hostMaxNumEquations,1);
    dim3  grid(numPairs,1);
#ifdef _DEBUG
    printf("%i pairs, %i threads/block\n",numPairs,hostMaxNumEquations);     
#endif

    size_t sharedMemBuild = ( CUDA_MAX_LOCAL_ARRAY_SIZE + // sharedQ
			      (CUDA_MAX_LOCAL_ARRAY_SIZE*CUDA_MAX_LOCAL_ARRAY_SIZE) + // sharedM
			      (CUDA_MAX_LOCAL_ARRAY_SIZE*VEC_DIMENSION) ) // temp for buidling halfspaces
	* sizeof(float);

    size_t sharedMemSelect = CUDA_MAX_LOCAL_ARRAY_SIZE*sizeof(float) + // validValues
	CUDA_MAX_LOCAL_ARRAY_SIZE*sizeof(bool); // Z0Basic

    size_t sharedMemSolve = (CUDA_MAX_LOCAL_ARRAY_SIZE +  // replacementEquation, reused for Z0Basic
			     5 +                          // forced shared common variables
			     CUDA_MAX_LOCAL_ARRAY_SIZE +  // forced shared variable thisEquationVar
			     CUDA_MAX_LOCAL_ARRAY_SIZE)   // forced shared variable coeff
	*sizeof(float);

#ifdef _DEBUG
    printf("%i shmem\n",sharedMemBuild);  
#endif
    CUT_CHECK_ERROR("before build"); 
    cudaBuildEquations<<<grid,eqThreads,sharedMemBuild>>>(devicePairList, deviceVertices, deviceIndices,
							  deviceEquationVar,deviceEquationC,deviceEquationW,deviceEquationZ, 
							  deviceZ, deviceStatus); 
    CUT_CHECK_ERROR("after build"); 

#ifdef _DEBUG
    printEquationStatus(numPairs,"init  ");
#endif

#if DISPLAY_PERFORMANCE
    static float avgTime = -1.0f;
    unsigned long long numPairsLong = (unsigned long long)numPairs;
#endif

#ifdef SEQUENTIAL_SOLVER_STEPPING

    unsigned long long tistart = startProfile();

    int iter = 0;
    for (; iter < hostMaxNumIterations; ++iter) 
    {

#ifdef _DEBUG
  	printf("%i shmem\n",sharedMemSelect+48);  
#endif
	CUT_CHECK_ERROR("before select"); 
	cudaSelectEquation<<<grid,eqThreads,sharedMemSelect>>>(deviceEquationVar,deviceEquationC,deviceEquationW,deviceEquationZ,
							       deviceNonBasicVariable, deviceDepartingVariable,
							       deviceStatus);
	CUT_CHECK_ERROR("after select"); 

#ifdef _DEBUG
	printEquationStatus(numPairs,"select");
    	printf("%i shmem\n",sharedMemSolve+48);    
#endif
	CUT_CHECK_ERROR("before solve"); 
	cudaSolveEquation<<<grid,eqThreads,sharedMemSolve>>>(deviceEquationVar,deviceEquationC,deviceEquationW,deviceEquationZ,
							     deviceNonBasicVariable,
							     deviceZ, deviceStatus);
	CUT_CHECK_ERROR("after solve"); 

#ifdef _DEBUG
	printEquationStatus(numPairs,"solve ");
#endif

	if (checkStatusForDone(numPairs))
	{
	    break;
	}
	    
    } // for < hostMaxIterations

    unsigned long long tistop = endProfile(tistart);

    // get Z array and build packed solution vector
    downloadBuffer<float>(hostZ,deviceZ,numPairs*hostMaxNumEquations);
    for (unsigned int i=0; i<numPairs; ++i)
    {
	memcpy(contactPointsList+(i*TWO_VEC_DIMENSION), hostZ+(i*hostMaxNumEquations), TWO_VEC_DIMENSION*sizeof(float));
    }

    if (iter == hostMaxNumIterations) 
    {
	rStatus = CUDA_SC_EXCEEDED_MAX_ITERATIONS; 
	iter--;
    }

#ifdef _DEBUG
    printf("solver needed %i iterations, ",iter+1);
    switch (rStatus)
    {
	case CUDA_SC_FOUND_SOLUTION:
	    printf("found solution\n");
	    break;
	case CUDA_SC_CANNOT_REMOVE_COMPLEMENTARY:
	    printf("cannot remove complementary\n");
	    break;
	case CUDA_SC_EXCEEDED_MAX_ITERATIONS:
	    printf("max iter exceeded\n");
	    break;
	default:
	    printf("other status (%i)\n",rStatus);
	    break;
    }
#endif


#else

    // max of select and solve
    size_t sharedMemProcess = (sharedMemSelect > sharedMemSolve) ? sharedMemSelect : sharedMemSolve;
#ifdef _DEBUG
    printf("%i shmem\n",sharedMemProcess+64);     
#endif
    CUT_CHECK_ERROR("before process"); 
    unsigned long long tistart = startProfile();
    cudaProcessEquations<<<grid,eqThreads,sharedMemProcess>>>(deviceEquationVar,deviceEquationC,deviceEquationW,deviceEquationZ,
							      deviceNonBasicVariable, deviceDepartingVariable,
							      deviceZ, deviceSolution, deviceStatus);
    unsigned long long tistop = endProfile(tistart);
    CUT_CHECK_ERROR("after process"); 

#ifdef _DEBUG
    printEquationStatus(numPairs,"solved");
#endif

    // get solution, packed vector already arranged on GPU
    downloadBuffer<float>(contactPointsList,deviceSolution,TWO_VEC_DIMENSION*numPairs);

#endif

    downloadBuffer<int>(hostStatus,deviceStatus,numPairs*hostMaxNumEquations);
    for (int i=0; i<numPairs; ++i)
    {
	statusList[i] = (CUDAStatusCode)hostStatus[i*hostMaxNumEquations];
    }

#if DISPLAY_PERFORMANCE
    {
	unsigned long long clockspeed = 3000000000UL; // 3.0 GHz
	if (avgTime < 0.0f)  
	    avgTime = float(tistop/numPairsLong);
	else  
	    avgTime = float(tistop/numPairsLong) * 0.2f + avgTime * 0.8f; 
	printf("CPU: avg time per pair % 7.0f, pairs/sec %i\n",avgTime,clockspeed/(unsigned long long)int(avgTime));
    }
#endif

    return rStatus;
}

void
cudaUploadPairList(float* pairList, unsigned int numPairs)
{
    assert(numPairs <= hostMaxNumPairs);
    uploadBuffer<float>(devicePairList,pairList,numPairs*4); 
}

// in pinned memory mode use special functions to get OS-pinned memory 
void*
allocHostMem(unsigned int s)
{
    void* mem;

    if (hostUseFastMem)
	CU_SAFE_CALL( cuMemAllocSystem( (void**)&mem, s ) ); 
    else
	mem = malloc(s);

    return mem;
}

void
freeHostMem(void* mem)
{
    if (hostUseFastMem)
	CU_SAFE_CALL( cuMemFreeSystem(mem) ); 
    else
	free(mem);
}
