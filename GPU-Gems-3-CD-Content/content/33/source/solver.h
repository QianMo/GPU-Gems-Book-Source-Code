// this is the C interface of the LCP solver

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_MAX_LOCAL_ARRAY_SIZE 48U
#define CUDA_MAX_NUM_VERTICES 16U
#define VEC_DIMENSION 3U
#define TWO_VEC_DIMENSION (2U*VEC_DIMENSION)

#define ZERO_TOLERANCE 0.0f
#define CUDA_HUGE_VAL 999999.0f 

    typedef enum // status codes
    {
	CUDA_SC_VALID = 0,
	CUDA_SC_INVALID = -1,
        CUDA_SC_FOUND_SOLUTION = -2,               // solution
        CUDA_SC_FOUND_TRIVIAL_SOLUTION = -3,       // solution (z = 0, w = q)
        CUDA_SC_CANNOT_REMOVE_COMPLEMENTARY = -4,  // no solution (unbounded)
        CUDA_SC_EXCEEDED_MAX_ITERATIONS = -5       // no solution (round-off problems?)
    } CUDAStatusCode;

    // system management
    void initCuda();
    void* allocHostMem(unsigned int s);
    void freeHostMem(void* mem);

    // solver configuration
    void cudaSetup(int numEquations, int maxNumObjects, int maxNumPairs, int maxIter=-1);
    void cudaShutdown();

    // communication
    void cudaUploadVertices(float* vertices);
    void cudaUploadIndices(unsigned int* indices);
    void cudaUploadPairList(float* pairList, unsigned int numPairs);

    // execute
    CUDAStatusCode cudaSolve(unsigned int numPairs, float* contactPointsList, CUDAStatusCode* statusList);

#ifdef __cplusplus
}
#endif
