/*
* Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/


/* CUda UTility Library */

#ifndef _CUT_
#define _CUT_

#ifdef _WIN32
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // helper typedefs for building DLL
#ifdef _WIN32
#  ifdef BUILD_DLL
#    define DLL_MAPPING  __declspec(dllexport)
#  else
#    define DLL_MAPPING  __declspec(dllimport)
#  endif
#else 
#  define DLL_MAPPING 
#endif

    ////////////////////////////////////////////////////////////////////////////////
    //! CUT bool type
    ////////////////////////////////////////////////////////////////////////////////
    enum CUTBoolean 
    {
        CUTFalse = 0,
        CUTTrue = 1
    };

    ////////////////////////////////////////////////////////////////////////////////
    //! Deallocate memory allocated within Cutil
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        void
        cutFree( void* ptr);

    ////////////////////////////////////////////////////////////////////////////////
    //! Helper for bank conflict checking (should only be used with the
    //! CUT_BANK_CHECKER macro)
    //! @param tidx  thread id in x dimension of block
    //! @param tidy  thread id in y dimension of block
    //! @param tidz  thread id in z dimension of block
    //! @param bdimx block size in x dimension
    //! @param bdimy block size in y dimension
    //! @param bdimz block size in z dimension
    //! @param file  name of the source file where the access takes place
    //! @param line  line in the source file where the access takes place
    //! @param aname name of the array which is accessed
    //! @param index index into the array
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        void
        cutCheckBankAccess( unsigned int tidx, unsigned int tidy, unsigned int tidz,
                            unsigned int bdimx, unsigned int bdimy, unsigned int bdimz,
                            const char* file, const int line, const char* aname,
                            const int index);

    //////////////////////////////////////////////////////////////////////////////
    //! Find the path for a filename
    //! @return the path if succeeded, otherwise 0
    //! @param filename        name of the file
    //! @param executablePath  optional absolute path of the executable
    //////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        char*
        cutFindFilePath(const char* filename, const char* executablePath);

    ////////////////////////////////////////////////////////////////////////////////
    //! Read file \filename containing single precision floating point data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutReadFilef( const char* filename, float** data, unsigned int* len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Read file \filename containing double precision floating point data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutReadFiled( const char* filename, double** data, unsigned int* len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Read file \filename containing integer data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutReadFilei( const char* filename, int** data, unsigned int* len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Read file \filename containing unsigned integer data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutReadFileui( const char* filename, unsigned int** data, unsigned int* len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Read file \filename containing char / byte data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutReadFileb( const char* filename, char** data, unsigned int* len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Read file \filename containing unsigned char / byte data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutReadFileub( const char* filename, unsigned char** data, unsigned int* len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename containing single precision floating point data
    //! @return CUTTrue if writing the file succeeded, otherwise false
    //! @param filename name of the file to write
    //! @param data  pointer to data to write
    //! @param len  number of data elements in data, -1 on error
    //! @param epsilon  epsilon for comparison
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutWriteFilef( const char* filename, const float* data, unsigned int len,
                       const float epsilon, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename containing double precision floating point data
    //! @return CUTTrue if writing the file succeeded, otherwise false
    //! @param filename name of the file to write
    //! @param data  pointer to data to write
    //! @param len  number of data elements in data, -1 on error
    //! @param epsilon  epsilon for comparison
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutWriteFiled( const char* filename, const float* data, unsigned int len,
                       const double epsilon, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename containing integer data
    //! @return CUTTrue if writing the file succeeded, otherwise false
    //! @param filename name of the file to write
    //! @param data  pointer to data to write
    //! @param len  number of data elements in data, -1 on error
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutWriteFilei( const char* filename, const int* data, unsigned int len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename containing unsigned integer data
    //! @return CUTTrue if writing the file succeeded, otherwise false
    //! @param filename name of the file to write
    //! @param data  pointer to data to write
    //! @param len  number of data elements in data, -1 on error
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutWriteFileui( const char* filename,const unsigned int* data,unsigned int len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename containing char / byte data
    //! @return CUTTrue if writing the file succeeded, otherwise false
    //! @param filename name of the file to write
    //! @param data  pointer to data to write
    //! @param len  number of data elements in data, -1 on error
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutWriteFileb( const char* filename, const char* data, unsigned int len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename containing unsigned char / byte data
    //! @return CUTTrue if writing the file succeeded, otherwise false
    //! @param filename name of the file to write
    //! @param data  pointer to data to write
    //! @param len  number of data elements in data, -1 on error
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutWriteFileub(const char* filename,const unsigned char* data,unsigned int len, bool verbose = false);

    ////////////////////////////////////////////////////////////////////////////////
    //! Load PGM image file (with unsigned char as data element type)
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutLoadPGMub( const char* file, unsigned char** data,
                      unsigned int *w,unsigned int *h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Load PPM image file (with unsigned char as data element type)
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean
    cutLoadPPMub( const char* file, unsigned char** data, 
                  unsigned int *w,unsigned int *h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Load PPM image file (with unsigned char as data element type), padding 4th component
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean
    cutLoadPPM4ub( const char* file, unsigned char** data, 
                   unsigned int *w,unsigned int *h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Load PGM image file (with unsigned int as data element type)
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutLoadPGMi( const char* file, unsigned int** data, 
                     unsigned int* w, unsigned int* h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Load PGM image file (with unsigned short as data element type)
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutLoadPGMs( const char* file, unsigned short** data, 
                     unsigned int* w, unsigned int* h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Load PGM image file (with float as data element type)
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    //! @note If a NULL pointer is passed to this function and it is initialized 
    //!       withing Cutil then cutFree() has to be used to deallocate the memory
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutLoadPGMf( const char* file, float** data,
                     unsigned int* w, unsigned int* h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Save PGM image file (with unsigned char as data element type)
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutSavePGMub( const char* file, unsigned char* data, 
                      unsigned int w, unsigned int h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Save PPM image file (with unsigned char as data element type)
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean
    cutSavePPMub( const char* file, unsigned char *data, 
                unsigned int w, unsigned int h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Save PPM image file (with unsigned char as data element type, padded to 4 bytes)
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
    CUTBoolean
    cutSavePPM4ub( const char* file, unsigned char *data, 
                   unsigned int w, unsigned int h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Save PGM image file (with unsigned int as data element type)
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutSavePGMi( const char* file, unsigned int* data,
                     unsigned int w, unsigned int h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Save PGM image file (with unsigned short as data element type)
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutSavePGMs( const char* file, unsigned short* data,
                     unsigned int w, unsigned int h);

    ////////////////////////////////////////////////////////////////////////////////
    //! Save PGM image file (with float as data element type)
    //! @param file  name of the image file
    //! @param data  handle to the data read
    //! @param w     width of the image
    //! @param h     height of the image
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutSavePGMf( const char* file, float* data,
                     unsigned int w, unsigned int h);

    ////////////////////////////////////////////////////////////////////////////////
    // Command line arguments: General notes
    // * All command line arguments begin with '--' followed by the token; 
    //   token and value are seperated by '='; example --samples=50
    // * Arrays have the form --model=[one.obj,two.obj,three.obj] 
    //   (without whitespaces)
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    //! Check if command line argument \a flag-name is given
    //! @return CUTTrue if command line argument \a flag_name has been given, 
    //!         otherwise 0
    //! @param argc  argc as passed to main()
    //! @param argv  argv as passed to main()
    //! @param flag_name  name of command line flag
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutCheckCmdLineFlag( const int argc, const char** argv, const char* flag_name);

    ////////////////////////////////////////////////////////////////////////////////
    //! Get the value of a command line argument of type int
    //! @return CUTTrue if command line argument \a arg_name has been given and
    //!         is of the requested type, otherwise CUTFalse
    //! @param argc  argc as passed to main()
    //! @param argv  argv as passed to main()
    //! @param arg_name  name of the command line argument
    //! @param val  value of the command line argument
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutGetCmdLineArgumenti( const int argc, const char** argv, 
                                const char* arg_name, int* val);

    ////////////////////////////////////////////////////////////////////////////////
    //! Get the value of a command line argument of type float
    //! @return CUTTrue if command line argument \a arg_name has been given and
    //!         is of the requested type, otherwise CUTFalse
    //! @param argc  argc as passed to main()
    //! @param argv  argv as passed to main()
    //! @param arg_name  name of the command line argument
    //! @param val  value of the command line argument
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutGetCmdLineArgumentf( const int argc, const char** argv, 
                                const char* arg_name, float* val);

    ////////////////////////////////////////////////////////////////////////////////
    //! Get the value of a command line argument of type string
    //! @return CUTTrue if command line argument \a arg_name has been given and
    //!         is of the requested type, otherwise CUTFalse
    //! @param argc  argc as passed to main()
    //! @param argv  argv as passed to main()
    //! @param arg_name  name of the command line argument
    //! @param val  value of the command line argument
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutGetCmdLineArgumentstr( const int argc, const char** argv, 
                                  const char* arg_name, char** val);

    ////////////////////////////////////////////////////////////////////////////////
    //! Get the value of a command line argument list those element are strings
    //! @return CUTTrue if command line argument \a arg_name has been given and
    //!         is of the requested type, otherwise CUTFalse
    //! @param argc  argc as passed to main()
    //! @param argv  argv as passed to main()
    //! @param arg_name  name of the command line argument
    //! @param val  command line argument list
    //! @param len  length of the list / number of elements
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutGetCmdLineArgumentListstr( const int argc, const char** argv, 
                                      const char* arg_name, char** val, 
                                      unsigned int* len);

    ////////////////////////////////////////////////////////////////////////////////
    //! Check for OpenGL error
    //! @return CUTTrue if no GL error has been encountered, otherwise 0
    //! @param file  __FILE__ macro
    //! @param line  __LINE__ macro
    //! @note The GL error is listed on stderr
    //! @note This function should be used via the CHECK_ERROR_GL() macro
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutCheckErrorGL( const char* file, const int line);

    ////////////////////////////////////////////////////////////////////////////////
    //! Extended assert
    //! @return CUTTrue if the condition \a val holds, otherwise CUTFalse
    //! @param val  condition to test
    //! @param file  __FILE__ macro
    //! @param line  __LINE__ macro
    //! @note This function should be used via the CONDITION(val) macro
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean
        cutCheckCondition( int val, const char* file, const int line);

    ////////////////////////////////////////////////////////////////////////////////
    //! Compare two float arrays
    //! @return  CUTTrue if \a reference and \a data are identical, 
    //!          otherwise CUTFalse
    //! @param reference  handle to the reference data / gold image
    //! @param data       handle to the computed data
    //! @param len        number of elements in reference and data
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutComparef( const float* reference, const float* data, const unsigned int len);

    ////////////////////////////////////////////////////////////////////////////////
    //! Compare two integer arrays
    //! @return  CUTTrue if \a reference and \a data are identical, 
    //!          otherwise CUTFalse
    //! @param reference  handle to the reference data / gold image
    //! @param data       handle to the computed data
    //! @param len        number of elements in reference and data
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING 
        CUTBoolean 
        cutComparei( const int* reference, const int* data, const unsigned int len ); 

    ////////////////////////////////////////////////////////////////////////////////
    //! Compare two unsigned char arrays
    //! @return  CUTTrue if \a reference and \a data are identical, 
    //!          otherwise CUTFalse
    //! @param reference  handle to the reference data / gold image
    //! @param data       handle to the computed data
    //! @param len        number of elements in reference and data
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING 
        CUTBoolean 
        cutCompareub( const unsigned char* reference, const unsigned char* data,
                      const unsigned int len ); 

    ////////////////////////////////////////////////////////////////////////////////
    //! Compare two float arrays with an epsilon tolerance for equality
    //! @return  CUTTrue if \a reference and \a data are identical, 
    //!          otherwise CUTFalse
    //! @param reference  handle to the reference data / gold image
    //! @param data       handle to the computed data
    //! @param len        number of elements in reference and data
    //! @param epsilon    epsilon to use for the comparison
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutComparefe( const float* reference, const float* data,
                      const unsigned int len, const float epsilon );

    ////////////////////////////////////////////////////////////////////////////////
    //! Compare two float arrays using L2-norm with an epsilon tolerance for equality
    //! @return  CUTTrue if \a reference and \a data are identical, 
    //!          otherwise CUTFalse
    //! @param reference  handle to the reference data / gold image
    //! @param data       handle to the computed data
    //! @param len        number of elements in reference and data
    //! @param epsilon    epsilon to use for the comparison
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutCompareL2fe( const float* reference, const float* data,
                        const unsigned int len, const float epsilon );

    ////////////////////////////////////////////////////////////////////////////////
    //! Timer functionality

    ////////////////////////////////////////////////////////////////////////////////
    //! Create a new timer
    //! @return CUTTrue if a time has been created, otherwise false
    //! @param  name of the new timer, 0 if the creation failed
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutCreateTimer( unsigned int* name);

    ////////////////////////////////////////////////////////////////////////////////
    //! Delete a timer
    //! @return CUTTrue if a time has been deleted, otherwise false
    //! @param  name of the timer to delete
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutDeleteTimer( unsigned int name);

    ////////////////////////////////////////////////////////////////////////////////
    //! Start the time with name \a name
    //! @param name  name of the timer to start
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutStartTimer( const unsigned int name);

    ////////////////////////////////////////////////////////////////////////////////
    //! Stop the time with name \a name. Does not reset.
    //! @param name  name of the timer to stop
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutStopTimer( const unsigned int name);

    ////////////////////////////////////////////////////////////////////////////////
    //! Resets the timer's counter.
    //! @param name  name of the timer to reset.
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        CUTBoolean 
        cutResetTimer( const unsigned int name);

    ////////////////////////////////////////////////////////////////////////////////
    //! Returns total execution time in milliseconds for the timer over all runs 
    //! since the last reset or timer creation.
    //! @param name  name of the timer to return the time of
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        float 
        cutGetTimerValue( const unsigned int name);

    ////////////////////////////////////////////////////////////////////////////////
    //! Return the average time in milliseconds for timer execution as the total 
    //! time for the timer dividied by the number of completed (stopped) runs the 
    //! timer has made.
    //! Excludes the current running time if the timer is currently running.
    //! @param name  name of the timer to return the time of
    ////////////////////////////////////////////////////////////////////////////////
    DLL_MAPPING
        float 
        cutGetAverageTimerValue( const unsigned int name);

    ////////////////////////////////////////////////////////////////////////////////
    //! Macros

#ifdef _DEBUG

#if __DEVICE_EMULATION__
    // Interface for bank conflict checker
#define CUT_BANK_CHECKER( array, index)                                      \
    (cutCheckBankAccess( threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x,  \
    blockDim.y, blockDim.z,                                                  \
    __FILE__, __LINE__, #array, index ),                                     \
    array[index])
#else
#define CUT_BANK_CHECKER( array, index)  array[index]
#endif

#  define CU_SAFE_CALL( call ) do {                                          \
    CUresult err = call;                                                     \
    if( CUDA_SUCCESS != err) {                                               \
        fprintf(stderr, "Cuda driver error %x in file '%s' in line %i.\n",   \
                err, __FILE__, __LINE__ );                                   \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)

#  define CUFFT_SAFE_CALL( call) do {                                        \
    cufftResult err = call;                                                  \
    if( CUFFT_SUCCESS != err) {                                              \
        fprintf(stderr, "CUFFT error in file '%s' in line %i.\n",            \
                __FILE__, __LINE__);                                         \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUT_SAFE_CALL( call)                                               \
    if( CUTTrue != call) {                                                   \
        fprintf(stderr, "Cut error in file '%s' in line %i.\n",              \
                __FILE__, __LINE__);                                         \
        exit(EXIT_FAILURE);                                                  \
    } 

    //! Check for CUDA error
#  define CUT_CHECK_ERROR(errorMessage) do {                                 \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

    //! Check for OpenGL error
#  define CUT_CHECK_ERROR_GL()                                               \
    if( CUTFalse == cutCheckErrorGL( __FILE__, __LINE__)) {                  \
        exit(EXIT_FAILURE);                                                  \
    }

    //! Check if conditon is true (flexible assert)
#  define CUT_CONDITION( val)                                                \
    if( CUTFalse == cutCheckCondition( val, __FILE__, __LINE__)) {           \
        exit(EXIT_FAILURE);                                                  \
    }

#else  // not DEBUG

#define CUT_BANK_CHECKER( array, index)  array[index]

    // void macros for performance reasons
#  define CUT_CHECK_ERROR(errorMessage)
#  define CUT_CHECK_ERROR_GL()
#  define CUT_CONDITION( val) 
#  define CU_SAFE_CALL( call) call
#  define CUDA_SAFE_CALL( call) call
#  define CUT_SAFE_CALL( call) call
#  define CUFFT_SAFE_CALL( call) call

#endif

#if __DEVICE_EMULATION__

#  define CUT_CHECK_DEVICE()

#else

#  define CUT_CHECK_DEVICE() do {                                            \
    int deviceCount;                                                         \
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));                        \
    if (deviceCount == 0) {                                                  \
        fprintf(stderr, "There is no device.\n");                            \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    int dev;                                                                 \
    for (dev = 0; dev < deviceCount; ++dev) {                                \
        cudaDeviceProp deviceProp;                                           \
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));           \
        if (strncmp(deviceProp.name, "Device Emulation", 16))                \
            break;                                                           \
    }                                                                        \
    if (dev == deviceCount) {                                                \
        fprintf(stderr, "There is no device supporting CUDA.\n");            \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    else                                                                     \
        CUDA_SAFE_CALL(cudaSetDevice(dev));                                  \
} while (0)

#endif

#define CUT_EXIT(argc, argv)                                                 \
    if (!cutCheckCmdLineFlag(argc, (const char**)argv, "noprompt")) {        \
        printf("\nPress ENTER to exit...\n");                                \
        getchar();                                                           \
    }                                                                        \
    exit(EXIT_SUCCESS);


#ifdef __cplusplus
}
#endif  // #ifdef _DEBUG (else branch)

#endif  // #ifndef _CUTIL_H_
