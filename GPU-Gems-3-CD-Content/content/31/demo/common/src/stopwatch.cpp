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

#include <vector>

// includes, file
#include <stopwatch.h>

// includes, project
#include <exception.h>

////////////////////////////////////////////////////////////////////////////////
// static variables

//! global index for all stop watches
#ifdef _WIN32
/*static*/ std::vector< StopWatchC* > StopWatchC::swatches;
#else
template<class OSPolicy>
/*static*/ std::vector< StopWatchBase<OSPolicy>* > 
StopWatchBase<OSPolicy>::    swatches;
#endif


// namespace, unnamed
namespace 
{
    // convenience typedef
    typedef  std::vector< StopWatchC* >::size_type  swatches_size_type;  

    //////////////////////////////////////////////////////////////////////////////
    //! Translate stop watch name to index
    //////////////////////////////////////////////////////////////////////////////
    swatches_size_type
    nameToIndex( const unsigned int& name) 
    {

#ifdef _DEBUG
        const swatches_size_type pos = name - 1;
        if(     (pos >= StopWatchC::swatches.size()) 
            || ( NULL == StopWatchC::swatches[pos])) 
        {
            RUNTIME_EXCEPTION( "No StopWatch with the requested name exist.");
        }

        return pos;
#else 
        return name - 1;
#endif
    }

} // end namespace, unnamed

// Stop watch
namespace StopWatch 
{
    //////////////////////////////////////////////////////////////////////////////
    //! Create a stop watch
    //////////////////////////////////////////////////////////////////////////////
    const unsigned int 
    create() 
    {
        // create new stopwatch
        StopWatchC* swatch = new StopWatchC();
        if( NULL == swatch) 
        {
            return 0;
        }

        // store new stop watch
        StopWatchC::swatches.push_back( swatch);

        // return the handle to the new stop watch
        return (unsigned int) StopWatchC::swatches.size();
    }

    //////////////////////////////////////////////////////////////////////////////
    // Get a handle to the stop watch with the name \a name
    //////////////////////////////////////////////////////////////////////////////
    StopWatchC& 
    get( const unsigned int& name) 
    {
        return *(StopWatchC::swatches[nameToIndex( name)]);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Delete the stop watch with the name \a name
    //////////////////////////////////////////////////////////////////////////////
    void
    destroy( const unsigned int& name) 
    {
        // get index into global memory
        swatches_size_type  pos = nameToIndex( name);
        // delete stop watch
        delete StopWatchC::swatches[pos];
        // invalidate storage
        StopWatchC::swatches[pos] = NULL;
    }

} // end namespace, StopWatch
