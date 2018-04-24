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

/* Helper to check for bank conflicts */

// includes, files
#include <bank_checker.h>

// includes, system
#include <iostream>
#include <map>

////////////////////////////////////////////////////////////////////////////////
// Static member variables

/*static*/ BankChecker  BankChecker::bank_checker;
/*static*/ std::map< unsigned int, std::map<std::string, unsigned int> >
    BankChecker::AccessLocation::thread_access;
 
/*static*/ const unsigned int BankChecker::AccessLocation::invalid = std::numeric_limits<unsigned int>::max();
/*static*/ const unsigned int BankChecker::AccessInfo::invalid = std::numeric_limits<unsigned int>::max();

////////////////////////////////////////////////////////////////////////////////
//! Constructor (private)
////////////////////////////////////////////////////////////////////////////////
BankChecker::BankChecker() :
    access_data(),
    last_ltid( warp_size - 1),
    total_num_conflicts( 0),
    is_active( false)
{ }

////////////////////////////////////////////////////////////////////////////////
//! Destructor
////////////////////////////////////////////////////////////////////////////////
BankChecker::~BankChecker() 
{ 
    if( is_active) {

        std::cout << "Total number of bank conflicts: " << total_num_conflicts 
                  << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Get a handle to the instance of this class
////////////////////////////////////////////////////////////////////////////////
/*static*/ BankChecker* const
BankChecker::getHandle() 
{ 
    return &bank_checker;
}

////////////////////////////////////////////////////////////////////////////////
//! Side effect of shared memory access
//! Side effect of shared memory access
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
void BankChecker::
access( unsigned int tidx, unsigned int tidy, unsigned int tidz,
        unsigned int bdimx, unsigned int bdimy, unsigned int bdimz,
        const char* file, const int line, const std::string& aname,
        const int index)
{ 
    is_active = true;

    // linearized thread id
    unsigned int ltid = getLtid( tidx, tidy, tidz, bdimx, bdimy, bdimz);

    // reset state if new warp
    if( 0 == (ltid & (warp_size - 1))) 
    {
        // double check to handle multiple shared mem accesses in one line
        if( last_ltid != 0) 
        {
            reset();
        }
    }

    AccessLocation loc( file, line, aname, ltid);
    AccessInfo info( ltid, tidx, tidy, tidz, index);
    access_data[loc][(index % warp_size)].push_back( info);

    if( 15 == (ltid & (warp_size - 1))) 
    {
        analyse( access_data. find( loc));
    }

    last_ltid = (ltid % warp_size);
}  

////////////////////////////////////////////////////////////////////////////////
//! Get the linearized thread id for a thread
//! @return linear thread id
//! @param tidx  thread id in x dimension of block
//! @param tidy  thread id in y dimension of block
//! @param tidz  thread id in z dimension of block
//! @param bdimx block size in x dimension
//! @param bdimy block size in y dimension
//! @param bdimz block size in z dimension
////////////////////////////////////////////////////////////////////////////////
unsigned int BankChecker::
getLtid( unsigned int tidx, unsigned int tidy, unsigned int tidz,
         unsigned int bdimx, unsigned int bdimy, unsigned int /*bdimz*/ ) 
{
    return tidx + (tidy * bdimx) + tidz * (bdimx * bdimy);
}

////////////////////////////////////////////////////////////////////////////////
//! Reset internal state after processing a warp
////////////////////////////////////////////////////////////////////////////////
void BankChecker::
reset() 
{ 
    access_data.clear();
    AccessLocation::thread_access.clear();
}

////////////////////////////////////////////////////////////////////////////////
//! Reset internal state after processing a warp
////////////////////////////////////////////////////////////////////////////////
void BankChecker::
analyse( const AccessDataCIter& iter_loc) 
{ 
    // do for all indices which have been accessed at this location
    for( IndexAInfoListCIter iter_index = iter_loc->second.begin(); 
         iter_index != iter_loc->second.end();
         ++iter_index )
    {
        // check if index has been accessed more than once
        if( (*iter_index).second.size() > 1) 
        {

            bool differences = false;
            unsigned int theindex = 0;

            // Check if all simultaneous bank accesses actually access the 
            // same index and hence are not really conflicts
            for( AccessInfoListCIter iter = (*iter_index).second.begin(); 
                 iter != (*iter_index).second.end();
                 ++iter) 
            {
                if( iter == (*iter_index).second.begin() )
                {
                    theindex = iter->getIndex();
                }
                else
                {
                    if( iter->getIndex() != theindex )
                    {
                        differences = true;
                        break;
                    }
                }
            }

            // If we see different indices and therefore real conflicts
            // then output this information
            if( differences ) 
            {
                ++total_num_conflicts;

                // print out meta information of the access
                std::cerr << "\"" << iter_loc->first.getFile() << "\""
                          << ", line " << iter_loc->first.getLine() 
                          << ", access " << iter_loc->first.getNumAccessLine()
                          << ", bank " << iter_index->first
                          << " ::\n";

                // print out meta information of all threds which have accessed the 
                // index
                for( AccessInfoListCIter iter = (*iter_index).second.begin(); 
                     iter != (*iter_index).second.end();
                     ++iter) 
                {
                    std::cerr << iter->getInfo() << '\n';
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// helper classes

////////////////////////////////////////////////////////////////////////////////
//! Constructor, default
////////////////////////////////////////////////////////////////////////////////
BankChecker::AccessLocation::AccessLocation() :
    file(),
    line( invalid),
    num_access_line( invalid),
    array_name()
{ 
    std::cerr<< "Warning: Default constructor of AcccessLocation()." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
//! Destructor
////////////////////////////////////////////////////////////////////////////////
BankChecker::AccessLocation::~AccessLocation() { }

////////////////////////////////////////////////////////////////////////////////
//! Constructor, default
////////////////////////////////////////////////////////////////////////////////
BankChecker::AccessInfo::
AccessInfo() :
    ltid( invalid),
    tidx( invalid),
    tidy( invalid),
    tidz( invalid)
{ 
    std::cerr << "Warning: Default constructor of AccessInfo()." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
//! Destructor
////////////////////////////////////////////////////////////////////////////////
BankChecker::AccessInfo::~AccessInfo() { }

////////////////////////////////////////////////////////////////////////////////
// Provide infos as string
////////////////////////////////////////////////////////////////////////////////
const std::string
BankChecker::AccessInfo::
getInfo() const 
{
    std::ostringstream oss;
    oss << "threadIdx.x = " << tidx
        << "  threadIdx.y = " << tidy
        << "  threadIdx.z = " << tidz
        << "  :: index = " << index; 

    return oss.str();
}

////////////////////////////////////////////////////////////////////////////////
// Return index 
////////////////////////////////////////////////////////////////////////////////
const unsigned int
BankChecker::AccessInfo::
getIndex() const 
{
    return index;
}
