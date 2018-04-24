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
#ifndef _EXCEPTION_H_
#define _EXCEPTION_H_

// includes, system
#include <exception>
#include <stdexcept>
#include <iostream>

//! Exception wrapper.
//! @param Std_Exception Exception out of namespace std for easy typing.
template<class Std_Exception>
class Exception : public Std_Exception 
{
public:

    //! @brief Static construction interface
    //! @return Alwayss throws ( Located_Exception<Exception>)
    //! @param file file in which the Exception occurs
    //! @param line line in which the Exception occurs
    //! @param detailed details on the code fragment causing the Exception
    static void throw_it( const char* file, 
                          const int line,
                          const char* detailed = "-" );  

    //! Static construction interface
    //! @return Alwayss throws ( Located_Exception<Exception>)
    //! @param file file in which the Exception occurs
    //! @param line line in which the Exception occurs
    //! @param detailed details on the code fragment causing the Exception
    static void throw_it( const char* file, 
                          const int line,      
                          const std::string& detailed);  

    //! Destructor
    virtual ~Exception() throw(); 

private:

    //! Constructor, default (private)
    Exception(); 

    //! Constructor, standard
    //! @param str string returned by what()
    Exception( const std::string& str); 

};

////////////////////////////////////////////////////////////////////////////////
//! Exception handler function for arbitrary exceptions
//! @param ex exception to handle
////////////////////////////////////////////////////////////////////////////////
template<class Exception_Typ>
inline void
handleException( const Exception_Typ& ex) 
{
    std::cerr << ex.what() << std::endl;

    exit( EXIT_FAILURE);
}

//! Convenience macros

//! Exception caused by dynamic program behavior, e.g. file does not exist
#define RUNTIME_EXCEPTION( msg) \
    Exception<std::runtime_error>::throw_it( __FILE__, __LINE__, msg)

//! Logic exception in program, e.g. an assert failed
#define LOGIC_EXCEPTION( msg) \
    Exception<std::logic_error>::throw_it( __FILE__, __LINE__, msg)

//! Out of range exception
#define RANGE_EXCEPTION( msg) \
    Exception<std::range_error>::throw_it( __FILE__, __LINE__, msg)

////////////////////////////////////////////////////////////////////////////////
//! Implementation

// includes, system
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
//! Static construction interface.
//! @param  Exception causing code fragment (file and line) and detailed infos.
////////////////////////////////////////////////////////////////////////////////
/*static*/ template<class Std_Exception>
void
Exception<Std_Exception>::
throw_it( const char* file, const int line, const char* detailed) 
{
    std::stringstream s;

    // Quiet heavy-weight but exceptions are not for 
    // performance / release versions
    s << "Exception in file '" << file << "' in line " << line << "\n"
      << "Detailed description: " << detailed << "\n";

    throw Exception( s.str());
}

////////////////////////////////////////////////////////////////////////////////
//! Static construction interface.
//! @param  Exception causing code fragment (file and line) and detailed infos.
////////////////////////////////////////////////////////////////////////////////
/*static*/ template<class Std_Exception>
void
Exception<Std_Exception>::
throw_it( const char* file, const int line, const std::string& msg) 
{
    throw_it( file, line, msg.c_str());
}

////////////////////////////////////////////////////////////////////////////////
//! Constructor, default (private).
////////////////////////////////////////////////////////////////////////////////
template<class Std_Exception>
Exception<Std_Exception>::Exception() :
 Exception("Unknown Exception.\n")
{ }

////////////////////////////////////////////////////////////////////////////////
//! Constructor, standard (private).
//! String returned by what().
////////////////////////////////////////////////////////////////////////////////
template<class Std_Exception>
Exception<Std_Exception>::Exception( const std::string& s) :
 Std_Exception( s)
{ }   

////////////////////////////////////////////////////////////////////////////////
//! Destructor
////////////////////////////////////////////////////////////////////////////////
template<class Std_Exception>
Exception<Std_Exception>::~Exception() throw() { }

// functions, exported

#endif // #ifndef _EXCEPTION_H_

