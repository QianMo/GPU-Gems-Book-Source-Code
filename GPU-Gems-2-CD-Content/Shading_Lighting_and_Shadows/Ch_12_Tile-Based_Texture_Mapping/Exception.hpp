/*
 * Exception.hpp
 *
 * the base class for all exceptions
 *
 * Li-Yi Wei
 * 3/10/2003
 *
 */

#ifndef _EXCEPTION_HPP
#define _EXCEPTION_HPP

#include <string>

using namespace std;

class Exception
{
public:
    Exception(const string & message);
    ~Exception(void);

    const string & Message(void) const;
protected:
    const string _message;
};

#endif
