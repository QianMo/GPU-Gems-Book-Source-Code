/*
 * Exception.cpp
 *
 * Li-Yi Wei
 * 3/10/2003
 *
 */

#include "Exception.hpp"

Exception::Exception(const string & message) : _message(message)
{
    // nothing to do
}

Exception::~Exception(void)
{
    // nothing to do
}

const string & Exception::Message(void) const
{
    return _message;
}
