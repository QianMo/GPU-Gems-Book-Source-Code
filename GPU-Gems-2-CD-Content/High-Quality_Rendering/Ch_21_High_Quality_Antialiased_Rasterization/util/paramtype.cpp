/////////////////////////////////////////////////////////////////////////////
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)

#include <cstdio>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <ctype.h>

#include "paramtype.h"

using namespace Gelato;


namespace Gelato {

int
ParamBaseTypeSize (int t)
{
    static int sizes[] = {
        // Absolutely MUST match the order of ParamBaseType
        0,
        0,
        sizeof(char *),
        sizeof(float), sizeof(float)/2, sizeof(double),
        3*sizeof(float), 3*sizeof(float), 3*sizeof(float),
        3*sizeof(float),
        4*sizeof(float), 16*sizeof(float),
        sizeof(char), sizeof(unsigned char), sizeof(short), sizeof(unsigned short),
        sizeof(int), sizeof(unsigned int)
    };
    assert (sizeof(sizes)/sizeof(sizes[0]) == PT_LAST);
    if (t < 0 || t >= PT_LAST)
        return 0;
    
    return sizes[t];
}



const char *
ParamBaseTypeNameString (int t)
{
    static const char *names[] = {
        // Absolutely MUST match the order of ParamBaseType
        "<unknown>",
        "void",
        "string",
        "float", "half", "double",
        "point", "vector", "normal",
        "color",
        "hpoint", "matrix",
        "int8", "uint8", "int16", "uint16", "int", "uint"
    };
    assert (sizeof(names)/sizeof(names[0]) == PT_LAST);

    if (t < 0 || t >= PT_LAST)
        return NULL;
    
    return names[t];
}



int
ParamBaseTypeNFloats (int t)
{
    static short nfloats[] = {
        // Absolutely MUST match the order of ParamBaseType
        0,
        0,
        0,
        1, 0, 0,
        3, 3, 3,
        3,
        4, 16,
        0, 0, 0, 0, 0, 0
    };
    assert (sizeof(nfloats)/sizeof(nfloats[0]) == PT_LAST);
    assert (t >= 0 && t < PT_LAST);
    // This one is for operator==.  Didn't want assert in the include file.
    assert (sizeof(ParamType) == sizeof(int));
    return nfloats[t];
}




// Helper function -- does the string in prefix head the string in start
// (with the next character in start being a space or separator.  In
// other words prefixmatch("float myvar", "float") succeeds, but
// prefixmatch("floater", "float") should fail.
// Upon failure, return 0.  Upon success, return a nonzero value that
// is the length, in characters, of the prefix.
static int
prefixmatch (const char *start, const char *prefix)
{
    const char *s = start;
    for ( ;  *prefix;  ++prefix, ++s)
        if (*s != *prefix)
            return 0;  // No match (including end of string), fail
    // Got to end of prefix -- a potential match, but only if there's a
    // separator next.
    if (*s == ' '  ||  *s == '['  ||  *s == 0)
        return int (s - start);
    else return 0;
}



// Parse a string and discern its ParamType.  The grammar is:
//    [ interpname ' ' ]  typename [ '[' int ']' ] ' ' variablename
int
ParamType::fromstring (const char *namestart, char *shortname)
{
    const char *name = namestart;
    int interp = INTERP_CONSTANT;
    int len = 0;
    if ((len = prefixmatch (name, "constant")))
        interp = INTERP_CONSTANT;
    else if ((len = prefixmatch (name, "perpiece")))
        interp = INTERP_PERPIECE;
    else if ((len = prefixmatch (name, "linear")))
        interp = INTERP_LINEAR;
    else if ((len = prefixmatch (name, "vertex")))
        interp = INTERP_VERTEX;
    name += len;
    while (*name == ' ')
        ++name;
    for (int t = 0;  t < PT_LAST;  ++t) {
        if ((len = prefixmatch (name, ParamBaseTypeNameString(t)))) {
            name += len;
            while (*name == ' ')
                ++name;
            int arraylen = 0;
            if (*name == '[') {
                ++name;
                while (*name >= '0' && *name <= '9') {
                    arraylen = 10*arraylen + (*name - '0');
                    ++name;
                }
                if (*name == ']')
                    ++name;
                else {
                    return 0;
                }
            }
            *this = ParamType ((ParamBaseType)t, arraylen, (ParamInterp)interp);
            // eliminate any leading whitespace
            while (isspace (*name) && *name != '\0')
                name++;
            
            if (shortname)
                strcpy (shortname, name);
            return int (name - namestart);
        }
    }
    if (shortname)
        shortname[0] = 0;
    return 0;   // No type name could be discerned
}



bool
ParamType::tostring (char *typestring, int maxlen, bool showinterp) const
{
    typestring[0] = '\0';
    const char *interp_str = "";
    if (showinterp) {
        switch (interp) {
        case INTERP_CONSTANT: break; // Don't prepend anything
        case INTERP_PERPIECE: interp_str = "perpiece "; break;
        case INTERP_LINEAR: interp_str = "linear "; break;
        case INTERP_VERTEX: interp_str = "vertex "; break;
        }
    }
    const char *name = ParamBaseTypeNameString (basetype);
    if (name == NULL)
        return false;
    int len = snprintf (typestring, maxlen, "%s%s", interp_str, name);
    if (len < 0)
        return false;
    typestring += len;
    maxlen -= len;
    if (isarray) {
        len = snprintf (typestring, maxlen, "[%d]", arraylen);
        typestring += len;
        maxlen -= len;
    }
    return true;
}

}; /* end namespace Gelato */
