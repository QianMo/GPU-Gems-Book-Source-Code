#ifndef PS2ARB_H
#define PS2ARB_H

#ifdef _WIN32
#include <ios>
#else
#include <iostream>
#endif


// Ugh.  There muse be a better way to communicate this...
extern std::istream *ps2arb_ps20code;

int convert_ps2arb (std::istream &ps20code, 
		    std::ostream &arbcode);

int ps2arb_parse(void);


#endif
