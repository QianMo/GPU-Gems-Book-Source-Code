//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
//---------------------------------------------------------------------------
//#include <iostream>
//#include <stdlib.h>
#include <time.h>
#pragma hdrstop
#include "Mathematic.h"
//#pragma package(smart_init)
//---------------------------------------------------------------------------
namespace Math {

void randomize() { 
	srand((unsigned)time(0)); 
}

//namespace
}

/*int _matherr(struct _exception *e) {
	std::cerr << "Matherror:" << e->name << '(' << e->arg1 << ','
				<< e->arg2 << ")=";
	return 0;
}*/


