//
// alloc2D.h
// Last Updated:		05.01.07
// 
// Mark Colbert & Jaroslav Krivanek
// colbert@cs.ucf.edu
//
// Copyright (c) 2007.
//
// The following code is freely distributed "as is" and comes with 
// no guarantees or required support by the authors.  Any use of 
// the code for commercial purposes requires explicit written consent 
// by the authors.
//

#ifndef _ALLOC_2D_H
#define _ALLOC_2D_H

#ifndef NULL
#define NULL 0
#endif

/// simple 2D memory allocation function for C++
/// > Ensures the data is aligned as one array and indexed by 
///   another array.  The is a trade-off between slower allocation
///   with faster random-access speed.
template<class T> inline T** alloc2D(int w, int h) {
	T *temp = new T[w*h];
	T **result = new T*[h];

	int pos=0;
	for (int i=0; i < h; i++) {
		result[i] = &temp[pos];
		pos += w;
	}

	return result;
}

/// corresponding free function
template<class T> inline void free2D(T** data) {
	delete[] data[0];
	delete[] data;
}

#endif
