//
// binaryobj.h
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

#ifndef _BINARY_OBJ_H
#define _BINARY_OBJ_H

class Wavefront;

/// Binary object class that loads/saves meshes as raw binary files
/// > Used in caching the binormals/tangents as well as removing
///   time spent on parsing ASCII files
class BinaryObj {
	public:
		BinaryObj(const char *filename);
		~BinaryObj();

		float *positions;
		float *normals;
		float *uvs;
		float *binormals;
		float *tangents;

		int vertexCount;
		
		static void saveAsBinaryObj(const char *filename, Wavefront *obj);
};

#endif