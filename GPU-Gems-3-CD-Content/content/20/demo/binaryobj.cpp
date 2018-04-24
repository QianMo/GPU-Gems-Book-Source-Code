//
// binaryobj.cpp
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

#include "binaryobj.h"
#include "wavefront.h"

#include <stdio.h>
#include <exception>
#include <cassert>
using namespace std;

BinaryObj::BinaryObj(const char *filename) {
	FILE *f = fopen(filename,"rb");

	if (!f) throw exception("Invalid Filename");

	fread(&vertexCount, sizeof(int), 1, f);

	if (vertexCount <= 0) throw exception("Invalid File Type");

	unsigned int bufferSize = vertexCount*(4 + 4 + 4 + 4 + 4);

	// allocate the memory
	positions = (float*) calloc(bufferSize, sizeof(float));

	// pass it out
	normals		= &positions[vertexCount*4];
	uvs			= &normals[vertexCount*4];
	binormals	= &uvs[vertexCount*4];
	tangents	= &binormals[vertexCount*4];

	// read in the file
	fread(positions, sizeof(float), bufferSize, f);

	fclose(f);
}

BinaryObj::~BinaryObj() {
	// only free the first allocated memory spot
	free(positions);
}

void BinaryObj::saveAsBinaryObj(const char *filename, Wavefront *obj) {
	FILE *f = fopen(filename,"wb");

	if (!f) throw exception("Unable to create cached binary file (try putting the program in a non-read-only folder)");

	fwrite(&obj->vertexCount, sizeof(int), 1, f);

	// check to make sure everything is of the proper size
	assert(obj->positions.size()/4	== obj->vertexCount);
	assert(obj->normals.size()/4	== obj->vertexCount);
	assert(obj->uvs.size()/4		== obj->vertexCount);
	assert(obj->binormals.size()/4	== obj->vertexCount);
	assert(obj->tangents.size()/4	== obj->vertexCount);

	// write the data
	fwrite(&obj->positions[0],	sizeof(float), obj->positions.size(),	f);
	fwrite(&obj->normals[0],	sizeof(float), obj->normals.size(),		f);
	fwrite(&obj->uvs[0],		sizeof(float), obj->uvs.size(),			f);
	fwrite(&obj->binormals[0],	sizeof(float), obj->binormals.size(),	f);
	fwrite(&obj->tangents[0],	sizeof(float), obj->tangents.size(),	f);

	fclose(f);
}