//
// wavefront.cpp
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

#include "wavefront.h"
#include "vectors.h"

#include <iostream>
using namespace std;

#define LINE_BUFFER_SIZE 256

Wavefront::Wavefront(const char* _filename) : filename(_filename), indexCount(0) {
	loadMesh();
	constructCoordinateFrame();
}


Wavefront::~Wavefront() {
}

// GETNUM just gets the next number from a line of input in an OBJ file
#ifndef GETNUM
#define GETNUM(lineBuffer, numBuffer, lindex, nindex, tval)  \
	nindex=0;\
	while ((lineBuffer[lindex] == ' ') || lineBuffer[lindex] == '/') lindex++;\
	while ((lineBuffer[lindex] != ' ') && (lineBuffer[lindex] != '/') && \
		   (lineBuffer[lindex] != '\0') && (lineBuffer[lindex] != '\n') && (lindex != LINE_BUFFER_SIZE)) { \
		numBuffer[nindex] = lineBuffer[lindex]; \
		nindex++; \
		lindex++; \
	} \
	numBuffer[nindex] = '\0'; \
	tval = atoi(numBuffer);
#endif


// Wavefront loadMesh
// the following will load a wavefront file for a given filename
// i do not particularily suggest trying to shift through this code.
void Wavefront::loadMesh() {
	FILE* fin;
	fin = fopen(filename.c_str(), "r");
	if (!fin) throw exception(("Cannot find the file: " + filename).c_str());
	
	// temporary input buffers
	
	vector<int> posIndex;
	vector<int> normalIndex;
	vector<int> uvIndex;
	
	vector<float> filePosition;
	vector<float> fileNormal;
	vector<float> fileUV;
		
	float x,y,z,u,v;

	char lineBuffer[LINE_BUFFER_SIZE];
	char numBuffer[32];
	int lindex=0;
	int nindex=0;
	int ival, uvval, nval;
	
	bool firstPos=true;

	// parse the data in
	while (fgets(lineBuffer, LINE_BUFFER_SIZE, fin)) {
		switch (lineBuffer[0]) {
			case 'v':
				// case vertex information
				if (lineBuffer[1] == ' ') {
					// regular vertex point
					sscanf(&lineBuffer[2], "%f %f %f", &x, &y, &z);

					if (firstPos) {
						minVector.x=x; minVector.y=y; minVector.z=z;
						maxVector.x=x; maxVector.y=y; maxVector.z=z;
						firstPos=false;
					} else {
						Vector3D v(x,y,z);
						minVector = Vector3D::minv(minVector, v);
						maxVector = Vector3D::maxv(maxVector, v);
					}

					filePosition.push_back(x);
					filePosition.push_back(y);
					filePosition.push_back(z);
					filePosition.push_back(1.f);

				} else if (lineBuffer[1] == 't') {
					// texture coordinates
					sscanf(&lineBuffer[3], "%f %f", &u, &v);
					
					fileUV.push_back(u);
					fileUV.push_back(v);

				} else if (lineBuffer[1] == 'n') {
					// normal vector
					sscanf(&lineBuffer[2], "%f %f %f", &x, &y, &z);

					fileNormal.push_back(x);
					fileNormal.push_back(y);
					fileNormal.push_back(z);
				}
				break;
			case 'f':
				// case face information
				lindex = 2;
				indexCount += 3;
				for (int i=0; i < 3; i++) {
					
					GETNUM(lineBuffer, numBuffer, lindex, nindex, ival)
					
					// obj files go from 1..n, this just allows me to access the memory
					// directly by droping the index value to 0...(n-1)
					ival--;
					posIndex.push_back(ival);
										
					if (lineBuffer[lindex] == '/') {
						lindex++;
						GETNUM(lineBuffer, numBuffer, lindex, nindex, uvval)
						uvIndex.push_back(uvval-1);
					}
					
					if (lineBuffer[lindex] == '/') {
						lindex++;
						GETNUM(lineBuffer, numBuffer, lindex, nindex, nval)
						normalIndex.push_back(nval-1);
					}
					lindex++;
				}
				break;
		}
	}
		
	fclose(fin);
	
	// merge everything back into one index array instead of multiple index arrays
	mergeIndicies(filePosition, fileNormal, fileUV, posIndex, normalIndex, uvIndex);
		
	filePosition.clear();
	fileNormal.clear();
	fileUV.clear();

	posIndex.clear();
	normalIndex.clear();
	uvIndex.clear();

}


void Wavefront::mergeIndicies(vector<float> &filePosition, vector<float> &fileNormal,  vector<float> &fileUV,
							  vector<int>   &posIndex,	   vector<int>   &normalIndex, vector<int> &uvIndex) {

	bool useNormals = !fileNormal.empty();
	bool useUVs = !fileUV.empty();
	
	if (!useNormals && !useUVs) { 
		positions = filePosition;
		index = posIndex;
		return;
	}
		
	// assumes that vertexIndex = normalIndex = uvIndex	
	
	for (unsigned int i=0; i < posIndex.size(); i++) {
		
		positions.push_back( filePosition[posIndex[i]*4+0] );
		positions.push_back( filePosition[posIndex[i]*4+1] );
		positions.push_back( filePosition[posIndex[i]*4+2] );
		positions.push_back( filePosition[posIndex[i]*4+3] );

		if (useNormals) {
			normals.push_back( fileNormal[normalIndex[i]*3+0] );
			normals.push_back( fileNormal[normalIndex[i]*3+1] );
			normals.push_back( fileNormal[normalIndex[i]*3+2] );
			normals.push_back( 1.f							  );
		}
		
		if (useUVs) { 
			uvs.push_back( fileUV[uvIndex[i]*2+0] );
			uvs.push_back( fileUV[uvIndex[i]*2+1] );
			uvs.push_back( 0.f					  );
			uvs.push_back( 0.f					  );
		}

		index.push_back(i);
	}
	
	vertexCount = (int) positions.size()/4;
}

void Wavefront::constructCoordinateFrame() {
	//Vector3D *tan1 = new Vector3D[vertexCount * 2];
	Vector3D *tan1 = (Vector3D*) calloc(vertexCount*2, sizeof(Vector3D));
    Vector3D *tan2 = tan1 + vertexCount;
    memset(tan1, 0, vertexCount * 2 * sizeof(Vector3D));
    
	Vector3D *p			= (Vector3D*) &positions[0];
	Vector3D *ns		= (Vector3D*) &normals[0];
	Vector3D *uv		= (Vector3D*) &uvs[0];

    for (unsigned int a = 0; a < indexCount; a+=3)
    {
            unsigned int i1 = index[a+0];
            unsigned int i2 = index[a+1];
            unsigned int i3 = index[a+2];
            
            Vector3D *v1 = &p[i1];
            Vector3D *v2 = &p[i2];
			Vector3D *v3 = &p[i3];
            
			Vector3D *w1 = &uv[i1];
			Vector3D *w2 = &uv[i2];
			Vector3D *w3 = &uv[i3];
            
            float x1 = v2->x - v1->x;
            float x2 = v3->x - v1->x;
            float y1 = v2->y - v1->y;
            float y2 = v3->y - v1->y;
            float z1 = v2->z - v1->z;
            float z2 = v3->z - v1->z;
			
            float s1 = w2->x - w1->x;
            float s2 = w3->x - w1->x;
            float t1 = w2->y - w1->y;
            float t2 = w3->y - w1->y;
			
            float r = 1.0F / (s1 * t2 - s2 * t1);
            Vector3D sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
            Vector3D tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);
			
            tan1[i1] += sdir;
            tan1[i2] += sdir;
            tan1[i3] += sdir;
            
            tan2[i1] += tdir;
            tan2[i2] += tdir;
            tan2[i3] += tdir;				
    }
    
	for (unsigned int a = 0; a < vertexCount; a++)
	{
		Vector3D &n = ns[a];
		Vector3D &t = tan1[a];

		// Gram-Schmidt orthogonalize
		Vector3D tan = Vector3D::normalize(t - n * (n * t));


		// Calculate handedness
		float tanw = (n % t * tan2[a] < 0.0F) ? -1.0F : 1.0F;
		
		// calculate the binormal
		Vector3D bn = (tan % n) * tanw;

		// copy the data
		tangents.push_back(tan.x);
		tangents.push_back(tan.y);
		tangents.push_back(tan.z);
		tangents.push_back(1.f);
		
		binormals.push_back(bn.x);
		binormals.push_back(bn.y);
		binormals.push_back(bn.z);
		binormals.push_back(1.f);
	}
    

	free(tan1);
}