//
// wavefront.h
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

#ifndef _WAVEFRONT_H
#define _WAVEFRONT_H

#include "vectors.h"
#include <string>
#include <vector>

/// Wavefront OBJ parser that also computes the binormal and tangent for each vertex 
/// depending upon the UV coordinate system.
class Wavefront {
	public:
		Wavefront(const char* filename);
		~Wavefront();
		
		std::vector<float> positions;
		std::vector<float> normals;
		std::vector<float> uvs;
		std::vector<int>   index;
		
		std::vector<float> binormals;
		std::vector<float> tangents;

		unsigned int indexCount;
		unsigned int vertexCount;

		inline Vector3D getMin() { return minVector; }
		inline Vector3D getMax() { return maxVector; }

	private:
		void loadMesh();
		void mergeIndicies(std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<int>&, std::vector<int>&, std::vector<int>&);
		void constructCoordinateFrame();

		std::string filename;	

		Vector3D minVector, maxVector;
};


#endif
