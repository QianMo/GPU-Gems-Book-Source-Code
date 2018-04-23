#include "slicing.h"

#include <vector>
#include <algorithm>

#include <iostream>

#include <glh/glh_linear.h>
#include <glh/glh_extensions.h>

using namespace glh;

// lookup tables
unsigned short edge_table[];
char           poly_table[][7];
unsigned char  edge_corners[][2];
unsigned int   corner_idx[];
float          corner_pos_float[];
vec3f         *corner_pos = reinterpret_cast<vec3f*>(corner_pos_float);

typedef std::vector<vec3f> PosVec;
typedef std::vector<int>   IdxVec;

PosVec vert_vec;
IdxVec first_vec, count_vec;

float frc(const float& _f)
{
	return _f - floor(_f);
}

void draw_slices(const float* _dir, float _n)
{
	float vals[8];
	vec3f ePos[12];
	vec3f dir(_dir);

	for(int i = 0; i < 8; ++i)
		vals[i] = dir.dot(corner_pos[i]);

	float min_val = *std::min_element(vals, vals + 8);
	float max_val = *std::max_element(vals, vals + 8);
	float h = 1.0f / _n;

	first_vec.resize(0);
	count_vec.resize(0);
	vert_vec.resize(0);

	for(float iso = ceilf(min_val*_n)*h; iso<max_val; iso+=h)
	{

		// generate the code to look up edgeTable and triaTable
		int c_code = 0;
		for(int i = 0; i < 8; ++i) 
			c_code |= (vals[i] > iso) << i;

		// calculate the intersection points along the edges
		unsigned short e_code = edge_table[c_code], mask = 1;
		for(int i = 0; i < 12; ++i, mask<<=1)
		{
			if(e_code & mask)
			{
				unsigned char first = edge_corners[i][0], second = edge_corners[i][1];
				float val = (iso - vals[first]) / (vals[second] - vals[first]);
				ePos[i] = corner_pos[first] * (1.0f - val) + corner_pos[second] * val;
			}
		}

		// add the polyhedrons
		int n = 0;
		first_vec.push_back((int)vert_vec.size());
		char* polyCode = poly_table[c_code];
		for(; polyCode[n] >= 0; ++n)
			vert_vec.push_back(ePos[polyCode[n]]);
		assert(n < 7);
		count_vec.push_back(n);
	}

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, vert_vec.front().v);
	glMultiDrawArrays(GL_POLYGON, &first_vec.front(), &count_vec.front(), (GLsizei)count_vec.size());
	glDisableClientState(GL_VERTEX_ARRAY);

}

void draw_cube()
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, corner_pos_float);
	glDrawElements(GL_QUADS, 24, GL_UNSIGNED_INT, corner_idx);
	glDisableClientState(GL_VERTEX_ARRAY);
}



// marching cubes edge table
unsigned short edge_table[256] = {
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
}; 	

// marching cubes triangle table
char poly_table[256][7] = {
	{-1},
	{0, 8, 3, -1},
	{0, 1, 9, -1},
	{1, 9, 8, 3, -1},
	{1, 2, 10, -1},
	{-1},
	{9, 0, 2, 10, -1},
	{2, 10, 9, 8, 3, -1},
	{3, 11, 2, -1},
	{0, 8, 11, 2, -1},
	{-1},
	{1, 9, 8, 11, 2, -1},
	{3, 11, 10, 1, -1},
	{0, 8, 11, 10, 1, -1},
	{3, 11, 10, 9, 0, -1},
	{9, 8, 11, 10, -1},
	{4, 7, 8, -1},
	{4, 7, 3, 0, -1},
	{-1},
	{4, 7, 3, 1, 9, -1},
	{-1},
	{-1},
	{-1},
	{2, 10, 9, 4, 7, 3, -1},
	{-1},
	{11, 2, 0, 4, 7, -1},
	{-1},
	{4, 7, 11, 2, 1, 9, -1},
	{-1},
	{1, 0, 4, 7, 11, 10, -1},
	{-1},
	{4, 7, 11, 10, 9, -1},
	{9, 5, 4, -1},
	{-1},
	{0, 1, 5, 4, -1},
	{8, 3, 1, 5, 4, -1},
	{-1},
	{-1},
	{5, 4, 0, 2, 10, -1},
	{2, 10, 5, 4, 8, 3, -1},
	{-1},
	{-1},
	{-1},
	{2, 1, 5, 4, 8, 11, -1},
	{-1},
	{-1},
	{5, 4, 0, 3, 11, 10, -1},
	{5, 4, 8, 11, 10, -1},
	{9, 5, 7, 8, -1},
	{9, 5, 7, 3, 0, -1},
	{0, 1, 5, 7, 8, -1},
	{1, 5, 7, 3, -1},
	{-1},
	{-1},
	{8, 0, 2, 10, 5, 7, -1},
	{2, 10, 5, 7, 3, -1},
	{-1},
	{9, 5, 7, 11, 2, 0, -1},
	{-1},
	{11, 2, 1, 5, 7, -1},
	{-1},
	{-1},
	{-1},
	{11, 10, 5, 7, -1},
	{10, 6, 5, -1},
	{-1},
	{-1},
	{-1},
	{1, 2, 6, 5, -1},
	{-1},
	{9, 0, 2, 6, 5, -1},
	{5, 9, 8, 3, 2, 6, -1},
	{-1},
	{-1},
	{-1},
	{-1},
	{6, 5, 1, 3, 11, -1},
	{0, 8, 11, 6, 5, 1, -1},
	{3, 11, 6, 5, 9, 0, -1},
	{6, 5, 9, 8, 11, -1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{10, 6, 4, 9, -1},
	{-1},
	{10, 6, 4, 0, 1, -1},
	{8, 3, 1, 10, 6, 4, -1},
	{1, 2, 6, 4, 9, -1},
	{-1},
	{0, 2, 6, 4, -1},
	{8, 3, 2, 6, 4, -1},
	{-1},
	{-1},
	{-1},
	{-1},
	{9, 1, 3, 11, 6, 4, -1},
	{-1},
	{3, 11, 6, 4, 0, -1},
	{6, 4, 8, 11, -1},
	{7, 8, 9, 10, 6, -1},
	{0, 9, 10, 6, 7, 3, -1},
	{10, 6, 7, 8, 0, 1, -1},
	{10, 6, 7, 3, 1, -1},
	{1, 2, 6, 7, 8, 9, -1},
	{-1},
	{7, 8, 0, 2, 6, -1},
	{7, 3, 2, 6, -1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{7, 11, 6, -1},
	{7, 6, 11, -1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{7, 6, 2, 3, -1},
	{7, 6, 2, 0, 8, -1},
	{-1},
	{1, 9, 8, 7, 6, 2, -1},
	{10, 1, 3, 7, 6, -1},
	{10, 1, 0, 8, 7, 6, -1},
	{0, 3, 7, 6, 10, 9, -1},
	{7, 6, 10, 9, 8, -1},
	{6, 11, 8, 4, -1},
	{3, 0, 4, 6, 11, -1},
	{-1},
	{9, 4, 6, 11, 3, 1, -1},
	{-1},
	{-1},
	{-1},
	{-1},
	{8, 4, 6, 2, 3, -1},
	{0, 4, 6, 2, -1},
	{-1},
	{1, 9, 4, 6, 2, -1},
	{8, 4, 6, 10, 1, 3, -1},
	{10, 1, 0, 4, 6, -1},
	{-1},
	{10, 9, 4, 6, -1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{-1},
	{6, 11, 8, 9, 5, -1},
	{3, 0, 9, 5, 6, 11, -1},
	{0, 1, 5, 6, 11, 8, -1},
	{6, 11, 3, 1, 5, -1},
	{-1},
	{-1},
	{-1},
	{-1},
	{5, 6, 2, 3, 8, 9, -1},
	{9, 5, 6, 2, 0, -1},
	{-1},
	{1, 5, 6, 2, -1},
	{-1},
	{-1},
	{-1},
	{10, 5, 6, -1},
	{11, 7, 5, 10, -1},
	{-1},
	{-1},
	{-1},
	{11, 7, 5, 1, 2, -1},
	{-1},
	{9, 0, 2, 11, 7, 5, -1},
	{-1},
	{2, 3, 7, 5, 10, -1},
	{8, 7, 5, 10, 2, 0, -1},
	{-1},
	{-1},
	{1, 3, 7, 5, -1},
	{0, 8, 7, 5, 1, -1},
	{9, 0, 3, 7, 5, -1},
	{9, 8, 7, 5, -1},
	{5, 10, 11, 8, 4, -1},
	{5, 10, 11, 3, 0, 4, -1},
	{-1},
	{-1},
	{2, 11, 8, 4, 5, 1, -1},
	{-1},
	{-1},
	{-1},
	{2, 3, 8, 4, 5, 10, -1},
	{5, 10, 2, 0, 4, -1},
	{-1},
	{-1},
	{8, 4, 5, 1, 3, -1},
	{0, 4, 5, 1, -1},
	{-1},
	{9, 4, 5, -1},
	{4, 9, 10, 11, 7, -1},
	{-1},
	{1, 10, 11, 7, 4, 0, -1},
	{-1},
	{4, 9, 1, 2, 11, 7, -1},
	{-1},
	{11, 7, 4, 0, 2, -1},
	{-1},
	{2, 3, 7, 4, 9, 10, -1},
	{-1},
	{-1},
	{-1},
	{4, 9, 1, 3, 7, -1},
	{-1},
	{4, 0, 3, 7, -1},
	{4, 8, 7, -1},
	{9, 10, 11, 8, -1},
	{3, 0, 9, 10, 11, -1},
	{0, 1, 10, 11, 8, -1},
	{3, 1, 10, 11, -1},
	{1, 2, 11, 8, 9, -1},
	{-1},
	{0, 2, 11, 8, -1},
	{3, 2, 11, -1},
	{2, 3, 8, 9, 10, -1},
	{9, 10, 2, 0, -1},
	{-1},
	{1, 10, 2, -1},
	{1, 3, 8, 9, -1},
	{0, 9, 1, -1},
	{0, 3, 8, -1},
	{-1}
}; 	

unsigned char edge_corners[12][2] =
{
	{0,1},
	{1,2},
	{2,3},
	{3,0},
	{4,5},
	{5,6},
	{6,7},
	{7,4},
	{0,4},
	{1,5},
	{2,6},
	{3,7}
};

unsigned int corner_idx[24] = 
{
	0, 1, 2, 3, 
	4, 5, 6, 7,
	0, 1, 5, 4,
	1, 2, 6, 5,
	2, 3, 7, 6,
	3, 0, 4, 7
};

float corner_pos_float[24] =
{
	0, 0, 0,
	1, 0, 0,
	1, 1, 0,
	0, 1, 0,
	0, 0, 1,
	1, 0, 1,
	1, 1, 1,
	0, 1, 1,
};

