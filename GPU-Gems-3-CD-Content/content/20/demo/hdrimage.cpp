//
// hdrimage.cpp
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

#define _USE_MATH_DEFINES

#include "hdrimage.h"
#include "rgbe.h"
#include "color.h"
#include "vectors.h"
#include <string>
#include <cmath>
#include <exception>

using namespace std;

inline static float sinc(float x) {
  if (fabs(x) < 1.0e-4) return 1.0 ;
  else return(sin(x)/x) ;
}

HDRImage::HDRImage(const char *name) : selfAlloc(false) {
	ReadImage(name);
}

HDRImage::~HDRImage() {
	if (selfAlloc) delete[] data;
}

void HDRImage::ReadImage(const char *name) {
	FILE *f;

	#if _MSC_VER >= 1400
	fopen_s(&f, name, "rb");
	#else
	f = fopen(name, "rb");
	#endif
	if (!f) throw exception((string("Unable to open file ") + string(name)).c_str());
	
	if (RGBE_ReadHeader(f, &width, &height, NULL) != RGBE_RETURN_SUCCESS) {
		throw exception((string("Unable to read header for ") + string(name)).c_str());
	}

	data = new float[width*height*3];
	components = 3;
	selfAlloc = true;
	if (data == NULL) {
		fclose(f);
		throw exception((string("Unable to allocated memory for ") + string(name)).c_str());
	}

	// read the data
	if (RGBE_ReadPixels_RLE(f, data, width, height) != RGBE_RETURN_SUCCESS) {
		fclose(f);
		throw exception((string("Unable to read datafor ") + string(name)).c_str());
	}
	
	fclose(f);
}

void HDRImage::WriteImage(const char *name, float *data, int w, int h) {
	FILE *fp;
	
	#if _MSC_VER >= 1400
	fopen_s(&fp, name, "wb");
	#else
	fp = fopen(name, "wb");
	#endif

	rgbe_header_info info;
	info.exposure = 1.f;
	info.gamma = 2.2f;
	
	#if _MSC_VER >= 1400
	strcpy_s(info.programtype, "RADIANCE");
	#else
	strcpy(info.programtype, "RADIANCE");
	#endif
	
	info.valid = RGBE_VALID_PROGRAMTYPE | RGBE_VALID_GAMMA | RGBE_VALID_EXPOSURE;
	
	if (RGBE_WriteHeader(fp, w, h, &info) != RGBE_RETURN_SUCCESS) {
		throw exception((string("Error In Writing Header for ") + string(name)).c_str());
	}

	if (RGBE_WritePixels_RLE(fp, data, w, h) != RGBE_RETURN_SUCCESS) {
		throw exception((string("Error In Writing Pixels for ") + string(name)).c_str());
	}

	fclose(fp);
}

Color3 HDRProbe::ComputeAverageColor() {
	Color3 c;
	int width = image->width;
	float *data = image->data;

	for (int i = 0 ; i < width ; i++) {
		for (int j = 0 ; j < width ; j++) {

			/* We now find the cartesian components for the point (i,j) */
			float u,v,r,theta,phi,domega ;

			v = (width/2.f - i)/(width/2.f);  /* v ranges from -1 to 1 */
			u = (j-width/2.f)/(width/2.f);    /* u ranges from -1 to 1 */
			r = sqrt(u*u+v*v) ;               /* The "radius" */
			if (r > 1.f) continue ;           /* Consider only circle with r<1 */

			theta = (float) M_PI*r ;                    /* theta parameter of (i,j) */
			phi = atan2f(v,u) ;                /* phi parameter */

			domega = (float) ((2.f*M_PI/width)*(2.f*M_PI/width)*sinc(theta));

			c += Color3(data[(width*i+j)*3+0], data[(width*i+j)*3+1],data[(width*i+j)*3+2]) * domega;
		}
	}
	return c;

}

// Spherical harmonics convolution code courtesy of Ravi Ramamoorthi
void HDRProbe::ConstructSHMatrices(float *r, float *g, float *b) {
	float coeffs[9][3];
	memset(coeffs, 0, sizeof(coeffs));
	
	float *data = (float*) image->data;
	int i,j ;
	int width = image->width;
	Vector3D vec;

	avgLum = 0.f;
	for (i = 0 ; i < width ; i++) {
		for (j = 0 ; j < width ; j++) {

			/* We now find the cartesian components for the point (i,j) */
			float u,v,r,theta,phi,domega ;

			v = (width/2.f - i)/(width/2.f);  /* v ranges from -1 to 1 */
			u = (j-width/2.f)/(width/2.f);    /* u ranges from -1 to 1 */
			r = sqrt(u*u+v*v) ;               /* The "radius" */
			if (r > 1.f) continue ;           /* Consider only circle with r<1 */

			theta = (float) M_PI*r ;                    /* theta parameter of (i,j) */
			phi = atan2f(v,u) ;                /* phi parameter */

			vec.x = sin(theta)*cos(phi) ;         /* Cartesian components */
			vec.y = sin(theta)*sin(phi) ;
			vec.z = cos(theta) ;

			/* Computation of the solid angle.  This follows from some
			elementary calculus converting sin(theta) d theta d phi into
			coordinates in terms of r.  This calculation should be redone 
			if the form of the input changes */

			domega = (float) ((2.f*M_PI/width)*(2.f*M_PI/width)*sinc(theta));

			UpdateCoeffs(coeffs, &data[(width*i+j)*3],domega,vec) ; /* Update Integration */

			avgLum += (0.212671f*data[(width*i+j)*3+0] + 
					   0.715160f*data[(width*i+j)*3+1] + 
					   0.072169f*data[(width*i+j)*3+2]) * domega;
		}
	}

	ConvertCoeffsToMatrix(coeffs, r, g, b);
}

void HDRProbe::UpdateCoeffs(float coeffs[][3], float hdr[3], float domega, const Vector3D &v) {
	int col ;
	for (col = 0 ; col < 3 ; col++) {
		float c ; /* A different constant for each coefficient */

		/* L_{00}.  Note that Y_{00} = 0.282095 */
		c = 0.282095f;
		coeffs[0][col] += hdr[col]*c*domega ;

		/* L_{1m}. -1 <= m <= 1.  The linear terms */
		c = 0.488603f;
		coeffs[1][col] += hdr[col]*(c*v.y)*domega ;   /* Y_{1-1} = 0.488603 y  */
		coeffs[2][col] += hdr[col]*(c*v.z)*domega ;   /* Y_{10}  = 0.488603 z  */
		coeffs[3][col] += hdr[col]*(c*v.x)*domega ;   /* Y_{11}  = 0.488603 x  */

		/* The Quadratic terms, L_{2m} -2 <= m <= 2 */

		/* First, L_{2-2}, L_{2-1}, L_{21} corresponding to xy,yz,xz */
		c = 1.092548f;
		coeffs[4][col] += hdr[col]*(c*v.x*v.y)*domega ; /* Y_{2-2} = 1.092548 xy */ 
		coeffs[5][col] += hdr[col]*(c*v.y*v.z)*domega ; /* Y_{2-1} = 1.092548 yz */ 
		coeffs[7][col] += hdr[col]*(c*v.x*v.z)*domega ; /* Y_{21}  = 1.092548 xz */ 

		/* L_{20}.  Note that Y_{20} = 0.315392 (3z^2 - 1) */
		c = 0.315392f;
		coeffs[6][col] += hdr[col]*(c*(3*v.z*v.z-1))*domega ; 

		/* L_{22}.  Note that Y_{22} = 0.546274 (x^2 - y^2) */
		c = 0.546274f;
		coeffs[8][col] += hdr[col]*(c*(v.x*v.x-v.y*v.y))*domega ;
	}
}

void HDRProbe::ConvertCoeffsToMatrix(float coeffs[][3], float *r, float *g, float *b) {
	const float c1 =0.429043f;
	const float c2 =0.511664f;
	const float c3 =0.743125f;
	const float c4 =0.886227f;
	const float c5 =0.2477083f;
	
	r[0]=c1*coeffs[8][0];  r[1]=c1*coeffs[4][0];  r[2]=c1*coeffs[7][0];  r[3]=c2*coeffs[3][0];
	r[4]=c1*coeffs[4][0];  r[5]=-c1*coeffs[8][0]; r[6]=c1*coeffs[5][0];  r[7]=c2*coeffs[1][0];
	r[8]=c1*coeffs[7][0];  r[9]=c1*coeffs[5][0];  r[10]=c3*coeffs[6][0]; r[11]=c2*coeffs[2][0];
	r[12]=c2*coeffs[3][0]; r[13]=c2*coeffs[1][0]; r[14]=c2*coeffs[2][0]; r[15]=c4*coeffs[0][0]-c5*coeffs[6][0];
			
	g[0]=c1*coeffs[8][1];  g[1]=c1*coeffs[4][1];  g[2]=c1*coeffs[7][1];  g[3]=c2*coeffs[3][1];
	g[4]=c1*coeffs[4][1];  g[5]=-c1*coeffs[8][1]; g[6]=c1*coeffs[5][1];  g[7]=c2*coeffs[1][1];
	g[8]=c1*coeffs[7][1];  g[9]=c1*coeffs[5][1];  g[10]=c3*coeffs[6][1]; g[11]=c2*coeffs[2][1];
	g[12]=c2*coeffs[3][1]; g[13]=c2*coeffs[1][1]; g[14]=c2*coeffs[2][1]; g[15]=c4*coeffs[0][1]-c5*coeffs[6][1];
	
	b[0]=c1*coeffs[8][2];  b[1]=c1*coeffs[4][2];  b[2]=c1*coeffs[7][2];  b[3]=c2*coeffs[3][2];
	b[4]=c1*coeffs[4][2];  b[5]=-c1*coeffs[8][2]; b[6]=c1*coeffs[5][2];  b[7]=c2*coeffs[1][2];
	b[8]=c1*coeffs[7][2];  b[9]=c1*coeffs[5][2];  b[10]=c3*coeffs[6][2]; b[11]=c2*coeffs[2][2];
	b[12]=c2*coeffs[3][2]; b[13]=c2*coeffs[1][2]; b[14]=c2*coeffs[2][2]; b[15]=c4*coeffs[0][2]-c5*coeffs[6][2];
}

float HDRProbe::GetAverageLuminance() {
	//return 2.f;
	
	return avgLum/((float) M_PI);
}

inline Color3 HDRProbe::BilinearInterpolate(Color3 *buffer, float x, float y, int w, int h) {
	float t[2]	= { x-floorf(x), y-floorf(y) };
	int xpos[2] = { (int) floorf(x), std::min((int) ceilf(x),w-1) };
	int ypos[2] = { (int) floorf(y), std::min((int) ceilf(y),h-1) };

	Color3 c[2];
	
	c[0] = buffer[ypos[0]*w+xpos[0]]*(1.f-t[0]) + buffer[ypos[0]*w+xpos[1]]*t[0];
	c[1] = buffer[ypos[1]*w+xpos[0]]*(1.f-t[0]) + buffer[ypos[1]*w+xpos[1]]*t[0];

	return c[0]*(1.f-t[1]) + c[1]*t[1];
}

inline Vector2D HDRProbe::ComputeSphericalCoord(const Vector3D &d) {
	float r = 0.159154943f*acosf(d.z)/sqrtf(d.x*d.x + d.y*d.y+0.0001f);
	return Vector2D(d.x*r + 0.5f, d.y*r + 0.5f);
}

void HDRProbe::ConstructCubeMap(float *_posx, float *_negx,
								float *_posy, float *_negy, 
								float *_posz, float *_negz, 
								int w, int h)
{
	Color3 *map[6];
	map[0] = (Color3*) _posx;	map[1] = (Color3*) _negx;
	map[2] = (Color3*) _posy;	map[3] = (Color3*) _negy;
	map[4] = (Color3*) _posz;	map[5] = (Color3*) _negz;

	Color3 *src	= (Color3*) image->data;

	float sfw = (float) image->width;
	float sfh = (float) image->height;

	float xinc = (float) (2.f/((float) w));
	float yinc = (float) (2.f/((float) h));
	float px=-1.f+xinc/2.f, py=-1.f+yinc/2.f;

	int bpos=0;

	Vector2D texcoord;

	for (int i=0; i < h; i++,py+=yinc) {
		px=-1.f+xinc/2.f;
		for (int j=0; j < w; j++,px+=xinc) {
			// compute the generated angle for the position on the texture

			texcoord = ComputeSphericalCoord(Vector3D::normalize(Vector3D(1.f, py, -px)));
			map[0][bpos] = BilinearInterpolate(src, texcoord.x*sfw, texcoord.y*sfh, image->width, image->height);

			texcoord = ComputeSphericalCoord(Vector3D::normalize(Vector3D(-1.f, py, px)));
			map[1][bpos] = BilinearInterpolate(src, texcoord.x*sfw, texcoord.y*sfh, image->width, image->height);

			texcoord = ComputeSphericalCoord(Vector3D::normalize(Vector3D(px, 1.f, -py)));
			map[2][bpos] = BilinearInterpolate(src, texcoord.x*sfw, texcoord.y*sfh, image->width, image->height);

			texcoord = ComputeSphericalCoord(Vector3D::normalize(Vector3D(px, -1.f, py)));
			map[3][bpos] = BilinearInterpolate(src, texcoord.x*sfw, texcoord.y*sfh, image->width, image->height);

			texcoord = ComputeSphericalCoord(Vector3D::normalize(Vector3D(px,py,1.f)));
			map[4][bpos] = BilinearInterpolate(src, texcoord.x*sfw, texcoord.y*sfh, image->width, image->height);

			texcoord = ComputeSphericalCoord(Vector3D::normalize(Vector3D(-px,py,-1.f)));
			map[5][bpos] = BilinearInterpolate(src, texcoord.x*sfw, texcoord.y*sfh, image->width, image->height);

			bpos++;
		}
	}
}

void HDRProbe::ConstrutDualParabolicMap(float *_fBuffer, float *_bBuffer, int w, int h) {
	Color3 *fBuffer  = (Color3*) _fBuffer;
	Color3 *bBuffer  = (Color3*) _bBuffer;
	Color3 *src		= (Color3*) image->data;

	float sfw = (float) image->width;
	float sfh = (float) image->height;

	const float parabolidWidth=2.4f;

	float xinc = (float) (parabolidWidth/w);
	float yinc = (float) (parabolidWidth/h);
	float py=-parabolidWidth/2.f+yinc/2.f;
	int bpos = 0;

	for (int i=0; i < h; i++,py+=yinc) {
		float px=-parabolidWidth/2.f+xinc/2.f;
		for (int j=0; j < w; j++,px+=xinc) {
			// compute the generated angle for the position on the texture
			float norm = (px*px + py*py + 1.f);
			Vector3D dir;
			dir.x = 2.f*px/norm;
			dir.y = 2.f*py/norm;
			dir.z = (-2.f + norm)/norm;

			// compute the position in the spherical image	
			Vector2D texcoord = ComputeSphericalCoord(dir);
			fBuffer[bpos] = BilinearInterpolate(src, texcoord.x*sfw, (1.f-texcoord.y)*sfh, image->width, image->height);

			texcoord = ComputeSphericalCoord(-dir);
			bBuffer[bpos] = BilinearInterpolate(src, texcoord.x*sfw, (1.f-texcoord.y)*sfh, image->width, image->height);
			//bBuffer[bpos]=Color3(1,0,0);

			bpos++;
		}
	}
}

void HDRProbe::ConstructLatLongMap(float *fBuffer, int w, int h) {
	Color3 *buffer  = (Color3*) fBuffer;
	Color3 *src		= (Color3*) image->data;
	float sfw = (float) image->width;
	float sfh = (float) image->height;

	float yinc = (float) ((1.f/((float) h)) * M_PI);
	float xinc = (float) ((1.f/((float) w)) * M_PI * 2.f);
	float px=0.f, py=0.f;

	int bpos=0;
	for (int i=0; i < h; i++) {
		for (int j=0; j < w; j++) {
			Vector3D coord;
			coord.x = cos(px) * sin(py);
			coord.y = cos(py);
			coord.z = sin(px) * sin(py);
	
			float r = 0.159154943f*acos(coord.z)/sqrt(coord.x*coord.x + coord.y*coord.y);
			coord = Vector3D(0.5f,0.5f,0.f) + coord * r;
	
			buffer[bpos++] = BilinearInterpolate(src, coord.x*sfw, (1.f-coord.y)*sfh, image->width, image->height);

			px+=xinc;
		}
		py+=yinc; px=0.f;
	}
}

#define POSX 0
#define NEGX 1
#define POSY 2
#define NEGY 3
#define POSZ 4
#define NEGZ 5

inline Vector2D HDRProbe::ComputeAngles(const Vector3D &v) {
	Vector3D vn = Vector3D::normalize(v);
	//cout << "vn: " << vn << endl;
	float phi = atan2f(vn.y, vn.x);
	phi = (phi>0)?phi:(2.f*((float) M_PI) + phi);
	float theta = acosf(vn.z);

	//cout << "vr: (" << sinf(theta)*cosf(phi) << " " << sinf(theta)*sinf(phi) << " " << cosf(theta) << ")" << endl;
	return Vector2D(theta, phi);
}

inline Vector3D HDRProbe::ConvertTexcoordToCubeMap(int face, float px, float py) {
	switch (face) {
		case POSX: return Vector3D(1.f, py, -px);
		case NEGX: return Vector3D(-1.f, py, px);
		case POSY: return Vector3D(px, 1.f, -py);
		case NEGY: return Vector3D(px, -1.f, py);
		case POSZ: return Vector3D(px,py,1.f);
		case NEGZ: return Vector3D(-px,py,-1.f);
	}
	return 0.f;
}

void HDRProbe::ComputeFilteredArea(int face, float px, float py, float inc, Vector2D *angles) {
	// angles[0] = upper-left
	// angles[1] = upper-right
	// angles[2] = lower-left
	// angles[3] = lower-right

	Vector2D cardinal[4];

	switch (face) {
		case POSX:
			// posz 1-(-1-px)
			cardinal[0] = ComputeAngles(((px - inc) < -1.f)?
							ConvertTexcoordToCubeMap(POSZ, 1-(-1-(px-inc)), py):
							ConvertTexcoordToCubeMap(POSX, px-inc, py));
			// negz
			cardinal[1] = ComputeAngles((px + inc > 1.f)?
							ConvertTexcoordToCubeMap(NEGZ, -1+(px+inc-1), py):
							ConvertTexcoordToCubeMap(POSX, px+inc, py));

			// posy rot 1-(-1-py)
			cardinal[2] = ComputeAngles((py - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGY, 1-(-1-(py-inc)), -px):
							ConvertTexcoordToCubeMap(POSX, px, py-inc));

			// negy
			cardinal[3] = ComputeAngles((py + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSY, 1-(py+inc-1), px):
							ConvertTexcoordToCubeMap(POSX, px, py+inc));
			break;
		case NEGX:
			// negz
			cardinal[0] = ComputeAngles((px - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGZ, 1-(-1-(px-inc)), py):
							ConvertTexcoordToCubeMap(NEGX, px-inc, py));
			// posz
			cardinal[1] = ComputeAngles((px + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSZ, -1+(px+inc-1), py):
							ConvertTexcoordToCubeMap(NEGX, px+inc, py));

			cardinal[2] = ComputeAngles((py - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGY, -1+(-1-(py-inc)), px):
							ConvertTexcoordToCubeMap(NEGX, px, py-inc));

			cardinal[3] = ComputeAngles((py + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSY, -1+(py+inc-1), -px):
							ConvertTexcoordToCubeMap(NEGX, px, py+inc));			
			break;

		case POSY:
			cardinal[0] = ComputeAngles((px - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGX, -py, 1-(-1-(px-inc))):
							ConvertTexcoordToCubeMap(POSY, px-inc, py));

			cardinal[1] = ComputeAngles((px + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSX, py, 1-(px+inc-1)):
							ConvertTexcoordToCubeMap(POSY, px+inc, py));

			cardinal[2] = ComputeAngles((py - inc < -1.f)?
							ConvertTexcoordToCubeMap(POSZ, px, 1-(-1-(py-inc))):
							ConvertTexcoordToCubeMap(POSY, px, py-inc));

			cardinal[3] = ComputeAngles((py + inc > 1.f)?
							ConvertTexcoordToCubeMap(NEGZ, -px, 1-(py+inc-1)):
							ConvertTexcoordToCubeMap(POSY, px, py+inc));			
			break;

		case NEGY:
			cardinal[0] = ComputeAngles((px - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGX, py, -1+(-1-(px-inc))):
							ConvertTexcoordToCubeMap(NEGY, px-inc, py));
			
			cardinal[1] = ComputeAngles((px + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSX, -py, -1+(px+inc-1)):
							ConvertTexcoordToCubeMap(NEGY, px+inc, py));

			cardinal[2] = ComputeAngles((py - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGZ, -px, -1+(-1-(py-inc))):
							ConvertTexcoordToCubeMap(NEGY, px, py-inc));

			cardinal[3] = ComputeAngles((py + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSZ, px, -1+(py+inc-1)):
							ConvertTexcoordToCubeMap(NEGY, px, py+inc));			
			break;

		case POSZ:
			cardinal[0] = ComputeAngles((px - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGX, 1-(-1-(px-inc)), py):
							ConvertTexcoordToCubeMap(POSZ, px-inc, py));
			
			cardinal[1] = ComputeAngles((px + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSX, -1+(px+inc-1), py):
							ConvertTexcoordToCubeMap(POSZ, px+inc, py));

			cardinal[2] = ComputeAngles((py - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGY, px, 1-(-1-(py-inc))):
							ConvertTexcoordToCubeMap(POSZ, px, py-inc));

			cardinal[3] = ComputeAngles((py + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSY, px, -1+(py+inc-1)):
							ConvertTexcoordToCubeMap(POSZ, px, py+inc));
			break;

		case NEGZ:
			cardinal[0] = ComputeAngles((px - inc < -1.f)?
							ConvertTexcoordToCubeMap(POSX, 1-(-1-(px-inc)), py):
							ConvertTexcoordToCubeMap(NEGZ, px-inc, py));
			
			cardinal[1] = ComputeAngles((px + inc > 1.f)?
							ConvertTexcoordToCubeMap(NEGX, -1+(px+inc-1), py):
							ConvertTexcoordToCubeMap(NEGZ, px+inc, py));

			cardinal[2] = ComputeAngles((py - inc < -1.f)?
							ConvertTexcoordToCubeMap(NEGY, -px, -1+(-1-(py-inc))):
							ConvertTexcoordToCubeMap(NEGZ, px, py-inc));

			cardinal[3] = ComputeAngles((py + inc > 1.f)?
							ConvertTexcoordToCubeMap(POSY, -px, 1-(py+inc-1)):
							ConvertTexcoordToCubeMap(NEGZ, px, py+inc));
			break;
	}

	// convert from cardinal coordinates to boxed coordinates
	Vector2D &currAngle = ComputeAngles(ConvertTexcoordToCubeMap(face, px, py));

	if (fabs(cardinal[2].y - currAngle.y) > M_PI) {
		if (currAngle.y < cardinal[2].y) {
			cardinal[2].y -= 2.f*M_PI;
		} else {
			cardinal[2].y += 2.f*M_PI;
		}
	}

	if (fabs(cardinal[3].y - currAngle.y) > M_PI) {
		if (currAngle.y < cardinal[3].y) {
			cardinal[3].y -= 2.f*M_PI;
		} else {
			cardinal[3].y += 2.f*M_PI;
		}
	}

	angles[0] = Vector2D(cardinal[0].x, cardinal[2].y);
	angles[1] = Vector2D(cardinal[1].x, cardinal[2].y);
	angles[2] = Vector2D(cardinal[0].x, cardinal[3].y);
	angles[3] = Vector2D(cardinal[1].x, cardinal[3].y);
}

Color3 HDRProbe::FilterArea(Vector2D &currAngle, Vector2D *angles, int steps, float sfw, float sfh) {
	Vector2D interpAngles[2];

	Color3 *src	= (Color3*) image->data;

	int shalf = steps/2;
	float stddev = ((float) shalf)/3.f;
	float stddev2 = 2.f*stddev*stddev;
	float iexp, jexp;

	Color3 c=0.f;
	for (int i=0; i <= steps; i++) {
		float alpha = ((float) i)/((float) steps);
		
		interpAngles[0] = angles[0]*(1.f-alpha) + angles[2]*alpha;
		interpAngles[1] = angles[1]*(1.f-alpha) + angles[3]*alpha;

		iexp = expf(-(i-shalf)*(i-shalf)/stddev2);

		for (int j=0; j <= steps; j++) {
			float beta = ((float) j)/((float) steps);

			Vector2D pos = interpAngles[0]*(1.f-beta) + interpAngles[1]*beta;

			float costheta = cosf(pos.x);
			float sintheta = sinf(pos.x);

			Vector2D texcoord = ComputeSphericalCoord(Vector3D(cosf(pos.y)*sintheta, sinf(pos.y)*sintheta, costheta));

			jexp = expf(-(j-shalf)*(j-shalf)/stddev2);
			c += BilinearInterpolate(src, texcoord.x*sfw, texcoord.y*sfh, image->width, image->height)*(iexp*jexp);
		}
	}

	c *= 1.f/(((float) M_PI)*stddev2);
	return c;
}

void HDRProbe::ConstructFilteredCubeMap(float *posx, float *negx, float *posy, float *negy, float *posz, float *negz, int res, int level) {
	// check to see if we are at the highest level in the mipmap
	// (i.e. 1 pixel)
	if ((res >> level) == 1) {
		// compute average of the environment
		Color3 avg = ComputeAverageColor();

		// set color to each map
		memcpy(posx, &avg, sizeof(Color3));
		memcpy(negx, &avg, sizeof(Color3));
		memcpy(posy, &avg, sizeof(Color3));
		memcpy(negy, &avg, sizeof(Color3));
		memcpy(posz, &avg, sizeof(Color3));
		memcpy(negz, &avg, sizeof(Color3));

	} else if (level == 0) {
		// base level of the mipmap so use the standard cubemap construction
		ConstructCubeMap(posx, negx, posy, negy, posz, negz, res, res);
		return;
	}

	Color3 *map[6];
	map[0] = (Color3*) posx;	map[1] = (Color3*) negx;
	map[2] = (Color3*) posy;	map[3] = (Color3*) negy;
	map[4] = (Color3*) posz;	map[5] = (Color3*) negz;

	Color3 *src	= (Color3*) image->data;

	float sfw = (float) image->width;
	float sfh = (float) image->height;

	int mapres = (res >> level);
	int steps = (1 << level);

	float inc = (float) (2.f/((float) (mapres)));

	float px=-1.f+inc/2.f, py=-1.f+inc/2.f;

	int bpos=0;

	Vector2D texcoord;
	Vector2D angles[4];

	for (int i=0; i < mapres; i++,py+=inc) {
		px=-1.f+inc/2.f;
		for (int j=0; j < mapres; j++,px+=inc) {
			// compute the generated angle for the position on the texture
			for (int k=0; k < 6; k++) {
				Vector2D &currAngle = ComputeAngles(ConvertTexcoordToCubeMap(k, px, py));

				ComputeFilteredArea(k, px, py, inc, angles);
				map[k][bpos] = FilterArea(currAngle, angles, steps, sfw, sfh);
			}

			bpos++;
		}
	}
}
