//Copyright and Disclaimer:
//This code is copyright Daniel Scherzer, 2004.
#ifndef PERLIN_H
#define PERLIN_H

#include <math.h>
#include <malloc.h>

class Perlin {
protected:   
	static float curve(float t) { 
		return t * t * t * (t * (t * 6 - 15) + 10); 
	}

	static int lerp(int t, int a, int b) { 
		return a + (t * (b - a) >> 12); 
	}

	static float lerp(float t, float a, float b) { 
		return a + t * (b - a); 
	}

	static float grad(int hash, float x, float y, float z) {
		int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
		float u = (h < 8) ? x : y;                 // INTO 12 GRADIENT DIRECTIONS.
		float v = (h < 4) ? y : ((h==12)||(h==14)) ? x : z;
		return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
	}

	static float grad(int hash, float x, float y) {
		int h = hash & 15;
		float u = (h < 8) ? x : y;
		float v = (h < 4) ? y : ((h==12)||(h==14)) ? x : 0.0f;
		return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
	}

	class Perm {
		int *p;
	public:
		Perm() {
			const int permutation[] = { 151,160,137,91,90,15,
				131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
				190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
				88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
				77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
				102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
				135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
				5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
				223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
				129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
				251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
				49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
				138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
			};
			p = new int[512];
			for(int i = 0; i < 256 ; i++) {
				p[256+i] = p[i] = permutation[i]; 
			}
		}
		~Perm() {
			delete[] p;
		}
		inline const int operator[](const int id) const {
			return p[id];
		}
	};

	const static Perm p;
public:
	typedef unsigned char BYTE;
	typedef BYTE Tuppel4[4];

	static float noise(float x, float y) {
		int X = (int)floor(x) & 255;
		int Y = (int)floor(y) & 255;
		
		//calc fractional part
		x -= floor(x);                                
		y -= floor(y);                               
		
		float u = curve(x);                                // COMPUTE FADE CURVES
		float v = curve(y);                                // FOR EACH OF X,Y.
		
		int A = p[X  ]+Y, AA = p[A], AB = p[A+1],
			B = p[X+1]+Y, BA = p[B], BB = p[B+1];

		float gAA = grad(p[AA], x, y);
		float gBA = grad(p[BA], x-1, y);
		float gAB = grad(p[AB], x, y-1);
		float gBB = grad(p[BB], x-1, y-1);

		return lerp(v, lerp(u, gAA, gBA), lerp(u, gAB, gBB));
	}

	static float noise(float x, float y, float z) {
		int X = (int)floor(x) & 255;                  // FIND UNIT CUBE THAT
		int Y = (int)floor(y) & 255;                  // CONTAINS POINT.
		int Z = (int)floor(z) & 255;
		
		//calc fractional part
		x -= floor(x);                                
		y -= floor(y);                               
		z -= floor(z);
		
		float u = curve(x);                                // COMPUTE FADE CURVES
		float v = curve(y);                                // FOR EACH OF X,Y,Z.
		float w = curve(z);
		
		int A = p[X  ]+Y, AA = p[A]+Z, AB = p[A+1]+Z,      // HASH COORDINATES OF
			B = p[X+1]+Y, BA = p[B]+Z, BB = p[B+1]+Z;      // THE 8 CUBE CORNERS,

		float gAA = grad(p[AA], x, y, z);
		float gBA = grad(p[BA], x-1, y, z);
		float gAB = grad(p[AB], x, y-1, z);
		float gBB = grad(p[BB], x-1, y-1, z);
		float gBA1 = grad(p[BA+1], x-1, y, z-1);
		float gAA1 = grad(p[AA+1], x, y, z-1);
		float gAB1 = grad(p[AB+1], x, y-1, z-1);
		float gBB1 = grad(p[BB+1], x-1, y-1, z-1);

		return lerp(w, 
			lerp(v, lerp(u,  gAA,  gBA), lerp(u,  gAB,  gBB)),
			lerp(v, lerp(u, gAA1, gBA1), lerp(u, gAB1 , gBB1)
			));
	}
	
	static unsigned char* createField(const unsigned X, const unsigned Y, const unsigned freq) {
		BYTE* const pField = (BYTE* const)new BYTE[X*Y];
		if(!pField) return 0;
		// Setup procedural image for the texture.
		const float fact = (float)(1.0/sqrt((float)X*Y))*freq;
		BYTE* p = pField;
		for(unsigned x = 0; x < X; x++) {
			for(unsigned y = 0; y < Y; y++, p++) { 
				const float n = Perlin::noise(x*fact,y*fact);
				*p = (BYTE)((n+1.0)*(256.0*0.5));
			}
		}
		return pField;
	}

	static Tuppel4* createFieldWithOctave123(const unsigned X, const unsigned Y, const unsigned baseFreq) {
		const unsigned CNT = 4;
		BYTE* const pField = (BYTE* const)new Tuppel4[X*Y];
		if(!pField) return 0;
		// Setup procedural image for the texture.
		unsigned freq = baseFreq;
		const float baseFact = 1.0f/sqrt((float)X*Y);
		for(BYTE *startPtr = pField; startPtr < pField+CNT; startPtr++) {
			const float fact = baseFact*freq;
			BYTE *ptr = startPtr;
			for(unsigned x = 0; x < X; x++) {
				for(unsigned y = 0; y < Y; y++, ptr += CNT) { 
					const float n = Perlin::noise(x*fact,y*fact);
					*ptr = (BYTE)((n+1.0)*(256.0*0.5));
				}
			}
			freq *= 2;
		}
		return (Tuppel4*)pField;
	}

	static Tuppel4* createFieldWithOctave123(const unsigned X, const unsigned Y, const unsigned Z, const unsigned baseFreq) {
		const unsigned CNT = 4;
		BYTE* const pField = (BYTE* const)new Tuppel4[X*Y*Z];
		if(!pField) return 0;
		// Setup procedural image for the texture.
		unsigned freq = baseFreq;
		const float baseFact = 1.0f/sqrt(sqrt((float)X*Y)*Z);
		for(BYTE *startPtr = pField; startPtr < pField+CNT; startPtr++) {
			const float fact = baseFact*freq;
			BYTE *ptr = startPtr;
			for(unsigned x = 0; x < X; x++) {
				for(unsigned y = 0; y < Y; y++) { 
					for(unsigned z = 0; z < Z; z++, ptr += CNT) { 
						const float n = Perlin::noise(x*fact,y*fact,z*fact);
						*ptr = (BYTE)((n+1.0)*(256.0*0.5));
					}
				}
			}
			freq *= 2;
		}
		return (Tuppel4*)pField;
	}

};

#endif