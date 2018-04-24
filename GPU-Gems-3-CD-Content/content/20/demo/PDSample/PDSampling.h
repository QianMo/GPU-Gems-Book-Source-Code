/// Provided Courtesy of Daniel Dunbar

#include <vector>
#include <cmath>
#include "RNG.h"

#define kMaxPointsPerCell 9

double timeInSeconds(void);

class RangeList;
class ScallopedRegion;

class Vec2 {
public:
	Vec2() {};
	Vec2(float _x, float _y) : x(_x), y(_y) {};

	float x,y;

	float length() { return sqrt(x*x + y*y); }

	bool operator ==(const Vec2 &b) const { return x==b.x && y==b.y; }
	Vec2 operator +(Vec2 b) { return Vec2(x+b.x, y+b.y); }
	Vec2 operator -(Vec2 b) { return Vec2(x-b.x, y-b.y); }
	Vec2 operator *(Vec2 b) { return Vec2(x*b.x, y*b.y); }
	Vec2 operator /(Vec2 b) { return Vec2(x/b.x, y*b.y); }

	Vec2 operator +(float n) { return Vec2(x+n, y+n); }
	Vec2 operator -(float n) { return Vec2(x-n, y-n); }
	Vec2 operator *(float n) { return Vec2(x*n, y*n); }
	Vec2 operator /(float n) { return Vec2(x/n, y*n); }

	Vec2 &operator +=(Vec2 b) { x+=b.x; y+=b.y; return *this; }
	Vec2 &operator -=(Vec2 b) { x-=b.x; y-=b.y; return *this; }
	Vec2 &operator *=(Vec2 b) { x*=b.x; y*=b.y; return *this; }
	Vec2 &operator /=(Vec2 b) { x/=b.x; y/=b.y; return *this; }

	Vec2 &operator +=(float n) { x+=n; y+=n; return *this; }
	Vec2 &operator -=(float n) { x-=n; y-=n; return *this; }
	Vec2 &operator *=(float n) { x*=n; y*=n; return *this; }
	Vec2 &operator /=(float n) { x/=n; y/=n; return *this; }
};

///

class PDSampler {
protected:
	RNG m_rng;
	std::vector<int> m_neighbors;
	
	int (*m_grid)[kMaxPointsPerCell];
	int m_gridSize;
	float m_gridCellSize;
	
public:
	std::vector<Vec2> points;
	float radius;
	bool isTiled;

public:
	PDSampler(float radius, bool isTiled, bool usesGrid=true);
	virtual ~PDSampler() { };

	//

	bool pointInDomain(Vec2 &a);

		// return shortest distance between _a_ 
		// and _b_ (accounting for tiling)
	float getDistanceSquared(Vec2 &a, Vec2 &b) { Vec2 v = getTiled(b-a); return v.x*v.x + v.y*v.y; }
	float getDistance(Vec2 &a, Vec2 &b) { return sqrt(getDistanceSquared(a, b)); }

		// generate a random point in square
	Vec2 randomPoint();

		// return tiled coordinates of _v_
	Vec2 getTiled(Vec2 v);

		// return grid x,y for point
	void getGridXY(Vec2 &v, int *gx_out, int *gy_out);

		// add _pt_ to point list and grid
	void addPoint(Vec2 pt);

		// populate m_neighbors with list of
		// all points within _radius_ of _pt_
		// and return number of such points
	int findNeighbors(Vec2 &pt, float radius);

		// return distance to closest neighbor within _radius_
	float findClosestNeighbor(Vec2 &pt, float radius);

		// find available angle ranges on boundary for candidate 
		// by subtracting occluded neighbor ranges from _rl_
	void findNeighborRanges(int index, RangeList &rl);

		// extend point set by boundary sampling until domain is 
		// full
	void maximize();

		// apply one step of Lloyd relaxation
	void relax();

	//

	virtual void complete() = 0;
};

///

class DartThrowing : public PDSampler {
private:
	int m_minMaxThrows, m_maxThrowsMult;

public:
	DartThrowing(float radius, bool isTiled, int minMaxThrows, int maxThrowsMult);

	virtual void complete();
};

///

class BestCandidate : public PDSampler {
private:
	int m_multiplier, m_N;

public:
	BestCandidate(float radius, bool isTiled, int multiplier);

	virtual void complete();
};

///

class BoundarySampler : public PDSampler {
public:
	BoundarySampler(float radius, bool isTiled) : PDSampler(radius, isTiled) {};

	virtual void complete();
};

///

class PureSampler : public PDSampler {
public:
	PureSampler(float radius) : PDSampler(radius, true) {};

	virtual void complete();
};

///

class LinearPureSampler : public PDSampler {
public:
	LinearPureSampler(float radius) : PDSampler(radius, true) {};

	virtual void complete();
};

///

class PenroseSampler : public PDSampler {
public:
	PenroseSampler(float radius) : PDSampler(radius, false, false) {};

	virtual void complete();
};

///

class UniformSampler : public PDSampler {
public:
	UniformSampler(float radius) : PDSampler(radius, false, false) {};

	virtual void complete();
};
