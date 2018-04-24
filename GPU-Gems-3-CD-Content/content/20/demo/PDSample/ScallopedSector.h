/// Provided Courtesy of Daniel Dunbar

#include <vector>

typedef struct {
	Vec2 P;
	float r, sign, d, theta, integralAtStart;
	float rSqrd, dSqrd;
} ArcData;

class ScallopedSector
{
public:
	Vec2 P;
	float a1, a2, area;
	
	ArcData arcs[2];

public:
	ScallopedSector(Vec2 &_Pt, float _a1, float _a2, Vec2 &P1, float r1, float sign1, Vec2 &P2, float r2, float sign2);

	float calcAreaToAngle(float angle);
	float calcAngleForArea(float area, RNG &rng);
	Vec2 sample(RNG &rng);

	float distToCurve(float angle, int index);

	void subtractDisk(Vec2 &C, float r, std::vector<ScallopedSector> *regions);

private:
	float canonizeAngle(float angle);

	void distToCircle(float angle, Vec2 &C, float r, float *d1_out, float *d2_out);
};

class ScallopedRegion
{
public:
	std::vector<ScallopedSector> *regions;
	float minArea;
	float area;

public:
	ScallopedRegion(Vec2 &P, float r1, float r2, float minArea=.00000001);
	~ScallopedRegion();

	bool isEmpty() { return regions->size()==0; }
	void subtractDisk(Vec2 C, float r);

	Vec2 sample(RNG &rng);
};
