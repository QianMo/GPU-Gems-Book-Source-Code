/// Provided Courtesy of Daniel Dunbar

#include "math.h"

#include <map>

#include "PDSampling.h"
#include "RangeList.h"
#include "ScallopedSector.h"
#include "WeightedDiscretePDF.h"

#include "quasisampler_prototype.h"

typedef std::vector<int> IntVector;

#ifdef _WIN32
#include <windows.h>
double timeInSeconds()
{
	static double perfFreq;
	static int hasPerfTimer = -1;

	if (hasPerfTimer==-1) {
		LARGE_INTEGER perfFreqCountsPerSec;

		if (QueryPerformanceFrequency(&perfFreqCountsPerSec)) {
			hasPerfTimer = 1;
			perfFreq = (double) perfFreqCountsPerSec.QuadPart;
		} else {
			hasPerfTimer = 0;
		}
	}

	LARGE_INTEGER count;
	if (hasPerfTimer && QueryPerformanceCounter(&count)) {
		return (double) count.QuadPart/perfFreq;
	} else {
		return GetTickCount()/1000.0;
	}
}
#else
#include <time.h>
#include <sys/time.h>

double timeInSeconds()
{
	struct timeval tv;
	struct timezone tz;
	
	gettimeofday(&tv, &tz);
	return tv.tv_sec + tv.tv_usec/1000000.0;
}
#endif


///

PDSampler::PDSampler(float _radius, bool _isTiled, bool usesGrid) :
	m_rng((unsigned long) (timeInSeconds()*1000)),
	radius(_radius),
	isTiled(_isTiled)
{
	if (usesGrid) {
			// grid size is chosen so that 4*radius search only 
			// requires searching adjacent cells, this also
			// determines max points per cell
		m_gridSize = (int) ceil(2./(4.*_radius));
		if (m_gridSize<2) m_gridSize = 2;

		m_gridCellSize = 2.0f/m_gridSize;
		m_grid = new int[m_gridSize*m_gridSize][kMaxPointsPerCell];

		for (int y=0; y<m_gridSize; y++) {
			for (int x=0; x<m_gridSize; x++) {
				for (int k=0; k<kMaxPointsPerCell; k++) {
					m_grid[y*m_gridSize + x][k] = -1;
				}
			}
		}
	} else {
		m_gridSize = 0;
		m_gridCellSize = 0;
		m_grid = 0;
	}
}
	
bool PDSampler::pointInDomain(Vec2 &a)
{
	return -1<=a.x && -1<=a.y && 1>=a.x && 1>=a.y;
}

Vec2 PDSampler::randomPoint()
{
	return Vec2(2*m_rng.getFloatL()-1, 2*m_rng.getFloatL()-1);
}

Vec2 PDSampler::getTiled(Vec2 v)
{
	float x = v.x, y = v.y;

	if (isTiled) {
		if (x<-1) x += 2;
		else if (x>1) x -= 2;

		if (y<-1) y += 2;
		else if (y>1) y -= 2;
	}

	return Vec2(x,y);
}

void PDSampler::getGridXY(Vec2 &v, int *gx_out, int *gy_out)
{
	int gx = *gx_out = (int) floor(.5*(v.x + 1)*m_gridSize);
	int gy = *gy_out = (int) floor(.5*(v.y + 1)*m_gridSize);
	if (gx<0 || gx>=m_gridSize || gy<0 || gy>=m_gridSize) {
		printf("Internal error, point outside grid was generated, ignoring.\n");
	}
}

void PDSampler::addPoint(Vec2 pt)
{
	int i, gx, gy, *cell;

	points.push_back(pt);

	if (m_grid) {
		getGridXY(pt, &gx, &gy);
		cell = m_grid[gy*m_gridSize + gx];
		for (i=0; i<kMaxPointsPerCell; i++) {
			if (cell[i]==-1) {
				cell[i] = (int) points.size()-1;
				break;
			}
		}
		if (i==kMaxPointsPerCell) {
			printf("Internal error, overflowed max points per grid cell. Exiting.\n");
			exit(1);
		}
	}
}

int PDSampler::findNeighbors(Vec2 &pt, float distance)
{
	if (!m_grid) {
		printf("Internal error, sampler cannot search without grid.\n");
		exit(1);
	}

	float distanceSqrd = distance*distance;
	int i, j, k, gx, gy, N = (int) ceil(distance/m_gridCellSize);
	if (N>(m_gridSize>>1)) N = m_gridSize>>1;
	
	m_neighbors.clear();
	getGridXY(pt, &gx, &gy);
	for (j=-N; j<=N; j++) {
		for (i=-N; i<=N; i++) {
			int cx = (gx+i+m_gridSize)%m_gridSize;
			int cy = (gy+j+m_gridSize)%m_gridSize;
			int *cell = m_grid[cy*m_gridSize + cx];

			for (k=0; k<kMaxPointsPerCell; k++) {
				if (cell[k]==-1) {
					break;
				} else {
					if (getDistanceSquared(pt, points[cell[k]])<distanceSqrd)
						m_neighbors.push_back(cell[k]);
				}
			}
		}
	}

	return (int) m_neighbors.size();
}

float PDSampler::findClosestNeighbor(Vec2 &pt, float distance)
{
	if (!m_grid) {
		printf("Internal error, sampler cannot search without grid.\n");
		exit(1);
	}

	float closestSqrd = distance*distance;
	int i, j, k, gx, gy, N = (int) ceil(distance/m_gridCellSize);
	if (N>(m_gridSize>>1)) N = m_gridSize>>1;
	
	getGridXY(pt, &gx, &gy);
	for (j=-N; j<=N; j++) {
		for (i=-N; i<=N; i++) {
			int cx = (gx+i+m_gridSize)%m_gridSize;
			int cy = (gy+j+m_gridSize)%m_gridSize;
			int *cell = m_grid[cy*m_gridSize + cx];

			for (k=0; k<kMaxPointsPerCell; k++) {
				if (cell[k]==-1) {
					break;
				} else {
					float d = getDistanceSquared(pt, points[cell[k]]);

					if (d<closestSqrd)
						closestSqrd = d;
				}
			}
		}
	}

	return sqrt(closestSqrd);
}

void PDSampler::findNeighborRanges(int index, RangeList &rl)
{
	if (!m_grid) {
		printf("Internal error, sampler cannot search without grid.\n");
		exit(1);
	}

	Vec2 &candidate = points[index];
	float rangeSqrd = 4*4*radius*radius;
	int i, j, k, gx, gy, N = (int) ceil(4*radius/m_gridCellSize);
	if (N>(m_gridSize>>1)) N = m_gridSize>>1;
	
	getGridXY(candidate, &gx, &gy);

	int xSide = (candidate.x - (-1 + gx*m_gridCellSize))>m_gridCellSize*.5;
	int ySide = (candidate.y - (-1 + gy*m_gridCellSize))>m_gridCellSize*.5;
	int iy = 1;
	for (j=-N; j<=N; j++) {
		int ix = 1;

		if (j==0) iy = ySide;
		else if (j==1) iy = 0;

		for (i=-N; i<=N; i++) {
			if (i==0) ix = xSide;
			else if (i==1) ix = 0;

				// offset to closest cell point
			float dx = candidate.x - (-1 + (gx+i+ix)*m_gridCellSize);
			float dy = candidate.y - (-1 + (gy+j+iy)*m_gridCellSize);

			if (dx*dx+dy*dy<rangeSqrd) {
				int cx = (gx+i+m_gridSize)%m_gridSize;
				int cy = (gy+j+m_gridSize)%m_gridSize;
				int *cell = m_grid[cy*m_gridSize + cx];

				for (k=0; k<kMaxPointsPerCell; k++) {
					if (cell[k]==-1) {
						break;
					} else if (cell[k]!=index) {
						Vec2 &pt = points[cell[k]];
						Vec2 v = getTiled(pt-candidate);
						float distSqrd = v.x*v.x + v.y*v.y;

						if (distSqrd<rangeSqrd) {
							float dist = sqrt(distSqrd);
							float angle = atan2(v.y,v.x);
							float theta = acos(.25f*dist/radius);

							rl.subtract(angle-theta, angle+theta);
						}
					}
				}
			}
		}
	}
}

void PDSampler::maximize()
{
	RangeList rl(0,0);
	int i, N = (int) points.size();

	for (i=0; i<N; i++) {
		Vec2 &candidate = points[i];

		rl.reset(0, (float) M_PI*2);
		findNeighborRanges(i, rl);
		while (rl.numRanges) {
			RangeEntry &re = rl.ranges[m_rng.getInt31()%rl.numRanges];
			float angle = re.min + (re.max-re.min)*m_rng.getFloatL();
			Vec2 pt = getTiled(Vec2(candidate.x + cos(angle)*2*radius,
									candidate.y + sin(angle)*2*radius));

			addPoint(pt);
			rl.subtract(angle - (float) M_PI/3, angle + (float) M_PI/3);
		}
	}
}

void PDSampler::relax()
{
	FILE *tmp = fopen("relaxTmpIn.txt","w");
	int dim, numVerts, numFaces;
	Vec2 *verts = 0;
	int numPoints = (int) points.size();

		// will overwrite later
	fprintf(tmp, "2                  \n");
	for (int i=0; i<(int) points.size(); i++) {
		Vec2 &pt = points[i];
		fprintf(tmp, "%f %f\n", pt.x, pt.y);
	}
	for (int y=-1; y<=1; y++) {
		for (int x=-1; x<=1; x++) {
			if (x || y) {
				for (int i=0; i<(int) points.size(); i++) {
					Vec2 &pt = points[i];
					if (fabs(pt.x+x*2)-1<radius*4 || fabs(pt.y+y*2)-1<radius*4) {
						fprintf(tmp, "%f %f\n", pt.x+x*2, pt.y+y*2);
						numPoints++;
					}
				}
			}
		}
	}
	fseek(tmp, 0, 0);
	fprintf(tmp, "2 %d", numPoints);
	fclose(tmp);

	tmp = fopen("relaxTmpOut.txt", "w");
	fclose(tmp);
	system("qvoronoi p FN < relaxTmpIn.txt > relaxTmpOut.txt");

	tmp = fopen("relaxTmpOut.txt", "r");
	fscanf(tmp, "%d\n%d\n", &dim, &numVerts);

	if (dim!=2) {
		printf("Error calling out to qvoronoi, skipping relaxation.\n");
		goto exit;
	}

	verts = new Vec2[numVerts];
	for (int i=0; i<numVerts; i++) {
		fscanf(tmp, "%f %f\n", &verts[i].x, &verts[i].y);
	}

	fscanf(tmp, "%d\n", &numFaces);

	for (int i=0; i<(int) points.size(); i++) {
		Vec2 center(0,0);
		int N, skip=0;

		fscanf(tmp, "%d", &N);
		for (int j=0; j<N; j++) {
			int index;

			fscanf(tmp, "%d", &index);
			if (index<0) {
				skip = 1;
			} else {
				center += verts[index];
			}
		}

		if (!skip) {
			center *= (1.0f/N);
			points[i] = getTiled(center);
		}
	}

exit:
	if (verts) delete verts;
}

///

DartThrowing::DartThrowing(float radius, bool isTiled, int minMaxThrows, int maxThrowsMult) :
	PDSampler(radius, isTiled),
	m_minMaxThrows(minMaxThrows),
	m_maxThrowsMult(maxThrowsMult)
{
	;
}

void DartThrowing::complete()
{
	while (1) {
		int i, N = (int) points.size()*m_maxThrowsMult;
		if (N<m_minMaxThrows) N = m_minMaxThrows;

		for (i=0; i<N; i++) {
			Vec2 pt = randomPoint();

			findNeighbors(pt, 2*radius);

			if (!m_neighbors.size()) {
				addPoint(pt);
				break;
			}
		}

		if (i==N)
			break;
	}
}

///

BestCandidate::BestCandidate(float radius, bool isTiled, int multiplier) :
	PDSampler(radius, isTiled),
	m_multiplier(multiplier),
	m_N((int) (.7/(radius*radius)))
{
	;
}

void BestCandidate::complete()
{
	for (int i=0; i<m_N; i++) {
		Vec2 best(0,0);
		float bestDistance = 0;
		int count = 1 + (int) points.size()*m_multiplier;

		for (int j=0; j<count; j++) {
			Vec2 pt = randomPoint();
			float closest = 2;

			closest = findClosestNeighbor(pt, 4*radius);
			if (j==0 || closest>bestDistance) {
				bestDistance = closest;
				best = pt;
			}
		}

		addPoint(best);
	}
}

///

void BoundarySampler::complete()
{
	RangeList rl(0,0);
	IntVector candidates;

	addPoint(randomPoint());
	candidates.push_back((int) points.size()-1);

	while (candidates.size()) {
		int c = m_rng.getInt32()%candidates.size();
		int index = candidates[c];
		Vec2 candidate = points[index];
		candidates[c] = candidates[candidates.size()-1];
		candidates.pop_back();

		rl.reset(0, (float) M_PI*2);
		findNeighborRanges(index, rl);
		while (rl.numRanges) {
			RangeEntry &re = rl.ranges[m_rng.getInt32()%rl.numRanges];
			float angle = re.min + (re.max-re.min)*m_rng.getFloatL();
			Vec2 pt = getTiled(Vec2(candidate.x + cos(angle)*2*radius,
									candidate.y + sin(angle)*2*radius));

			addPoint(pt);
			candidates.push_back((int) points.size()-1);

			rl.subtract(angle - (float) M_PI/3, angle + (float) M_PI/3);
		}
	}
}

///

typedef std::map<int, ScallopedRegion*> RegionMap;

void PureSampler::complete()
{
	Vec2 pt = randomPoint();
	ScallopedRegion *rgn = new ScallopedRegion(pt, radius*2, radius*4);
	RegionMap regions;
	WeightedDiscretePDF<int> regionsPDF;

	addPoint(pt);
	regions[(int) points.size()-1] = rgn;
	regionsPDF.insert((int) points.size()-1, rgn->area);

	while (regions.size()) {
		int idx = regionsPDF.choose(m_rng.getFloatL());
		
		pt = getTiled(((*regions.find(idx)).second)->sample(m_rng));
		rgn = new ScallopedRegion(pt, radius*2, radius*4);

		findNeighbors(pt, radius*8);
		for (IntVector::const_iterator it=m_neighbors.begin(); it!=m_neighbors.end(); it++) {
			int nIdx = *it;
			Vec2 &n = points[nIdx];
			
			rgn->subtractDisk(pt+getTiled(n-pt), radius*4);

			RegionMap::iterator entry = regions.find(nIdx);
			if (entry!=regions.end()) {
				ScallopedRegion *nRgn = (*entry).second;
				nRgn->subtractDisk(n+getTiled(pt-n), radius*2);

				if (nRgn->isEmpty()) {
					regions.erase(entry);
					regionsPDF.remove(nIdx);
					delete nRgn;
				} else {
					regionsPDF.update(nIdx, nRgn->area);
				}
			}
		}

		addPoint(pt);

		if (!rgn->isEmpty()) {
			regions[(int) points.size()-1] = rgn;
			regionsPDF.insert((int) points.size()-1, rgn->area);
		} else {
			delete rgn;
		}
	}
}

///

void LinearPureSampler::complete()
{
	IntVector candidates;

	addPoint(randomPoint());
	candidates.push_back((int) points.size()-1);

	while (candidates.size()) {
		int c = m_rng.getInt32()%candidates.size();
		int index = candidates[c];
		Vec2 candidate = points[index];
		candidates[c] = candidates[candidates.size()-1];
		candidates.pop_back();

		ScallopedRegion sr(candidate, radius*2, radius*4);
		findNeighbors(candidate, radius*8);
		
		for (IntVector::const_iterator it=m_neighbors.begin(); it!=m_neighbors.end(); it++) {
			int nIdx = *it;
			Vec2 &n = points[nIdx];
			Vec2 nClose = candidate + getTiled(n-candidate);

			if (nIdx<index) {
				sr.subtractDisk(nClose, radius*4);
			} else {
				sr.subtractDisk(nClose, radius*2);
			}
		}

		while (!sr.isEmpty()) {
			Vec2 p = sr.sample(m_rng);
			Vec2 pt = getTiled(p);

			addPoint(pt);
			candidates.push_back((int) points.size()-1);

			sr.subtractDisk(p, radius*2);
		}
	}
}

///

class PenroseQuasisampler : public Quasisampler {
	unsigned int val;

public:
	PenroseQuasisampler(unsigned int _val) : Quasisampler(100,100), val(_val) {}

	unsigned int getImportanceAt(Point2D pt) { return val; }
};

void PenroseSampler::complete()
{
	PenroseQuasisampler s((unsigned int) (9.1/(radius*radius)));
	std::vector<Point2D> pts = s.getSamplingPoints();

	for (std::vector<Point2D>::iterator it=pts.begin(); it!=pts.end(); it++ ) {
		Vec2 pt((float) it->x/50.f - 1.0f, (float) it->y/50.f - 1.0f);

		if (pointInDomain(pt)) {
			addPoint(pt);
		}
	}
}

///

void UniformSampler::complete()
{
	int N = (int) (.75/(radius*radius));

	for (int i=0; i<N; i++) {
		addPoint(randomPoint());
	}
}

