class pBoundBox
{
	public:
		pVector	min,	//!< bounding box minimum point
				max;	//!< bounding box maximum point

	static int facevert[6][4];		//!< indices of the corresponding vertices for each face
	static int edgevert[12][2];		//!< indices of the corresponding vertices for each edge
	static int edgefaces[12][2];	//!< indices of the corresponding faces for each edge
	static pVector vertnorm[8];		//!< normals of each vertex 
	static pVector edgenorm[12];	//!< normals of each edge 
	static pVector facenorm[6];		//!< normals of each face 
	static pVector facetexcoord[4];	//!< texture coordinates for each face vertex
	static pVector edgedir[12];		//!< normalized edge directions for each edge
	static pVector edgedirth[12];	//!< edgedir divided by 100
	
	//! Default constructor
	pBoundBox() 
	{ 
		min = 0.0;
		max = 0.0;
	}

	//! Construct a bounding box given the minimum and maximum points
	pBoundBox(const pVector& min, const pVector& max) 
	{ 
		pBoundBox::min = min;
		pBoundBox::max = max;
	}

	//! Copy-constructor
	pBoundBox(const pBoundBox& in)  
	{
		min = in.min;
		max = in.max;
	}

	//! Destructor
	virtual ~pBoundBox()  
	{ }

	//! Atribuition operator
	void operator=(const pBoundBox& in) 
	{ 
		min = in.min;
		max = in.max;
	}

	//! Return the vertex corresponding to the given index
	inline pVector get_vert(int ind) const 
	{
		switch(ind)
		{
		case 0: return min;
		case 1: return max;
		case 2: return pVector(max.x,min.y,min.z);
		case 3: return pVector(min.x,max.y,max.z);
		case 4: return pVector(max.x,max.y,min.z);
		case 5: return pVector(min.x,min.y,max.z);
		case 6: return pVector(min.x,max.y,min.z);
		case 7: return pVector(max.x,min.y,max.z);
		default: return pVector(0,0,0);
		}
	}
	
	//! Return the distance of the plane corresponding to the given index (0=min[x], 1=min[y], 2=min[z], 3=max[x], 4=max[y], 5=max[z])
	inline float get_plane_dist(int ind) const 
	{
		return ind>2?-min[ind-3]:max[ind];
	}

	//! Test for clipping between this and the bounding box whose minimum and maximum points are, respectively, bbmin and bbmax
	inline int clip_bbox(const pVector& bbmin, const pVector& bbmax) const 
	{
		if (max.x>=bbmin.x && min.x<=bbmax.x &&
			max.y>=bbmin.y && min.y<=bbmax.y &&
			max.z>=bbmin.z && min.z<=bbmax.z)
			return 1;
		return 0;
	}

	//! Test if point p is inside this bounding box
	inline int is_inside(const pVector& p) const 
	{
		return	p.x>=min.x && p.x<=max.x &&
				p.y>=min.y && p.y<=max.y &&
				p.z>=min.z && p.z<=max.z;
	}
	
	//! Test if point p is inside this bounding box
	inline int is_inside(const float *f) const 
	{
		return	f[0]>=min.x && f[0]<=max.x &&
				f[1]>=min.y && f[1]<=max.y &&
				f[2]>=min.z && f[2]<=max.z;
	}

	//! Collide ray defined by ray origin (ro) and ray direction (rd) with the bounding box
	int ray_intersect(const pVector& ro,const pVector& rd,float& tnear,float& tfar) const;
	//! Collide edge (p1,p2) moving in direction dir with edge (p3,p4)
	int edge_collision(const pVector& p1,const pVector& p2,const pVector& dir,const pVector& p3,const pVector& p4,float& dist,pVector& ip) const;
	//! Collide the bounding box moving in the direction dir with movement magnitude len with another bounding box (bbox)
	int collide(const pBoundBox& bbox,const pVector& dir,float& len,pVector& normal,pVector& ip) const;

	//! Reset all data 
	inline void reset() 
	{ 
		min.vec(BIG,BIG,BIG); 
		max.vec(-BIG,-BIG,-BIG); 
	}

	//! Add a point to the bounding box (expand its boundaries if necessary)
	void add_point(const pVector &p);
	void add_point(const float *p);

	//! draw boundbox as lines
	void draw() const;

	int collide(pRender *r,const pVector& p,const pVector& dir,float& len,pVector& ip,pVector& n) const;
};
