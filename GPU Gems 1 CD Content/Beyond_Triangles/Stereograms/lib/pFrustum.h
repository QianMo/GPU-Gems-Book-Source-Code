class pFrustum
{
	public:
		pVector verts[5];		//!< the frustum vertices
		pVector planes[5];		//!< the frustum planes
		int bboxindx[8][3];		//!< the frustum bounding box's vertices index

	//! Default constructor
	pFrustum() 
	{ 
		int i;
		for (i = 0; i<5; i++) 
		{
			verts[i]=0.0f;
			planes[i]=0.0f;
			bboxindx[i][0]=0;
			bboxindx[i][1]=0;
			bboxindx[i][2]=0;
		}
	}

	//! Copy-constructor
	pFrustum(const pFrustum& in) 
	{ 
		int i;
		for (i = 0; i<5; i++) 
		{
			verts[i]=in.verts[i];
			planes[i]=in.planes[i];
			bboxindx[i][0]=in.bboxindx[i][0];
			bboxindx[i][1]=in.bboxindx[i][1];
			bboxindx[i][2]=in.bboxindx[i][2];
		}
	}

	//! operator =
	void operator=(const pFrustum& in) 
	{ 
		int i;
		for (i = 0; i<5; i++) 
		{
			verts[i]=in.verts[i];
			planes[i]=in.planes[i];
			bboxindx[i][0]=in.bboxindx[i][0];
			bboxindx[i][1]=in.bboxindx[i][1];
			bboxindx[i][2]=in.bboxindx[i][2];
		}
	}

	//! Test the given bounding box against the frustum for clipping
	int clip_bbox(const pBoundBox& bbox) const;
	
	//! Build the view frustum from position 'pos' and system given by (X,Y,Z)
	void build(
		const pVector& pos,
		const pVector& X,const pVector& Y,const pVector& Z,
		float camangle,float aspect,float farplane);
	
	//! Draw the view frustum
	void draw() const;
};
