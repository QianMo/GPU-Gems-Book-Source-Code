class pMesh
{
	public:
		int nvert;
		int nface;
		int nmat;
		int ncam;
		int nlight;
		int *facevert;
		pFace *face;
		pVertex *vert;
		pVector *tangent;
		pMaterial *mat;
		pCamera *cam;
		pLight *light;
		pVector ambient;
		pVector bg_color;
		pString bg_pic;
		pBoundBox bbox;
		int bg_mode;
		pString filename;
	
	pMesh();
	~pMesh();

	void reset();
	void compute_normals(int mode=15);
	void compute_tangents();

	void set_numface(int nf,int keepold=1);
	void set_numvert(int nv,int keepold=1);
	void set_nummat(int nm,int keepold=1);
	void set_numcam(int nc,int keepold=1);
	void set_numlight(int nl,int keepold=1);
	
	int load_p3d(const char *file);
	int save_p3d(const char *file) const;
#ifdef P_SUPPORT_3DS
	int load_3ds(const char *file);
#endif
	void draw_faces(pRender *r,const pArray<pFace *>& f,int mode=-1) const;
	void draw_faces_shadow(const pRender *r,const pArray<pFace *>& f,const pVector& lightpos) const;
	void draw_wire(const pArray<pFace *>& f) const;

	void compute_nuv(const pFace *f,const pVector& ip,pVector& normal,pVector& texcoord) const;
	int ray_intersect(const pVector& ro,const pVector& rd,pVector& ip,float& dist) const;

	void group_faces_material(pFace **f,int num) const;

	void array_lock(int drawflag) const;
	void array_draw() const;
	void array_unlock() const;
};
