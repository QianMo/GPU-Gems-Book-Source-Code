#define OCTREE_MINFACES 16
#define OCTREE_MAXSTACK 256

class pOcTree
{
public:
	pOcTreeNode *root;
	pVertex *vert;
	int *facevert;

	//! Default constructor
	pOcTree();
	
	//! Default destructor
	virtual ~pOcTree();
	
	//! Free the tree data
	void reset();

	//! Builds the pOcTree for the given triangle face or Bezier face
	void build_tree(int nf,pFace *f,pVertex *v,int *fv,int maxdepth);
	
	pFace *ray_intersect(const pVector& ro,const pVector& rd,pVector& ip,float& dist,int thread,int isshadowray=0) const;

	void clip_ray(const pVector& ro,const pVector& rd,pArray<pOcTreeNode *>& nodes,int thread) const;
	void sort_nodes(pOcTreeNode **n,int num,int thread) const;

	void clip_frustum(const pFrustum& view,pArray<pOcTreeNode *>& nodes) const;
	void clip_frustum(int curdraw,const pFrustum& view,pArray<pFace *>& faces) const;
	void clip_bbox(int curdraw,const pBoundBox& bbox,pArray<pFace *>& faces,int shadow=0) const;
	void clip_frustum_bbox(int curdraw,const pFrustum& view,const pBoundBox& bbox,pArray<pFace *>& faces) const;

	void draw_boxes(const pFrustum& view);
};
