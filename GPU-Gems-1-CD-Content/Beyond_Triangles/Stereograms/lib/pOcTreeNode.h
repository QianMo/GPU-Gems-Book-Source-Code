class pOcTreeNode 
{
public:
	pBoundBox bbox;				//!< Node bound box
	pArray<pFace *> faces;		//!< Node faces
	pOcTreeNode *nodes[8];		//!< Node childs
	float maxdist[P_MAX_THREADS];	//!< Distance from bbox center to camera
	
	//! Default constructor
	pOcTreeNode();

	//! Default destructor
	virtual ~pOcTreeNode();

	//! Copy constructor
	pOcTreeNode(pOcTreeNode& in);

	//! Split faces into child nodes if a subdivison is need (used on the pOcTree build process)
	void build_node(pVertex *vert,int *facevert,int depth,int maxdepth);
};

