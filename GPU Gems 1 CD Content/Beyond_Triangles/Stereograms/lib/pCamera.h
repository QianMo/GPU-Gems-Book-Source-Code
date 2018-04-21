class pCamera
{
	public:
		pVector pos;	//!< translation position
		pVector	X,		//!< rotation local axis X direction
				Y,		//!< rotation local axis Y direction
				Z;		//!< rotation local axis Z direction
		pMatrix	mat,	//!< rotation matrix
				mat_t;	//!< inverse rotation matrix (transpose of the rotation matrix)
		float fov;		//!< camera angle
		pString name;	//!< camera name

		pAnimation anim_pos;
		pAnimation anim_fov;
		pAnimation anim_rot;

	//! Update the rotation matrices
	void update_mat();
	//! Rotate the local system axis by 'rot'
	void rotate(const pVector& rot);
	//! Rotate the local system axis of angle 'ang' around vector 'v'
	void rotate(float ang,const pVector& v);
	//! Rotate the local system axis from 'v' to 'u', in the plane defined by 'v' and 'u', at a maximum angle of 'maxang' degrees
	void rotate(const pVector &v, const pVector &u, float maxang=360);
	//! Align the z axis of the local system with the given vector
	void align_z(const pVector& z);
	
	//! Default constructor
	pCamera() : 
		pos(0,0,0), X(1,0,0), Y(0,1,0), Z(0,0,1), fov(60)
	{ }

	//! Copy-constructor
	pCamera(const pCamera& in) : 
		pos(in.pos), X(in.X), Y(in.Y), Z(in.Z), 
		mat(in.mat), mat_t(in.mat_t), 
		fov(in.fov), name(in.name),
		anim_pos(in.anim_pos),
		anim_fov(in.anim_fov),
		anim_rot(in.anim_rot)
	{ }

	//! Atribuition operator
	void operator=(const pCamera& in) 
	{ 
		pos = in.pos;
		X = in.X;
		Y = in.Y;
		Z = in.Z;
		mat = in.mat;
		mat_t = in.mat_t;
		fov = in.fov;
		name = in.name;
		anim_pos = in.anim_pos;
		anim_fov = in.anim_fov;
		anim_rot = in.anim_rot;
	}

	void write(FILE *fp) const;
	void read(FILE *fp);
};

class pCameraCtrl
{
	public:

	pCameraCtrl()
	{ }

	virtual ~pCameraCtrl()
	{ }

	virtual void check_input(pRender* r)=0;
};

class pCameraCtrlObs : public pCameraCtrl
{
	public:
		float movevel;
		float rotvel;
		float radius;

	pCameraCtrlObs() :
		movevel(100.0f),
		rotvel(100.0f),
		radius(20.0f)
	{ }

	virtual ~pCameraCtrlObs()
	{ }

	void check_input(pRender* r);
};

class pCameraCtrlWalk : public pCameraCtrl
{
	public:
		float mass;
		float bump;
		float friction;
		float gravity;
		float movevel;
		float rotvel;
		float radius;
		float height;
		float stepheight;
		float stepvel;
		float jumpheight;

		pVector vel;
		pVector force;
		float stepzmove;
		int contact;
		int flag;

	pCameraCtrlWalk() :
		mass(1.0f),
		bump(0.0f),
		friction(1.0f),
		gravity(980.0f),
		movevel(100.0f),
		rotvel(100.0f),
		radius(20.0f),
		height(60.0f),
		stepheight(30.0f),
		stepvel(100.0f),
		jumpheight(30.0f),
		
		vel(0),
		force(0),
		stepzmove(0),
		contact(0),
		flag(0)
	{ }

	virtual ~pCameraCtrlWalk()
	{ }

	void check_input(pRender* r);
};
