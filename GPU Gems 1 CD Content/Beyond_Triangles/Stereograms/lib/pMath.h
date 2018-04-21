//! random floating point number
#define FRAND (((rand()%10000)-5000)/5000.0f)
//! absolute random floating point number
#define FABSRAND ((rand()%10000)/10000.0f)

//! two vectors dot product
#define VECDOT(v1,v2) ((v1).x*(v2).x+(v1).y*(v2).y+(v1).z*(v2).z)

//! lower bound for overflow detection
#define SMALL 1.0e-4f
//! upper bound for overflow detection
#define BIG   1.0e+10f
//! underflow detection
#define ISZERO(a) ((a)>-SMALL && (a)<SMALL)

//! represents the value of 2*pi
#define TWOPI			6.28318530718f
//! represents the value of pi
#define PI				3.14159265359f
//! represents the value of pi/2
#define PI2				1.57079632679f
//! represents the value of pi/180
#define PIOVER180		1.74532925199433E-002f
//! represents the value of 180/pi
#define PIUNDER180		5.72957795130823E+001f
//! represents cos(45)
#define COS45			0.7071067811865475244f
//! represents the square root of 3
#define SQRT3			1.7320508075688772935f
//! represents the inverse square root of 3
#define INVSQRT3		0.5773502691896257645f

//! returns the bitwise representation of a floating point number
#define FPBITS(fp)		(*(int *)&(fp))
//! returns the absolute value of a floating point number in bitwise form
#define FPABSBITS(fp)	(FPBITS(fp)&0x7FFFFFFF)
//! returns the signal bit of a floating point number
#define FPSIGNBIT(fp)	(FPBITS(fp)&0x80000000)
//! returns the absolute value of a floating point number
#define FPABS(fp)		(*((int *)&fp)=FPABSBITS(fp))
//! returns the value of 1.0f in bitwise form
#define FPONEBITS		0x3F800000

//! returns maximum of 2 values
#define MAX2(a,b) ((a)>(b)?(a):(b))
//! returns maximum of 3 values
#define MAX3(a,b,c) MAX2(MAX2(a,b),c)
//! returns grayscale value for float color components
#define Intensity3(r,g,b) ((r)*0.299f + (g)*0.587f + (b)*0.114f)
//! returns grayscale value for float color vector
#define Intensity(rgb) Intensity3(rgb.x,rgb.y,rgb.z)

//! allocs aligned memory 
void *aligned_alloc(int n);
//! frees aligned memory 
void aligned_free(void *data);

float intpower(float a,unsigned n);
void calc_reflection_vector(const pVector& Rd,const pVector& N,pVector& newRd);
int calc_refraction_vector(const pVector Rd,const pVector& N,pVector& newRd,float n); 

class pVector
{
	public:
		float	x,	//!< the first component
				y,	//!< the second component
				z,	//!< the third component
				w;	//!< the fourth (of optional usage) component

	//! Default constructor
	pVector() : x(0), y(0), z(0), w(1)
	{ }

	//! Constructor from a floating point value, sets x, y and z to this value, and w to 1.0f
	pVector(float f) 
	{ x=y=z=f; w=1.0f; }

	//! Copy-constructor
	pVector(const pVector& v) : x(v.x), y(v.y), z(v.z), w(v.w) 
	{ }

	//! Constructor from 3 floating point numbers, sets x, y and z to their respective values, and w to 1.0f
	pVector(float x0,float y0,float z0)
	{ x=x0; y=y0; z=z0; w=1;}

	//! Constructor from 4 floating point numbers
	pVector(float x0,float y0,float z0,float w0)
	{ x=x0; y=y0; z=z0; w=w0; }

	//! Atribuition operator
	void operator=(const pVector& in) 
	{ x = in.x; y = in.y; z = in.z; w = in.w; }

	//! Nullifying function, sets x, y and z to zero, and w to 1.0f
	inline void null(void)
	{ x=y=z=0; w=1; }

	//! Returns the length of the pVector
	inline float length(void)
	{ return (float)sqrt(x*x+y*y+z*z); }

	//! Returns the square of the length
	inline float length2(void)
	{ return (float)x*x+y*y+z*z; }

	//!	Returns the distance to another pVector
	inline float distto(const pVector& v) const
	{ return (float)sqrt((v.x-x)*(v.x-x)+(v.y-y)*(v.y-y)+(v.z-z)*(v.z-z)); }

	//!	Returns the square of the distance to another pVector
	inline float distto2(const pVector& v) const
	{ return (float)(v.x-x)*(v.x-x)+(v.y-y)*(v.y-y)+(v.z-z)*(v.z-z); }

	//!	Set all the components to the given floating point value
	inline void vec(float f)
	{ x=y=z=w=f; }

	//!	Set components to the given floating point values, and set w=1.0f
	inline void vec(float x0,float y0,float z0)
	{ x=x0; y=y0; z=z0; w=1; }

	//!	Set components to the given floating point values
	inline void vec(float x0,float y0,float z0,float w0)
	{ x=x0; y=y0; z=z0; w=w0; }

	//!	Negate the first 3 components
	inline void negate(void)
	{ x=-x; y=-y; z=-z; }

	//!	Compute the cross-product of two given vectors
	inline void cross(const pVector& v1, const pVector& v2)
	{
		x=v1.y*v2.z-v1.z*v2.y;
		y=v1.z*v2.x-v1.x*v2.z;
		z=v1.x*v2.y-v1.y*v2.x;
	}

	//!	Normalize the pVector
	inline void normalize(void)
	{
		float len=(float)sqrt(x*x+y*y+z*z);
		if (FPBITS(len)==0) return;
		len=1.0f/len;
		x*=len; y*=len; z*=len;
	}

	inline float color_intensity()
	{	return x*0.299f+y*0.587f+z*0.114f; }

	//!	Reference indexing operator
	inline float& operator[](int i) { return (&x)[i]; };
	//!	Indexing operator
	inline float operator[](int i) const { return (&x)[i]; }

	//!	Negates the pVector (unary operator -)
	inline pVector operator-() const { return pVector(-x,-y,-z); }
};

//! Multiplies a pVector by a floating point value
inline void operator*=(pVector& v,float f)
{
  v.x*=f; v.y*=f; v.z*=f;
}
//! Divides a pVector by a floating point value
inline void operator/=(pVector& v,float f)
{
  v.x/=f; v.y/=f; v.z/=f;
}
//! Subtracts pVector 'v2' from the original pVector
inline void operator-=(pVector& v1,const pVector& v2)
{
  v1.x-=v2.x; v1.y-=v2.y; v1.z-=v2.z;
}
//! Adds the original pVector with another
inline void operator+=(pVector& v1, const pVector& v2)
{
  v1.x+=v2.x; v1.y+=v2.y; v1.z+=v2.z;
}
//! Multiplies the original pVector with another
inline void operator*=(pVector& v1, const pVector& v2)
{
  v1.x*=v2.x; v1.y*=v2.y; v1.z*=v2.z;
}
//! Divides the original pVector by another
inline void operator/=(pVector& v1, const pVector& v2)
{
  v1.x/=v2.x; v1.y/=v2.y; v1.z/=v2.z;
}

//! Subtracts pVector 'v2' from 'v1'
inline pVector operator-(pVector v1, const pVector& v2)
{
   v1.x-=v2.x; v1.y-=v2.y; v1.z-=v2.z;
   return v1;
}

//! Adds two vectors
inline pVector operator+(pVector v1, const pVector& v2)
{
   v1.x+=v2.x; v1.y+=v2.y; v1.z+=v2.z;
   return v1;
}
//! Multiplication between vectors
inline pVector operator*(pVector v1, const pVector& v2)
{
   v1.x*=v2.x; v1.y*=v2.y; v1.z*=v2.z;
   return v1;
}
//! Divides pVector 'v1' by pVector 'v2'
inline pVector operator/(pVector v1, const pVector& v2)
{
   v1.x/=v2.x; v1.y/=v2.y; v1.z/=v2.z;
   return v1;
}
//! Multiply a floating point value by a pVector
inline pVector operator*(float f,pVector v)
{
   v.x*=f; v.y*=f; v.z*=f;
   return v;
}
//! Multiply a pVector by a floating point value
inline pVector operator*(pVector v,float f)
{
   v.x*=f; v.y*=f; v.z*=f;
   return v;
}
//! Divide a pVector by a floating point value
inline pVector operator/(pVector v,float f)
{
   v.x/=f; v.y/=f; v.z/=f;
   return v;
}

class pPlane
{
	public:
		pVector normal;		//!< pPlane normal
		float d0;			//!< Perpendicular distance from the pPlane to the origin

	//! Default constructor
	pPlane() : d0(0) 
	{ }
	
	//! Constructor from components
	pPlane(const pVector& n, float dist) : 
		normal(n),d0(dist)
	{ }

	//! Copy-constructor
	pPlane(const pPlane& in) : normal(in.normal), d0(in.d0) 
	{ }

	//! Atribuition operator
	void operator=(const pPlane& in) 
	{ 
		normal = in.normal;
		d0 = in.d0;
	}

	//! Compute the perpendicular distance from a point to the pPlane
	inline float distance(const pVector &v) const
		{ return VECDOT(normal,v)-d0; }

	//! Intersect a ray (ro,rd) with the pPlane and return intersection point (ip) and distance to intersection (dist)
	int ray_intersect(const pVector& ro,const pVector& rd,pVector& ip,float& dist);
};

class pMatrix
{
	public:
		float m[4][4]; //!< matrix elements

	//! Default constructor, sets the identity matrix
	pMatrix() 
	{
		m[0][0]=m[1][1]=m[2][2]=m[3][3]=1.0f;
		m[0][1]=m[0][2]=m[0][3]=0.0f;
		m[1][0]=m[1][2]=m[1][3]=0.0f;
		m[2][0]=m[2][1]=m[2][3]=0.0f;
		m[3][0]=m[3][1]=m[3][2]=0.0f;
	}

	//! Copy-constructor
	pMatrix(const pMatrix& in) 
	{ 
		m[0][0]=in.m[0][0]; m[0][1]=in.m[0][1]; m[0][2]=in.m[0][2]; m[0][3]=in.m[0][3];
		m[1][0]=in.m[1][0]; m[1][1]=in.m[1][1]; m[1][2]=in.m[1][2]; m[1][3]=in.m[1][3];
		m[2][0]=in.m[2][0]; m[2][1]=in.m[2][1]; m[2][2]=in.m[2][2]; m[2][3]=in.m[2][3];
		m[3][0]=in.m[3][0]; m[3][1]=in.m[3][1]; m[3][2]=in.m[3][2]; m[3][3]=in.m[3][3];
	}

	//! Atribuition operator
	void operator=(const pMatrix& in) 
	{ 
		m[0][0]=in.m[0][0]; m[0][1]=in.m[0][1]; m[0][2]=in.m[0][2]; m[0][3]=in.m[0][3];
		m[1][0]=in.m[1][0]; m[1][1]=in.m[1][1]; m[1][2]=in.m[1][2]; m[1][3]=in.m[1][3];
		m[2][0]=in.m[2][0]; m[2][1]=in.m[2][1]; m[2][2]=in.m[2][2]; m[2][3]=in.m[2][3];
		m[3][0]=in.m[3][0]; m[3][1]=in.m[3][1]; m[3][2]=in.m[3][2]; m[3][3]=in.m[3][3];
	}

	//! Nullify all elements
	inline void null(void)
	{
		m[0][0]=m[0][1]=m[0][2]=m[0][3]= 
		m[1][0]=m[1][1]=m[1][2]=m[1][3]= 
		m[2][0]=m[2][1]=m[2][2]=m[2][3]= 
		m[3][0]=m[3][1]=m[3][2]=m[3][3]=0.0f;
	}

	//! Load the identity matrix
	inline void load_identity(void)
	{
		m[0][0]=m[1][1]=m[2][2]=m[3][3]=1.0f;
		m[0][1]=m[0][2]=m[0][3]=0.0f;
		m[1][0]=m[1][2]=m[1][3]=0.0f;
		m[2][0]=m[2][1]=m[2][3]=0.0f;
		m[3][0]=m[3][1]=m[3][2]=0.0f;
	}

	//! Set the matrix as the rotation matrix of angle given by 'ang' around direction 'dir'
	void set_rotation( float ang, const pVector& dir );
	//! Multiply the matrix by another with rotation given by 'ang' around direction 'dir'
	void rotate( float ang, const pVector& dir );

	//! Multiplication operator
	inline pMatrix operator*(const pMatrix& m1) const
	{
	  pMatrix m2;
	  for(int i=0; i<4; i++)
		{
			m2.m[i][0] = m[i][0]*m1.m[0][0] + m[i][1]*m1.m[1][0] +
				m[i][2]*m1.m[2][0] + m[i][3]*m1.m[3][0];
			m2.m[i][1] = m[i][0]*m1.m[0][1] + m[i][1]*m1.m[1][1] +
				m[i][2]*m1.m[2][1] + m[i][3]*m1.m[3][1];
			m2.m[i][2] = m[i][0]*m1.m[0][2] + m[i][1]*m1.m[1][2] +
				m[i][2]*m1.m[2][2] + m[i][3]*m1.m[3][2];
			m2.m[i][3] = m[i][0]*m1.m[0][3] + m[i][1]*m1.m[1][3] +
				m[i][2]*m1.m[2][3] + m[i][3]*m1.m[3][3];
		}
	  return m2;
	}
};

//! Multiplies a vector by a matrix
inline pVector operator*(const pVector& v,const pMatrix& m)
{
	pVector r;
	register float *f=(float *)&m;
	r.x = v.x*f[0] + v.y*f[4] + v.z*f[8] + v.w*f[12];
	r.y = v.x*f[1] + v.y*f[5] + v.z*f[9] + v.w*f[13];
	r.z = v.x*f[2] + v.y*f[6] + v.z*f[10] + v.w*f[14];
	r.w = v.x*f[3] + v.y*f[7] + v.z*f[11] + v.w*f[15];
	return r;
}
//! Multiplies a matrix by a vector 
inline pVector operator*(const pMatrix& m, const pVector& v)
{
	pVector r;
	register float *f=(float *)&m;
	r.x = v.x*f[0] + v.y*f[4] + v.z*f[8] + v.w*f[12];
	r.y = v.x*f[1] + v.y*f[5] + v.z*f[9] + v.w*f[13];
	r.z = v.x*f[2] + v.y*f[6] + v.z*f[10] + v.w*f[14];
	r.w = v.x*f[3] + v.y*f[7] + v.z*f[11] + v.w*f[15];
	return r;
}

class pQuaternion : public pVector
{
public:
	//! Copy-constructor
	pQuaternion(const pQuaternion& in) : pVector(in) 
	{ }

	//! Default constructor
	pQuaternion() : pVector() 
	{ }

	//! Construct the quaternion from a matrix
	pQuaternion(const pMatrix &mat);

	//! Construct the quaternion from the angle and axis
	pQuaternion(float angle, const pVector &axis)
	{
		float f=(float)sin(angle*PIOVER180*0.5f);
		x=axis.x*f;
		y=axis.y*f;
		z=axis.z*f;
		w=(float)cos(angle*PIOVER180*0.5f);
	}

	//! Normalize the quaternion
	void normalize();
	//! Converts the quaternion into a matrix
	void get_mat(pMatrix &mat) const;
	//! Get the rotation angle and axis
	void get_rotate(float &angle, pVector &axis) const;
	//! Interpolate two quaternions
	void lerp(const pQuaternion& q1,const pQuaternion& q2,float t);

	//! Multiplication between quaternions
	pQuaternion operator *(const pQuaternion& q);
	//! Addition between quaternions
	pQuaternion operator +(const pQuaternion& q);
};
