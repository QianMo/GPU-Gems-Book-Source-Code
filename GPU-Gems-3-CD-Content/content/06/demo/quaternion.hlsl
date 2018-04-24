///////////////// ///////////////// Quat Lib ///////////////// ///////////////// 
#define USE_FAST_QUAT_2_MATRIX
//#define NORMALIZE_FAST_QUAT_2_MATRIX

float4 quatAxisAngle(float3 axis, float angle)
{
	float sinha = sin(angle * 0.5);
	float cosha = cos(angle * 0.5);

	return float4(
		axis * sinha,
		cosha );
}

float4 quatAroundY(float angle)
{
	float sinha = sin(angle * 0.5);
	float cosha = cos(angle * 0.5);
	return float4(0, sinha, 0, cosha);
}

float4 quatConjugate(float4 q)
{
	return float4(-q.x, -q.y, -q.z, q.w);
}

float3 quatRotateVec(float3 v, float4 q)
{
	// _q = conjugate(q) = {-q.x, -q.y, -q.z, q.w}
	// r = _q * {v.x, v.y, v.z, 0} * q

	float4 _qv =
		float4( + v.x* q.w + v.y*-q.z - v.z*-q.y,
				- v.x*-q.z + v.y* q.w + v.z*-q.x,
				+ v.x*-q.y - v.y*-q.x + v.z* q.w,
				- v.x*-q.x - v.y*-q.y - v.z*-q.z );
	
	float3 r =
		float3( q.w*_qv.x + q.x*_qv.w + q.y*_qv.z - q.z*_qv.y,
				q.w*_qv.y - q.x*_qv.z + q.y*_qv.w + q.z*_qv.x,
				q.w*_qv.z + q.x*_qv.y - q.y*_qv.x + q.z*_qv.w );

	return r;
}

float3x3 quatToMatrix(float4 q)
{
#ifdef USE_FAST_QUAT_2_MATRIX
	// add: 12 = 9 + 3
	// mul: 16 = 12 + 4 
	// div:	 1

	// Quat to matrix conversion, see "Advanced Rendering Techniques" pp 363-364
	float s, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;
#ifdef NORMALIZE_FAST_QUAT_2_MATRIX
	s = 2.0f / dot(q, q);
#else
	s = 2.0f;
#endif

	xs = q.x * s;	ys = q.y * s;	zs = q.z * s;
	wx = q.w * xs;	wy = q.w * ys;	wz = q.w * zs;
	xx = q.x * xs;	xy = q.x * ys;	xz = q.x * zs;
	yy = q.y * ys;	yz = q.y * zs;	zz = q.z * zs;
	
	float3x3 r =
		float3x3(
			1.f - (yy + zz),	xy + wz,			xz - wy,
			xy - wz,			1.f - (xx + zz),	yz + wx,
			xz + wy,			yz - wx,			1.f - (xx + yy) );
			
#else
	// add: 15 = 6 + 9
	// mul: 16 = 12 + 4
	// div:	 1

	float dxy = q.x * q.y * 2.0f;
	float dxz = q.x * q.z * 2.0f;
	float dyz = q.y * q.z * 2.0f;
	float dwx = q.w * q.x * 2.0f;
	float dwy = q.w * q.y * 2.0f;
	float dwz = q.w * q.z * 2.0f;
	
	float x2 = q.x * q.x;
	float y2 = q.y * q.y;
	float z2 = q.z * q.z;
	float w2 = q.w * q.w;
	
	float3x3 r =
		float3x3(
			w2+x2-y2-z2,	dxy+dwz,		dxz-dwy,
			dxy-dwz,		w2-x2+y2-z2,	dyz+dwz,
			dxz-dwy,		dyz-dwx,		w2-x2-y2+z2 );				
#endif
	return r;
}
///////////////// /////////////////          ///////////////// /////////////////