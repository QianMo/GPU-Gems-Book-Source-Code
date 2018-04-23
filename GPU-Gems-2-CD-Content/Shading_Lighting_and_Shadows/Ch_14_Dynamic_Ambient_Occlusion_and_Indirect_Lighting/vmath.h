typedef struct float2 {
    float x, y;
} float2;

typedef struct float3 {
    float x, y, z;
} float3;

typedef struct float4 {
    float x, y, z, w;
} float4;

#define vadd(d, s1) { \
	(d).x += (s1).x; \
	(d).y += (s1).y; \
	(d).z += (s1).z; \
}

#define vsub(d, s1, s2) { \
	(d).x = (s1).x - (s2).x; \
	(d).y = (s1).y - (s2).y; \
	(d).z = (s1).z - (s2).z; \
}

#define vadd2(d, s1, s2) { \
	(d).x = (s1).x + (s2).x; \
	(d).y = (s1).y + (s2).y; \
	(d).z = (s1).z + (s2).z; \
}

#define vdiv(d, s1) { \
	float tt = 1.0f/(s1); \
	(d).x *= tt; \
	(d).y *= tt; \
	(d).z *= tt; \
}

#define vmuls(d, s1) { \
	(d).x *= (s1); \
	(d).y *= (s1); \
	(d).z *= (s1); \
}

#define vmul(d, s1) { \
	(d).x *= (s1).x; \
	(d).y *= (s1).y; \
	(d).z *= (s1).z; \
}

#define vmulacc(d, f, s1) { \
	(d).x += (f).x*(s1); \
	(d).y += (f).y*(s1); \
	(d).z += (f).z*(s1); \
}

#define dot(a, b) ((a).x*(b).x + (a).y*(b).y + (a).z*(b).z)

#define veq(a, b) ((a).x==(b).x && (a).y==(b).y && (a).z==(b).z)

#define cross(d, v1, v2) { \
    (d).x = (v1).y*(v2).z - (v1).z*(v2).y; \
    (d).y = (v1).z*(v2).x - (v1).x*(v2).z; \
    (d).z = (v1).x*(v2).y - (v1).y*(v2).x; \
}