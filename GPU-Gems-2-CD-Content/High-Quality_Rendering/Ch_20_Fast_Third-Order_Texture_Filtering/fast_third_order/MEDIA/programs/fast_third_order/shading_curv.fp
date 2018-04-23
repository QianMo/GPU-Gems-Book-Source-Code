!!ARBfp1.0

OPTION	ARB_precision_hint_nicest;

PARAM	const = { 0.0, 0.5, 1.0, 2.0 };
PARAM	diffuse_vec = program.env[ 0 ];
PARAM	specular_vec = program.env[ 2 ];
PARAM	diffuse_color = program.env[ 3 ];
PARAM	ambient_color = program.env[ 4 ];

PARAM	const_range_scale = {1,1,1,0};
PARAM	const_range_bias  = {0.5,0.5,0.5,0};

TEMP	gradient;
TEMP	diffuse, specular;
TEMP	screenpos;

ALIAS	intersect = gradient;
ALIAS	shaded = screenpos;
ALIAS	normal = gradient;

MUL		screenpos, fragment.position, {0.001953125,0.001953125,0.001953125,0.001953125};
TEX		gradient, screenpos, texture[ 0 ], 2D;
TEMP	curv_kappa;
TEX		curv_kappa, screenpos, texture[ 1 ], 2D;
DP3		shaded.x, gradient, gradient;
RSQ		shaded.x, shaded.x;
MUL		normal.rgb, gradient, shaded.x;
DP3		diffuse, normal, diffuse_vec;
MUL		diffuse, diffuse, diffuse_color;
ADD		diffuse, diffuse, ambient_color;
DP3		specular, normal, specular_vec;
MUL		specular, specular, specular;
MUL		specular, specular, specular;
MUL		specular, specular, specular;
MUL		specular, specular, specular;
MOV		curv_kappa.zw, const.x;
MOV		curv_kappa.z, curv_kappa.x;
MAD		curv_kappa.z, curv_kappa, const_range_scale, const_range_bias;
TEX		shaded.rgb, curv_kappa.z, texture[ 3 ], 1D;
MAD		shaded.rgb, shaded, diffuse, specular;
MOV		shaded.a, const.z;
MUL		result.color, shaded, intersect.a;

END
