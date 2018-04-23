!!ARBfp1.0

OPTION	ARB_precision_hint_nicest;
OPTION	ATI_draw_buffers;

PARAM	const = { 0.0, 0.5, 1.0, 2.0 };

TEMP	screenpos;
TEMP	tex_gradient;
TEMP	tex_H_mixed;
TEMP	tex_H_diag;

TEMP	mat_P_row0;
TEMP	mat_P_row1;
TEMP	mat_P_row2;

TEMP	mat_H_row0;
TEMP	mat_H_row1;
TEMP	mat_H_row2;

TEMP	mat_PH_row0;
TEMP	mat_PH_row1;
TEMP	mat_PH_row2;

ALIAS	intersect = screenpos;
ALIAS	normal = tex_gradient;
ALIAS	length = screenpos;

ALIAS	mat_G_row0 = mat_H_row0;
ALIAS	mat_G_row1 = mat_H_row1;
ALIAS	mat_G_row2 = mat_H_row2;

ALIAS	trace = tex_H_diag;
ALIAS	frobnorm = tex_H_mixed;

ALIAS	curv_kappa = trace;

MUL		screenpos, fragment.position, {0.001953125,0.001953125,0.001953125,0.001953125};
TEX		tex_gradient, screenpos, texture[ 0 ], 2D;
TEX		tex_H_mixed, screenpos, texture[ 1 ], 2D;
TEX		tex_H_diag, screenpos, texture[ 2 ], 2D;
MOV		intersect.a, tex_gradient;
DP3		length.x, tex_gradient, tex_gradient;
RSQ		length.x, length.x;
MUL		normal.rgb, tex_gradient, length.x;
MOV		normal.a, length.x;

MAD		mat_P_row0.xyz, -normal, normal.x, {1,0,0,0};
MAD		mat_P_row1.xyz, -normal, normal.y, {0,1,0,0};
MAD		mat_P_row2.xyz, -normal, normal.z, {0,0,1,0};

MUL		mat_H_row0.x, -normal.a, tex_H_diag.x;
MUL		mat_H_row1.y, -normal.a, tex_H_diag.y;
MUL		mat_H_row2.z, -normal.a, tex_H_diag.z;

MUL		mat_H_row0.yz, -normal.a, tex_H_mixed.xzyw;
MUL		mat_H_row1.xz, -normal.a, tex_H_mixed.zxxw;
MUL		mat_H_row2.xy, -normal.a, tex_H_mixed.yxxw;

DP3		mat_PH_row0.x, mat_P_row0, mat_H_row0;
DP3		mat_PH_row0.y, mat_P_row0, mat_H_row1;
DP3		mat_PH_row0.z, mat_P_row0, mat_H_row2;

DP3		mat_PH_row1.x, mat_P_row1, mat_H_row0;
DP3		mat_PH_row1.y, mat_P_row1, mat_H_row1;
DP3		mat_PH_row1.z, mat_P_row1, mat_H_row2;

DP3		mat_PH_row2.x, mat_P_row2, mat_H_row0;
DP3		mat_PH_row2.y, mat_P_row2, mat_H_row1;
DP3		mat_PH_row2.z, mat_P_row2, mat_H_row2;

DP3		mat_G_row0.x, mat_PH_row0, mat_P_row0;
DP3		mat_G_row0.y, mat_PH_row0, mat_P_row1;
DP3		mat_G_row0.z, mat_PH_row0, mat_P_row2;

DP3		mat_G_row1.x, mat_PH_row1, mat_P_row0;
DP3		mat_G_row1.y, mat_PH_row1, mat_P_row1;
DP3		mat_G_row1.z, mat_PH_row1, mat_P_row2;

DP3		mat_G_row2.x, mat_PH_row2, mat_P_row0;
DP3		mat_G_row2.y, mat_PH_row2, mat_P_row1;
DP3		mat_G_row2.z, mat_PH_row2, mat_P_row2;

ADD		trace.x, mat_G_row0.x, mat_G_row1.y;
ADD		trace.x, trace.x, mat_G_row2.z;

DP3		frobnorm.x, mat_G_row0, mat_G_row0;
DP3		frobnorm.y, mat_G_row1, mat_G_row1;
DP3		frobnorm.z, mat_G_row2, mat_G_row2;

DP3		frobnorm.x, frobnorm, {2,2,2,0};

MAD		trace.z, trace.x, -trace.x, frobnorm.x;
RSQ		trace.z, trace.z;
RCP		trace.z, trace.z;
MUL		trace.z, trace.z, const.y;

MAD		curv_kappa.y, const.y, trace.x, -trace.z;
MAD		curv_kappa.x, const.y, trace.x,  trace.z;

MOV		result.color.rg, curv_kappa;
MOV		result.color.a, intersect.a;

MOV		result.color[ 1 ].rgb, mat_G_row0;
MOV		result.color[ 1 ].a, intersect.a;
MOV		result.color[ 2 ].rgb, mat_G_row1;
MOV		result.color[ 2 ].a, intersect.a;
MOV		result.color[ 3 ].rgb, mat_G_row2;
MOV		result.color[ 3 ].a, intersect.a;

END
