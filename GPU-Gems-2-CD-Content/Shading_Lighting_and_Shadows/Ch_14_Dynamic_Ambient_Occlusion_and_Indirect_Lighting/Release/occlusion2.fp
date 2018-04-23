!!ARBfp1.0
OPTION NV_fragment_program2;
# cgc version 1.3.0001, build date Sep 28 2004 16:01:04
# command line args: -profile fp40 -DPASS=2
# source file: c:\3d\DynamicAO\occlusion.cg
#vendor NVIDIA Corporation
#version 1.0.02
#profile fp40
#program main
#semantic main.lastResultMap : TEXUNIT0
#semantic main.elementPositionMap : TEXUNIT1
#semantic main.elementNormalMap : TEXUNIT2
#semantic main.indexMap : TEXUNIT3
#semantic passNumber
#var float4 position : $vin.WPOS : WPOS : 0 : 1
#var float4 normOffset :  :  : 1 : 0
#var samplerRECT lastResultMap : TEXUNIT0 : texunit 0 : 2 : 1
#var samplerRECT elementPositionMap : TEXUNIT1 : texunit 1 : 3 : 1
#var samplerRECT elementNormalMap : TEXUNIT2 : texunit 2 : 4 : 1
#var samplerRECT indexMap : TEXUNIT3 : texunit 3 : 5 : 1
#var float passNumber :  :  : -1 : 0
#var float4 main : $vout.COL : COL : -1 : 1
#const c[0] = 0 0.5 255 1
#const c[1] = 1e-016 4 3 0.6
#const c[2] = 0.4
PARAM c[3] = { { 0, 0.5, 255, 1 },
		{ 1e-016, 4, 3, 0.60000002 },
		{ 0.40000001 } };
TEMP R0;
TEMP R1;
TEMP R2;
TEMP R3;
TEMP R4;
TEMP R5;
TEMP R6;
TEMP H0;
TEMP RC;
TEMP HC;
TEX   R1.xyz, fragment.position, texture[1], RECT;
MOVR  R0.x, c[0];
TEX   R4.xyz, fragment.position, texture[2], RECT;
MOVR  R2.xy, c[0].y;
MOVR  R5.xy, fragment.position;
LOOP c[0].zxww;
SNERC HC.x, R2, c[0];
BRK   (EQ.x);
LOOP c[0].zxww;
SNERC HC.x, R2, c[0];
BRK   (EQ.x);
MOVR  R3.xy, R2;
TEX   R2.xyz, R3, texture[1], RECT;
ADDR  R0.yzw, R2.xxyz, -R1.xxyz;
DP3R  R2.x, R0.yzww, R0.yzww;
ADDR  R5.z, R2.x, c[1].x;
RSQR  R4.w, R5.z;
TEX   R2, R3, texture[2], RECT;
MULR  R5.w, -R2, c[1].y;
MULR  R0.yzw, R4.w, R0;
SLTR  H0.w, R5.z, R5;
DP3R_SAT R2.x, R2, R0.yzww;
RCPR  R1.w, R5.z;
TEX   R3, R3, texture[3], RECT;
MOVRC HC.x, H0.w;
MOVRC HC.w, H0;
MOVR  R2.w(GT.x), c[0].x;
MOVR  R3.xy(GT.w), R3.zwzw;
DP3R  R2.y, R4, R0.yzww;
MADR  R2.w, |R2|, R1, c[0];
RSQR  R0.y, R2.w;
TEX   R0.w, R3, texture[0], RECT;
MADR  R2.x, -R0.y, R2, R2;
MULR_SAT R2.w, R2.y, c[1].z;
MULR  R1.w, R2.x, R2;
MOVR  R2.xy, R3;
MADR  R0.x, R1.w, R0.w, R0;
ENDLOOP;
ENDLOOP;
ADDR_SAT R0.w, -R0.x, c[0];
TEX   R1.x, R5, texture[0], RECT;
MULR  R0.x, R1, c[2];
MADR  result.color, R0.w, c[1].w, R0.x;
END
# 43 instructions, 7 R-regs, 1 H-regs
# 43 inst, (5 mov, 7 tex, 3 complex, 28 math)
# non-mov args: 0 20 8 5 5
# mov args:     0 1 4 0 0
