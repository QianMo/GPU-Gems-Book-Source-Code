!!ARBfp1.0
OPTION NV_fragment_program2;
# cgc version 1.3.0001, build date Sep 28 2004 16:01:04
# command line args: -profile fp40 -DPASS=1
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
#var samplerRECT lastResultMap : TEXUNIT0 :  : 2 : 0
#var samplerRECT elementPositionMap : TEXUNIT1 : texunit 1 : 3 : 1
#var samplerRECT elementNormalMap : TEXUNIT2 : texunit 2 : 4 : 1
#var samplerRECT indexMap : TEXUNIT3 : texunit 3 : 5 : 1
#var float passNumber :  :  : -1 : 0
#var float4 main : $vout.COL : COL : -1 : 1
#const c[0] = 0 0.5 255 1
#const c[1] = 1e-016 4 3
PARAM c[2] = { { 0, 0.5, 255, 1 },
		{ 1e-016, 4, 3 } };
TEMP R0;
TEMP R1;
TEMP R2;
TEMP R3;
TEMP R4;
TEMP R5;
TEMP H0;
TEMP RC;
TEMP HC;
TEX   R3.xyz, fragment.position, texture[1], RECT;
MOVR  R0.x, c[0];
TEX   R4.xyz, fragment.position, texture[2], RECT;
MOVR  R2.xy, c[0].y;
LOOP c[0].zxww;
SNERC HC.x, R2, c[0];
BRK   (EQ.x);
LOOP c[0].zxww;
SNERC HC.x, R2, c[0];
BRK   (EQ.x);
TEX   R1.xyz, R2, texture[1], RECT;
ADDR  R0.yzw, R1.xxyz, -R3.xxyz;
TEX   R1, R2, texture[2], RECT;
DP3R  R4.w, R0.yzww, R0.yzww;
MOVR  R3.w, R1;
ADDR  R4.w, R4, c[1].x;
RSQR  R5.x, R4.w;
MULR  R1.w, -R1, c[1].y;
MULR  R0.yzw, R5.x, R0;
SLTR  H0.w, R4, R1;
DP3R_SAT R1.x, R1, R0.yzww;
RCPR  R1.w, R4.w;
MOVRC HC.w, H0;
DP3R  R0.y, R4, R0.yzww;
MOVR  R3.w(GT), c[0].x;
MADR  R0.w, |R3|, R1, c[0];
RSQR  R0.z, R0.w;
TEX   R2, R2, texture[3], RECT;
MOVRC HC.w, H0;
MADR  R1.x, -R0.z, R1, R1;
MULR_SAT R0.w, R0.y, c[1].z;
MOVR  R2.xy(GT.w), R2.zwzw;
MADR  R0.x, R1, R0.w, R0;
ENDLOOP;
ENDLOOP;
ADDR_SAT result.color, -R0.x, c[0].w;
END
# 36 instructions, 6 R-regs, 1 H-regs
# 36 inst, (3 mov, 5 tex, 3 complex, 25 math)
# non-mov args: 0 17 6 5 5
# mov args:     0 2 1 0 0
