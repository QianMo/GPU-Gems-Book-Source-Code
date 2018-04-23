!!FP1.0
# cgc version 1.3.0001, build date Sep 28 2004 16:01:04
# command line args: -profile fp30
# source file: c:\3d\DynamicAO\surface.cg
#vendor NVIDIA Corporation
#version 1.0.02
#profile fp30
#program main
#semantic lightDirection : LIGHTDIRECTION
#var float4 color : $vin.TEX0 : TEX0 : 0 : 1
#var float4 lightDirection : LIGHTDIRECTION :  : -1 : 0
#var float4 main : $vout.COL : COL : -1 : 1
MOVR  o[COLR], f[TEX0];
END
# 1 instructions, 0 R-regs, 0 H-regs
# 1 inst, (1 mov, 0 tex, 0 complex, 0 math)
# non-mov args: 0 0 0 0 0
# mov args:     0 0 0 0 1
