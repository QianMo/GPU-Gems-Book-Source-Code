!!ARBfp1.0

OPTION	ARB_precision_hint_nicest;

# tc0 : volume texture coordinate in [0, 1]^3
# tex1: Hessian diagonal
# tex2: Hessian off-diagonal

# ******* attributes *********

# framebuffer coordinate
ATTRIB unit_tc = fragment.texcoord[0]; 

# ******* parameters *********

PARAM const      = {0, 0.5, 1, 2};
PARAM delta_x    = {1, 0, 0, 0};

# ******** temporaries *********

TEMP normal;
TEMP H_mixed;
TEMP H_diag;
     
TEMP P_r0, P_r1, P_r2;
TEMP H_r0, H_r1, H_r2;

ALIAS A_r0 = H_mixed;
ALIAS A_r1 = P_r0;
ALIAS A_r2 = P_r1;

ALIAS B_r0 = A_r0;
ALIAS B_r1 = A_r1;
ALIAS B_r2 = A_r2;

ALIAS kappa = H_diag;

# ******** program *********

TEX normal, unit_tc, texture[0], 2D;
TEX H_diag, unit_tc, texture[1], 2D;
TEX H_mixed, unit_tc, texture[2], 2D;

# normalize gradient
DP3 normal.w, normal, normal;
RSQ normal.w, normal.w;
MUL normal.xyz, normal, normal.w;

# P = 1 - n * n'
MAD P_r0.xyz, -normal, normal.x, delta_x.rgba;
MAD P_r1.xyz, -normal, normal.y, delta_x.brga;
MAD P_r2.xyz, -normal, normal.z, delta_x.gbra;

# H = - Hessian / |grad|
MUL H_r0.x, -normal.w, H_diag.x;
MUL H_r1.y, -normal.w, H_diag.y;
MUL H_r2.z, -normal.w, H_diag.z;
    
MUL H_r0.yz, -normal.w, H_mixed.zxyw;
MUL H_r1.xz, -normal.w, H_mixed.xyzw;
MUL H_r2.xy, -normal.w, H_mixed.yzxw;

# A = - P * H
DP3 A_r0.x, P_r0, H_r0;
DP3 A_r0.y, P_r0, H_r1;
DP3 A_r0.z, P_r0, H_r2;
    
DP3 A_r1.x, P_r1, H_r0;
DP3 A_r1.y, P_r1, H_r1;
DP3 A_r1.z, P_r1, H_r2;
    
DP3 A_r2.x, P_r2, H_r0;
DP3 A_r2.y, P_r2, H_r1;
DP3 A_r2.z, P_r2, H_r2;

# kappa.w = trace(A)
ADD kappa.w, A_r0.x, A_r1.y;
ADD kappa.w, kappa.w, A_r2.z;

# B = A + n*n'
MAD B_r0.xyz, normal, normal.x, A_r0;
MAD B_r1.xyz, normal, normal.y, A_r1;
MAD B_r2.xyz, normal, normal.z, A_r2;

# kappa.x = det(B)
MUL kappa.xyz, B_r1.zxyw, B_r2.yzxw;
MAD kappa.xyz, B_r1.yzxw, B_r2.zxyw, -kappa;
DP3 kappa.x, B_r0, kappa;

#MOV result.color.r, kappa.x;
#MOV result.color.g, kappa.w;

# kappa = -trace(A)/2 +- sqrt((trace(A)/2)^2 - det(B))
MUL kappa.y, kappa.w, const.y;
MAD kappa.z, kappa.y, kappa.y, -kappa.x;
RSQ kappa.w, kappa.z;
MAD result.color.r, kappa.z, -kappa.w, -kappa.y;
MAD result.color.g, kappa.z,  kappa.w, -kappa.y;
    
END
