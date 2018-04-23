!!ARBfp1.0

# env1: texres
# tex1: position texture
# tex2: gradient texture
# tex3: curvature texture
# tc0 : window position ([0,1] x [1,0] x ? x [1])

# ******* attributes *********

# framebuffer coordinate
ATTRIB unit_tc = fragment.texcoord[0]; 

# ******* parameters *********

PARAM texres     = program.env[1];
PARAM light_pos  = state.light[0].position;
PARAM mv_mat[]   = { state.matrix.modelview.row[0..2] };
PARAM const      = {0, 0.5, 1, 2};
PARAM delta_x    = {1, 0, 0, 32};
PARAM border     = {0.001, 0.998, 0, 0};
ALIAS spec_exp   = delta_x;

# ******** temporaries *********

# texcoord for solid-texture and eye-position
TEMP pos_obj, pos_eye;
TEMP normal, normal2;
TEMP light, light_r;
TEMP view;
TEMP phong;

# ******** program *********

# read fragment object position
TEX pos_obj.xyz, unit_tc, texture[1], 2D;
MOV pos_obj.w, const.z;

# transform position to eye space
DP4 pos_eye.x, mv_mat[0], pos_obj;
DP4 pos_eye.y, mv_mat[1], pos_obj;
DP4 pos_eye.z, mv_mat[2], pos_obj;

# read gradient
TEX normal.xyz, unit_tc, texture[2], 2D;

# set normals outside of cube
ADD pos_obj.xyz, pos_obj, -border.x;
CMP normal.xyz, pos_obj.x, -delta_x.xyzw, normal;
CMP normal.xyz, pos_obj.y, -delta_x.zxyw, normal;
CMP normal.xyz, pos_obj.z, -delta_x.yzxw, normal;

ADD pos_obj.xyz, border.y, -pos_obj;
CMP normal.xyz, pos_obj.x,  delta_x.xyzw, normal;
CMP normal.xyz, pos_obj.y,  delta_x.zxyw, normal;
CMP normal.xyz, pos_obj.z,  delta_x.yzxw, normal;

# transform gradient to eye space
DP3 normal2.x, mv_mat[0], normal;
DP3 normal2.y, mv_mat[1], normal;
DP3 normal2.z, mv_mat[2], normal;

# normalize gradient
DP3 normal.w, normal2, normal2;
RSQ normal.w, normal.w;
MUL normal.xyz, normal2, normal.w;
MUL normal2.xyz, normal, const.w;

# compute and normalize light vector
ADD light.xyz, light_pos, -pos_eye;
DP3 light.w, light, light;
RSQ light.w, light.w;
MUL light.xyz, light, light.w;

# compute light reflect vector
DP3 light_r.w, light, normal;
MAD light_r.xyz, normal2, light_r.w, -light;

# compute and normalize view direction
DP3 view.w, pos_eye, pos_eye;
RSQ view.w, view.w;
MUL view.xyz, -pos_eye, view.w;

# diffuse color
TEX phong.xy, unit_tc, texture[3], 2D;
ADD phong.w, phong.x, phong.y;
TEX phong.xyz, phong.w, texture[6], 1D;
DP3 phong.w, normal, light;
MUL phong.xyz, phong, phong.w;

# specular
DP3_SAT phong.w, view, light_r;
POW phong.w, phong.w, spec_exp.w;
ADD result.color, phong, phong.w;

END
