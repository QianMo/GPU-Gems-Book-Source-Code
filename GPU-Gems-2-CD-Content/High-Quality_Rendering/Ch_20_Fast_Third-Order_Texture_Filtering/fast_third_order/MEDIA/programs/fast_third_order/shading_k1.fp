!!ARBfp1.0

# tex0: position texture
# tex1: gradient texture
# tex2: curvature texture
# tc0 : window position ([0,1] x [1,0] x ? x [1])
# env3: {?, ?, 0.5/volres+eps, eps}
# env4: curvature bias

# ******* attributes *********

# framebuffer coordinate
ATTRIB unit_tc = fragment.texcoord[0]; 

# ******* parameters *********

PARAM edge       = program.env[3];
PARAM curv_bias  = program.env[4];
PARAM light_pos  = state.light[0].position;
PARAM mv_mat[]   = { state.matrix.modelview.row[0..2] };
PARAM const      = {0, 0.5, 1, 2};
PARAM delta_x    = {1, 0, 0, 32};
ALIAS spec_exp   = delta_x;

# ******** temporaries *********

TEMP pos_obj, pos_eye;
TEMP normal, normal2;
TEMP light, light_r;
TEMP view;
TEMP color;

ALIAS curv = normal;
ALIAS edge_test = light;

# ******** program *********

# read fragment object position, gradient and curvature
TEX pos_eye, unit_tc, texture[0], 2D;
TEX curv, unit_tc, texture[2], 2D;
MOV normal.w, curv.x;
TEX normal.xyz, unit_tc, texture[1], 2D;

# move to pos_obj (pos_eye contains depth)
MOV pos_obj.xyz, pos_eye;
MOV pos_obj.w, const.z;

# set normals outside of cube
ADD edge_test.z, -edge.y, edge.z;
DP3 edge_test.w, const.z, edge;

ADD edge_test.xyz, edge_test.z, pos_obj;
CMP normal, edge_test.x, -delta_x.xyzw, normal;
CMP normal, edge_test.y, -delta_x.zxyw, normal;
CMP normal, edge_test.z, -delta_x.yzxw, normal;

ADD edge_test.xyz, edge_test.w, -pos_obj;
CMP normal, edge_test.x,  delta_x.xyzw, normal;
CMP normal, edge_test.y,  delta_x.zxyw, normal;
CMP normal, edge_test.z,  delta_x.yzxw, normal;

# transform position to eye space
DP4 pos_eye.x, mv_mat[0], pos_obj;
DP4 pos_eye.y, mv_mat[1], pos_obj;
DP4 pos_eye.z, mv_mat[2], pos_obj;

# transform gradient to eye space
DP3 normal2.x, mv_mat[0], normal;
DP3 normal2.y, mv_mat[1], normal;
DP3 normal2.z, mv_mat[2], normal;

# set normal at near plane
ADD view.w, pos_eye.w, -edge.w;
CMP normal2, view.w, delta_x.yzxw, normal2;

# normalize gradient
DP3 normal2.w, normal2, normal2;
RSQ normal2.w, normal2.w;
MUL normal.xyz, normal2, normal2.w;
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

# ambient from curvature
MAD normal.w, normal.w, curv_bias.x, curv_bias.z;
TEX color.xyz, normal.w, texture[3], 1D;

# diffuse color
DP3 color.w, normal, light;
MUL color.xyz, color, color.w;

# specular
DP3_SAT color.w, view, light_r;
POW color.w, color.w, spec_exp.w;
ADD color.xyz, color, color.w;
CMP result.color, pos_eye.w, fragment.color, color;

END
