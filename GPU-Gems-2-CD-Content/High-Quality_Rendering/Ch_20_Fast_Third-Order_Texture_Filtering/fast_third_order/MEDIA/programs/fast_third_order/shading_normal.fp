!!ARBfp1.0

# tex0: position texture
# tex1: gradient texture
# tex2: curvature texture
# tc0 : window position ([0,1] x [1,0] x ? x [1])
# env3: {?, ?, 0.5/volres+eps, eps}

# ******* attributes *********

# framebuffer coordinate
ATTRIB unit_tc = fragment.texcoord[0]; 

# ******* parameters *********

PARAM edge       = program.env[3];
PARAM const      = {0, 0.5, 1, 2};
PARAM delta_x    = {1, 0, 0, 0};

# ******** temporaries *********

# texcoord for solid-texture and eye-position
TEMP pos_obj, normal, edge_test;

# ******** program *********

# read fragment object position
TEX pos_obj, unit_tc, texture[0], 2D;

# read gradient
TEX normal.xyz, unit_tc, texture[1], 2D;

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

# set normal at near plane
ADD normal.w, pos_obj.w, -edge.w;
CMP normal.xyz, normal.w, state.matrix.modelview.row[2], normal;

# normalize gradient
DP3 normal.w, normal, normal;
RSQ normal.w, normal.w;
MUL normal.xyz, normal, normal.w;

# bias normal
MAD normal, normal, const.y, const.y;

CMP result.color, pos_obj.w, fragment.color, normal;

END
