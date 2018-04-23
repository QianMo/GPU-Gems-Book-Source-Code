!!ARBfp1.0

# tex1: depth texture
# tc0 : window position ([0,1] x [0,1] x ? x [1])

# ******* attributes *********

# framebuffer coordinate
ATTRIB unit_tc = fragment.texcoord[0]; 

# ******* parameters *********

# volume texture resolution parameters 
PARAM mvp_inv[]  = { state.matrix.mvp.inverse };
PARAM const      = {0, 0.999, 1, 2};

# ******** temporaries *********

# fragment position in window and object coordinates
TEMP pos_win, temp;

# ******** program *********

# read fragment window position [0,1]^3
TEX pos_win.z, unit_tc, texture[1], 2D;

# kill fragment if on far plane
ADD pos_win.w, const.y, -pos_win.z;
KIL pos_win.w;

# write depth
MOV result.color.w, pos_win.z;

# init window position
MOV pos_win.xyw, unit_tc;

# bias to fragment device coordinates [-1,1]^3
MAD pos_win.xyz, pos_win, const.a, -const.b;

# compute homogenous clipping coordinate 
DP4 temp.w, mvp_inv[3], pos_win;
RCP temp.w, temp.w;

# unnormalize to clipping coordinates
MUL pos_win, pos_win, temp.w;

# transform to object coordinates
DP4 result.color.x, mvp_inv[0], pos_win;
DP4 result.color.y, mvp_inv[1], pos_win;
DP4 result.color.z, mvp_inv[2], pos_win;

END
