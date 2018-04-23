!!ARBfp1.0

# tex0: position texture
# tex1: gradient texture
# tex2: curvature texture
# tc0 : window position ([0,1] x [1,0] x ? x [1])

# ******* attributes *********

# framebuffer coordinate
ATTRIB unit_tc = fragment.texcoord[0]; 

# ******* parameters *********

PARAM const      = {0, 0.5, 1, 2};

# ******** temporaries *********

TEMP depth;

# ******** program *********

# read fragment depth
TEX depth.w, unit_tc, texture[0], 2D;
CMP result.color, depth.w, fragment.color, depth.w;

END
