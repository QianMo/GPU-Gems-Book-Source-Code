!!ARBfp1.0

# env0: {1-1/tileres, 0.5/tileres, ?, ?}
# env1: texres
# env2: 1/texres
# tex0: volume texture
# tex1: position texture
# tex4: offset/weight value texture
# tex5: offset/weight derivative texture
# tc0 : window position ([0,1] x [1,0] x ? x [1])

# ******* attributes *********

# framebuffer coordinate
ATTRIB unit_tc = fragment.texcoord[0]; 

# ******* parameters *********

# volume texture resolution parameters 
PARAM tileres    = program.env[0];
PARAM texres     = program.env[1];
PARAM texres_inv = program.env[2];
PARAM const      = {0, 0.5, 1, 2};
PARAM delta_x    = {1, 0, 0, 8};

# ******** temporaries *********

# texcoord for solid- and offset-texture
TEMP pos_obj, pos_ofs;

# offsets and weight
TEMP x_ofs, y_ofs, z_ofs;

# sample texture coordinates
TEMP lun_tc, luf_tc;
TEMP run_tc, ruf_tc;
TEMP lln_tc, llf_tc;
TEMP rln_tc, rlf_tc;

# sample values
TEMP n_val, f_val;

# ******** program *********

# read fragment object position
TEX pos_obj.xyz, unit_tc, texture[1], 2D;

# scale and shift ofs_texcoord
MAD pos_ofs.xyz, texres, pos_obj, -const.y;

# shrink to edge sampling (h/2 + (1-h)*x)
FRC pos_ofs.xyz, pos_ofs;
MAD pos_ofs.xyz, pos_ofs, tileres.r, tileres.g;

# read offsets and weight (indirection)
TEX x_ofs.xyz, pos_ofs.x, texture[4], 1D;
TEX y_ofs.xyz, pos_ofs.y, texture[5], 1D;
TEX z_ofs.xyz, pos_ofs.z, texture[4], 1D;

# setup texture coordinate
# left/right upper/lower near/far
MUL pos_ofs.xyz, delta_x.rgba, texres_inv;
MAD llf_tc.xyz, pos_ofs, -x_ofs.x, pos_obj;
MAD rlf_tc.xyz, pos_ofs,  x_ofs.y, pos_obj;
              
MUL pos_ofs.xyz, delta_x.brga, texres_inv;
MAD luf_tc.xyz, pos_ofs, -y_ofs.x, llf_tc;
MAD ruf_tc.xyz, pos_ofs, -y_ofs.x, rlf_tc;
MAD llf_tc.xyz, pos_ofs,  y_ofs.y, llf_tc;
MAD rlf_tc.xyz, pos_ofs,  y_ofs.y, rlf_tc;
              
MUL pos_ofs.xyz, delta_x.gbra, texres_inv;
MAD lun_tc.xyz, pos_ofs, -z_ofs.x, luf_tc;
MAD run_tc.xyz, pos_ofs, -z_ofs.x, ruf_tc;
MAD lln_tc.xyz, pos_ofs, -z_ofs.x, llf_tc;
MAD rln_tc.xyz, pos_ofs, -z_ofs.x, rlf_tc;
MAD luf_tc.xyz, pos_ofs,  z_ofs.y, luf_tc;
MAD ruf_tc.xyz, pos_ofs,  z_ofs.y, ruf_tc;
MAD llf_tc.xyz, pos_ofs,  z_ofs.y, llf_tc;
MAD rlf_tc.xyz, pos_ofs,  z_ofs.y, rlf_tc;

# read values (2 indirections)
TEX n_val.x, lun_tc, texture[0], 3D;
TEX n_val.y, run_tc, texture[0], 3D;
TEX n_val.z, lln_tc, texture[0], 3D;
TEX n_val.w, rln_tc, texture[0], 3D;
TEX f_val.x, luf_tc, texture[0], 3D;
TEX f_val.y, ruf_tc, texture[0], 3D;
TEX f_val.z, llf_tc, texture[0], 3D;
TEX f_val.w, rlf_tc, texture[0], 3D;

# output derivative in y direction
LRP f_val, z_ofs.z, n_val, f_val;
ADD f_val.zw, -f_val.yzxy, f_val;
MUL f_val.zw, y_ofs.z, f_val;
#LRP result.color, x_ofs.z, f_val.z, f_val.w;
LRP pos_obj, x_ofs.z, f_val.z, f_val.w;
MAD result.color, pos_obj, delta_x.w, const.y;

END
