!!ARBfp1.0

# env0: {1-1/tileres, 0.5/tileres, ?, ?}
# env1: texres
# env2: 1/texres
# tex0: volume texture
# tex1: position texture
# tex3: offset/weight value texture
# tex4: offset/weight derivative texture
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
PARAM delta_x    = {1, 0, 0, 0};

# ******** temporaries *********

# texcoord for solid- and offset-texture
TEMP pos_obj, pos_ofs;

# offsets and weight
ALIAS y_ofs = texres_inv;
TEMP x_ofs, z_ofs;

# sample texture coordinates
TEMP lun_tc, luf_tc;
TEMP run_tc, ruf_tc;
TEMP lmn_tc, lmf_tc;
TEMP rmn_tc, rmf_tc;
TEMP lln_tc, llf_tc;
TEMP rln_tc, rlf_tc;

# sample values
TEMP l_val, m_val, u_val;

# ******** program *********

# read fragment object position
TEX pos_obj.xyz, unit_tc, texture[1], 2D;

# scale and shift ofs_texcoord
MAD pos_ofs.xyz, texres, pos_obj, -const.y;

# shrink to edge sampling (h/2 + (1-h)*x)
FRC pos_ofs.xyz, pos_ofs;
MAD pos_ofs.xyz, pos_ofs, tileres.r, tileres.g;

# read offsets and weight (indirection)
TEX x_ofs.xyz, pos_ofs.x, texture[3], 1D;
TEX z_ofs.xyz, pos_ofs.z, texture[3], 1D;

# setup texture coordinate
# left/right upper/lower near/far
MUL pos_ofs.xyz, delta_x.rgba, texres_inv;
MAD lmf_tc.xyz, pos_ofs, -x_ofs.x, pos_obj;
MAD rmf_tc.xyz, pos_ofs,  x_ofs.y, pos_obj;
              
MUL pos_ofs.xyz, delta_x.gbra, texres_inv;
MAD lmn_tc.xyz, pos_ofs, -z_ofs.x, lmf_tc;
MAD rmn_tc.xyz, pos_ofs, -z_ofs.x, rmf_tc;
MAD lmf_tc.xyz, pos_ofs,  z_ofs.y, lmf_tc;
MAD rmf_tc.xyz, pos_ofs,  z_ofs.y, rmf_tc;

#MUL pos_ofs.xyz, delta_x.brga, texres_inv;
MAD lln_tc.xyz, delta_x.brga, -y_ofs.x, lmn_tc;
MAD rln_tc.xyz, delta_x.brga, -y_ofs.x, rmn_tc;
MAD llf_tc.xyz, delta_x.brga, -y_ofs.x, lmf_tc;
MAD rlf_tc.xyz, delta_x.brga, -y_ofs.x, rmf_tc;
MAD lun_tc.xyz, delta_x.brga,  y_ofs.x, lmn_tc;
MAD run_tc.xyz, delta_x.brga,  y_ofs.x, rmn_tc;
MAD luf_tc.xyz, delta_x.brga,  y_ofs.x, lmf_tc;
MAD ruf_tc.xyz, delta_x.brga,  y_ofs.x, rmf_tc;

# read values (2 indirections)
TEX l_val.x, lln_tc, texture[0], 3D;
TEX l_val.y, rln_tc, texture[0], 3D;
TEX l_val.z, llf_tc, texture[0], 3D;
TEX l_val.w, rlf_tc, texture[0], 3D;

TEX m_val.x, lmn_tc, texture[0], 3D;
TEX m_val.y, rmn_tc, texture[0], 3D;
TEX m_val.z, lmf_tc, texture[0], 3D;
TEX m_val.w, rmf_tc, texture[0], 3D;

TEX u_val.x, lun_tc, texture[0], 3D;
TEX u_val.y, run_tc, texture[0], 3D;
TEX u_val.z, luf_tc, texture[0], 3D;
TEX u_val.w, ruf_tc, texture[0], 3D;

# output derivative in x direction
LRP l_val.zw, z_ofs.z, l_val.yzxy, l_val;    
LRP m_val.zw, z_ofs.z, m_val.yzxy, m_val;    
LRP u_val.zw, z_ofs.z, u_val.yzxy, u_val;

LRP l_val.x, x_ofs.z, l_val.z, l_val.w;     
LRP l_val.y, x_ofs.z, m_val.z, m_val.w;     
LRP l_val.z, x_ofs.z, u_val.z, u_val.w;

DP3 result.color, l_val, {1,-2,1,0};

END
