!!ARBfp1.0

# env0: {1-1/lu_bias, 0.5/lu_bias, ?, ?}
# env1: vol_res
# env2: 1/vol_res
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
PARAM lu_bias     = program.env[0];
PARAM vol_res     = program.env[1];
PARAM vol_res_inv = program.env[2];
PARAM const       = {0, 0.5, 1, 2};
PARAM delta_x     = {1, 0, 0, 0};

# ******** temporaries *********

# texcoord for solid- and offset-texture
TEMP pos_obj, pos_ofs;

# offsets and weight
ALIAS x_ofs = vol_res_inv;
TEMP y_ofs, z_ofs;

# sample texture coordinates
TEMP lun_tc, luf_tc;
TEMP mun_tc, muf_tc;
TEMP run_tc, ruf_tc;
TEMP lln_tc, llf_tc;
TEMP mln_tc, mlf_tc;
TEMP rln_tc, rlf_tc;

# sample values
TEMP l_val, m_val, r_val;

# ******** program *********

# read fragment object position
TEX pos_obj.xyz, unit_tc, texture[1], 2D;

# scale and shift ofs_texcoord
MAD pos_ofs.xyz, vol_res, pos_obj, -const.y;

# shrink to edge sampling (h/2 + (1-h)*x)
FRC pos_ofs.xyz, pos_ofs;
MAD pos_ofs.xyz, pos_ofs, lu_bias.r, lu_bias.g;

# read offsets and weight (indirection)
TEX y_ofs.xyz, pos_ofs.y, texture[3], 1D;
TEX z_ofs.xyz, pos_ofs.z, texture[3], 1D;

# setup texture coordinate
# left/right upper/lower near/far
MUL pos_ofs.xyz, delta_x.brga, vol_res_inv;
MAD mlf_tc.xyz, pos_ofs, -y_ofs.x, pos_obj;
MAD muf_tc.xyz, pos_ofs,  y_ofs.y, pos_obj;
              
MUL pos_ofs.xyz, delta_x.gbra, vol_res_inv;
MAD mln_tc.xyz, pos_ofs, -z_ofs.x, mlf_tc;
MAD mun_tc.xyz, pos_ofs, -z_ofs.x, muf_tc;
MAD mlf_tc.xyz, pos_ofs,  z_ofs.y, mlf_tc;
MAD muf_tc.xyz, pos_ofs,  z_ofs.y, muf_tc;

#MUL pos_ofs.xyz, delta_x.rgba, vol_res_inv;
MAD lln_tc.xyz, delta_x.rgba, -x_ofs.x, mln_tc;
MAD lun_tc.xyz, delta_x.rgba, -x_ofs.x, mun_tc;
MAD llf_tc.xyz, delta_x.rgba, -x_ofs.x, mlf_tc;
MAD luf_tc.xyz, delta_x.rgba, -x_ofs.x, muf_tc;
MAD rln_tc.xyz, delta_x.rgba,  x_ofs.x, mln_tc;
MAD run_tc.xyz, delta_x.rgba,  x_ofs.x, mun_tc;
MAD rlf_tc.xyz, delta_x.rgba,  x_ofs.x, mlf_tc;
MAD ruf_tc.xyz, delta_x.rgba,  x_ofs.x, muf_tc;

# read values (2 indirections)
TEX l_val.x, lln_tc, texture[0], 3D;
TEX l_val.y, lun_tc, texture[0], 3D;
TEX l_val.z, llf_tc, texture[0], 3D;
TEX l_val.w, luf_tc, texture[0], 3D;

TEX m_val.x, mln_tc, texture[0], 3D;
TEX m_val.y, mun_tc, texture[0], 3D;
TEX m_val.z, mlf_tc, texture[0], 3D;
TEX m_val.w, muf_tc, texture[0], 3D;

TEX r_val.x, rln_tc, texture[0], 3D;
TEX r_val.y, run_tc, texture[0], 3D;
TEX r_val.z, rlf_tc, texture[0], 3D;
TEX r_val.w, ruf_tc, texture[0], 3D;

# output derivative in x direction
LRP l_val.zw, z_ofs.z, l_val.yzxy, l_val;    
LRP m_val.zw, z_ofs.z, m_val.yzxy, m_val;    
LRP r_val.zw, z_ofs.z, r_val.yzxy, r_val;

LRP l_val.x, y_ofs.z, l_val.z, l_val.w;     
LRP l_val.y, y_ofs.z, m_val.z, m_val.w;     
LRP l_val.z, y_ofs.z, r_val.z, r_val.w;

DP3 result.color, l_val, {1,-2,1,0};

END
