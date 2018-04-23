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
ALIAS z_ofs = texres_inv;
TEMP x_ofs, y_ofs;

# sample texture coordinates
TEMP lun_tc, lum_tc, luf_tc;
TEMP run_tc, rum_tc, ruf_tc;
TEMP lln_tc, llm_tc, llf_tc;
TEMP rln_tc, rlm_tc, rlf_tc;

# sample values
TEMP n_val, m_val, f_val;

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
TEX y_ofs.xyz, pos_ofs.y, texture[3], 1D;

# setup texture coordinate
# left/right upper/lower near/far
MUL pos_ofs.xyz, delta_x.rgba, texres_inv;
MAD llm_tc.xyz, pos_ofs, -x_ofs.x, pos_obj;
MAD rlm_tc.xyz, pos_ofs,  x_ofs.y, pos_obj;
              
MUL pos_ofs.xyz, delta_x.brga, texres_inv;
MAD llm_tc.xyz, pos_ofs, -y_ofs.x, llm_tc;
MAD rlm_tc.xyz, pos_ofs, -y_ofs.x, rlm_tc;
MAD lum_tc.xyz, pos_ofs,  y_ofs.y, llm_tc;
MAD rum_tc.xyz, pos_ofs,  y_ofs.y, rlm_tc;

#MUL pos_ofs.xyz, delta_x.gbra, texres_inv;
MAD lln_tc.xyz, delta_x.gbra, -z_ofs.x, llm_tc;
MAD rln_tc.xyz, delta_x.gbra, -z_ofs.x, rlm_tc;
MAD lun_tc.xyz, delta_x.gbra, -z_ofs.x, lum_tc;
MAD run_tc.xyz, delta_x.gbra, -z_ofs.x, rum_tc;
MAD llf_tc.xyz, delta_x.gbra,  z_ofs.x, llm_tc;
MAD rlf_tc.xyz, delta_x.gbra,  z_ofs.x, rlm_tc;
MAD luf_tc.xyz, delta_x.gbra,  z_ofs.x, lum_tc;
MAD ruf_tc.xyz, delta_x.gbra,  z_ofs.x, rum_tc;

# read values (2 indirections)
TEX n_val.x, lln_tc, texture[0], 3D;
TEX n_val.y, rln_tc, texture[0], 3D;
TEX n_val.z, lun_tc, texture[0], 3D;
TEX n_val.w, run_tc, texture[0], 3D;

TEX m_val.x, llm_tc, texture[0], 3D;
TEX m_val.y, rlm_tc, texture[0], 3D;
TEX m_val.z, lum_tc, texture[0], 3D;
TEX m_val.w, rum_tc, texture[0], 3D;

TEX f_val.x, llf_tc, texture[0], 3D;
TEX f_val.y, rlf_tc, texture[0], 3D;
TEX f_val.z, luf_tc, texture[0], 3D;
TEX f_val.w, ruf_tc, texture[0], 3D;

# output derivative in x direction
LRP n_val.zw, y_ofs.z, n_val.yzxy, n_val;    
LRP m_val.zw, y_ofs.z, m_val.yzxy, m_val;    
LRP f_val.zw, y_ofs.z, f_val.yzxy, f_val;

LRP n_val.x, x_ofs.z, n_val.z, n_val.w;     
LRP n_val.y, x_ofs.z, m_val.z, m_val.w;     
LRP n_val.z, x_ofs.z, f_val.z, f_val.w;

DP3 result.color, n_val, {1,-2,1,0};

END
