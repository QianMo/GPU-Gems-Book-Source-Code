# parameters              = 8
# parameters_native       = 8
# attribs                 = 1
# attribs_native          = 1
# temporaries             = 21
# temporaries_native      = 22
# instructions            = 43
# instructions_native     = 52
# instructions_alu        = 28
# instructions_alu_native = 37
# instructions_tex        = 15
# instructions_tex_native = 15
# indirections_tex        = 4
# indirections_tex_native = 4
#
!!ARBfp1.0
	OPTION	ARB_precision_hint_nicest;
	PARAM	const_0512 = { 0.0, 0.5, 1.0, 2.0 };
	PARAM	filter_srcofs = program.env[ 6 ];
	PARAM	filter_srcscale = program.env[ 7 ];
	TEMP	gradient;
	TEMP	screenpos;
	MUL		screenpos, fragment.position, {0.001953125,0.001953125,0.001953125,0.001953125};
	TEMP	tex_volpos;
	ALIAS	intersect = tex_volpos;
	TEX		tex_volpos, screenpos, texture[ 0 ], 2D;
	ALIAS	coord_src = tex_volpos;
	PARAM	const_e_x = { 1, 0, 0, 0 };
	PARAM	const_e_y = { 0, 1, 0, 0 };
	PARAM	const_e_z = { 0, 0, 1, 0 };
	ALIAS	const_e_xp = const_e_y;
	ALIAS	const_e_yp = const_e_z;
	ALIAS	const_e_zp = const_e_x;
	TEMP	coord_ow;
	MAD		coord_ow, filter_srcscale, coord_src, -const_0512.y;
	TEMP	tex_ow_xp;
	TEMP	tex_ow_yp;
	TEX		tex_ow_xp, coord_ow.y, texture[ 2 ], 1D;
	TEX		tex_ow_yp, coord_ow.z, texture[ 2 ], 1D;
	MUL		tex_ow_xp.xy, tex_ow_xp, filter_srcofs.y;
	MUL		tex_ow_yp.xy, tex_ow_yp, filter_srcofs.z;
	TEMP	coord_src000;
	TEMP	coord_src100;
	TEMP	coord_src010;
	TEMP	coord_src110;
	MAD		coord_src100, const_e_xp,  tex_ow_xp.x, coord_src;
	MAD		coord_src000, const_e_xp, -tex_ow_xp.y, coord_src;
	MAD		coord_src110, const_e_yp,  tex_ow_yp.x, coord_src100;
	MAD		coord_src010, const_e_yp,  tex_ow_yp.x, coord_src000;
	MAD		coord_src100, const_e_yp, -tex_ow_yp.y, coord_src100;
	MAD		coord_src000, const_e_yp, -tex_ow_yp.y, coord_src000;
	TEMP	coord_src00m;
	TEMP	coord_src10m;
	TEMP	coord_src01m;
	TEMP	coord_src11m;
	TEMP	coord_src00p;
	TEMP	coord_src10p;
	TEMP	coord_src01p;
	TEMP	coord_src11p;
	MAD		coord_src00p, const_e_zp,  filter_srcofs, coord_src000;
	MAD		coord_src10p, const_e_zp,  filter_srcofs, coord_src100;
	MAD		coord_src01p, const_e_zp,  filter_srcofs, coord_src010;
	MAD		coord_src11p, const_e_zp,  filter_srcofs, coord_src110;
	MAD		coord_src00m, const_e_zp, -filter_srcofs, coord_src000;
	MAD		coord_src10m, const_e_zp, -filter_srcofs, coord_src100;
	MAD		coord_src01m, const_e_zp, -filter_srcofs, coord_src010;
	MAD		coord_src11m, const_e_zp, -filter_srcofs, coord_src110;
	TEMP	tex_srcm;
	TEMP	tex_src0;
	TEMP	tex_srcp;
	TEX		tex_srcm.r, coord_src00m, texture[ 1 ], 3D;
	TEX		tex_srcm.g, coord_src10m, texture[ 1 ], 3D;
	TEX		tex_srcm.b, coord_src01m, texture[ 1 ], 3D;
	TEX		tex_srcm.a, coord_src11m, texture[ 1 ], 3D;
	TEX		tex_src0.r, coord_src000, texture[ 1 ], 3D;
	TEX		tex_src0.g, coord_src100, texture[ 1 ], 3D;
	TEX		tex_src0.b, coord_src010, texture[ 1 ], 3D;
	TEX		tex_src0.a, coord_src110, texture[ 1 ], 3D;
	TEX		tex_srcp.r, coord_src00p, texture[ 1 ], 3D;
	TEX		tex_srcp.g, coord_src10p, texture[ 1 ], 3D;
	TEX		tex_srcp.b, coord_src01p, texture[ 1 ], 3D;
	TEX		tex_srcp.a, coord_src11p, texture[ 1 ], 3D;
	LRP		tex_srcm.rg, tex_ow_yp.z, tex_srcm.baaa, tex_srcm;
	LRP		tex_srcm.ba, tex_ow_yp.z, tex_src0, tex_src0.rrrg;
	LRP		tex_srcp.rg, tex_ow_yp.z, tex_srcp.baaa, tex_srcp;
	LRP		tex_src0.r, tex_ow_xp.z, tex_srcm.g, tex_srcm.r;
	LRP		tex_src0.g, tex_ow_xp.z, tex_srcm.a, tex_srcm.b;
	LRP		tex_src0.b, tex_ow_xp.z, tex_srcp.g, tex_srcp.r;
	DP3		tex_src0.r, tex_src0, {1,-2,1,0};
	ALIAS	tex_filtered = tex_src0;
	MOV		gradient, tex_filtered.x;
	MOV		result.color.rgb, gradient;
	MOV		result.color.a, intersect.a;
	END
