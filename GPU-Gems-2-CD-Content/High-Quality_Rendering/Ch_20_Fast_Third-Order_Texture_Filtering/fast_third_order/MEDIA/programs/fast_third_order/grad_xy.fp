# parameters              = 7
# parameters_native       = 7
# attribs                 = 1
# attribs_native          = 1
# temporaries             = 17
# temporaries_native      = 18
# instructions            = 39
# instructions_native     = 40
# instructions_alu        = 27
# instructions_alu_native = 28
# instructions_tex        = 12
# instructions_tex_native = 12
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
	TEMP	coord_ow;
	MAD		coord_ow, filter_srcscale, coord_src, -const_0512.y;
	TEMP	tex_ow_x;
	TEMP	tex_ow_y;
	TEX		tex_ow_x, coord_ow.x, texture[ 3 ], 1D;
	TEX		tex_ow_y, coord_ow.y, texture[ 3 ], 1D;
	TEMP	tex_ow_z;
	TEX		tex_ow_z, coord_ow.z, texture[ 2 ], 1D;
	MUL		tex_ow_x.xy, tex_ow_x, filter_srcofs.x;
	MUL		tex_ow_y.xy, tex_ow_y, filter_srcofs.y;
	MUL		tex_ow_z.xy, tex_ow_z, filter_srcofs.z;
	TEMP	coord_src000;
	TEMP	coord_src100;
	TEMP	coord_src010;
	TEMP	coord_src110;
	MAD		coord_src100, const_e_x,  tex_ow_x.x, coord_src;
	MAD		coord_src000, const_e_x, -tex_ow_x.y, coord_src;
	MAD		coord_src110, const_e_y,  tex_ow_y.x, coord_src100;
	MAD		coord_src010, const_e_y,  tex_ow_y.x, coord_src000;
	MAD		coord_src100, const_e_y, -tex_ow_y.y, coord_src100;
	MAD		coord_src000, const_e_y, -tex_ow_y.y, coord_src000;
	TEMP	coord_src001;
	TEMP	coord_src101;
	TEMP	coord_src011;
	TEMP	coord_src111;
	MAD		coord_src111, const_e_z,  tex_ow_z.x, coord_src110;
	MAD		coord_src011, const_e_z,  tex_ow_z.x, coord_src010;
	MAD		coord_src101, const_e_z,  tex_ow_z.x, coord_src100;
	MAD		coord_src001, const_e_z,  tex_ow_z.x, coord_src000;
	MAD		coord_src110, const_e_z, -tex_ow_z.y, coord_src110;
	MAD		coord_src010, const_e_z, -tex_ow_z.y, coord_src010;
	MAD		coord_src100, const_e_z, -tex_ow_z.y, coord_src100;
	MAD		coord_src000, const_e_z, -tex_ow_z.y, coord_src000;
	TEMP	tex_src0;
	TEX		tex_src0.r, coord_src000, texture[ 1 ], 3D;
	TEX		tex_src0.g, coord_src010, texture[ 1 ], 3D;
	TEX		tex_src0.b, coord_src001, texture[ 1 ], 3D;
	TEX		tex_src0.a, coord_src011, texture[ 1 ], 3D;
	TEMP	tex_src1;
	TEX		tex_src1.r, coord_src100, texture[ 1 ], 3D;
	TEX		tex_src1.g, coord_src110, texture[ 1 ], 3D;
	TEX		tex_src1.b, coord_src101, texture[ 1 ], 3D;
	TEX		tex_src1.a, coord_src111, texture[ 1 ], 3D;
	ALIAS	tex_ow_xp = tex_ow_y;
	ALIAS	tex_ow_yp = tex_ow_z;
	ALIAS	tex_ow_zp = tex_ow_x;
	ADD		tex_src0, tex_src0, -tex_src1;
	MUL		tex_src0, tex_ow_zp.z, tex_src0;
	LRP		tex_src0.rg, tex_ow_yp.z, tex_src0.baaa, tex_src0;
	ADD		tex_src0.r, tex_src0.r, -tex_src0.g;
	MUL		tex_src0.r, tex_ow_xp.z, tex_src0.r;
	ALIAS	tex_filtered = tex_src0;
	MOV		gradient, tex_filtered.x;
	MOV		result.color.rgb, gradient;
	MOV		result.color.a, intersect.a;
	END
