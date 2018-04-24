//input: nonempty_cell_list_geom

struct a2vConnector {
  uint z8_y8_x8_case8 : TEX2;
};

struct v2gConnector {
  uint z8_y8_x8_null5_edgeFlags3 : TEX2;
};

v2gConnector main(a2vConnector a2v)
{
  int cube_case = (int)(a2v.z8_y8_x8_case8 & 0xFF);
  int  bit0 = (cube_case     ) & 1;
  int  bit3 = (cube_case >> 3) & 1;
  int  bit1 = (cube_case >> 1) & 1;
  int  bit4 = (cube_case >> 4) & 1;
  int3 build_vert_on_edge = abs( int3(bit3,bit1,bit4) - bit0.xxx );

  uint bits = a2v.z8_y8_x8_case8 & 0xFFFFFF00;
  if (build_vert_on_edge.x != 0)
    bits |= 1;
  if (build_vert_on_edge.y != 0)
    bits |= 2;
  if (build_vert_on_edge.z != 0)
    bits |= 4;
  
  v2gConnector v2g;
  v2g.z8_y8_x8_null5_edgeFlags3 = bits;
  return v2g;
}
