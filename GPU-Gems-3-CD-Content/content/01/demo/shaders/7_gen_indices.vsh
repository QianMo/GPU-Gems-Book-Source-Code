//input: nonempty_cell_list_geom

struct a2vConnector {
  uint z8_y8_x8_case8 : TEX2;
};

struct v2gConnector {
  uint z8_y8_x8_case8 : TEX2;
};

v2gConnector main(a2vConnector a2v)
{
  v2gConnector v2g;
  v2g.z8_y8_x8_case8 = a2v.z8_y8_x8_case8;
  return v2g;
}

