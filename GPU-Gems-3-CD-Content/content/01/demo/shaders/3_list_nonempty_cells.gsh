struct v2gConnector {
  uint z8_y8_x8_case8 : TEX2;
};

struct g2fConnector {
  uint z8_y8_x8_case8 : TEX2;
};

[maxvertexcount (1)]
void main(inout PointStream<g2fConnector> Stream, point v2gConnector input[1])
{
  uint cube_case = (input[0].z8_y8_x8_case8 & 0xFF);
  if (cube_case * (255-cube_case) > 0)
  {
    g2fConnector output;
    output.z8_y8_x8_case8 = input[0].z8_y8_x8_case8;
    Stream.Append(output);
  }
}