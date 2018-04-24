struct v2gConnector {
  uint z8_y8_x8_null5_edgeFlags3 : TEX2;
};

struct g2fConnector {
  uint z8_y8_x8_null4_edgeNum4 : TEX2;
};

[maxvertexcount (3)]
void main(inout PointStream<g2fConnector> Stream, point v2gConnector input[1])
{
  g2fConnector output;

  uint z8_y8_x8_null8 = input[0].z8_y8_x8_null5_edgeFlags3 & 0xFFFFFF00;
  
  if (input[0].z8_y8_x8_null5_edgeFlags3 & 1) {
    output.z8_y8_x8_null4_edgeNum4 = z8_y8_x8_null8 | 3;
    Stream.Append(output);
  }
  if (input[0].z8_y8_x8_null5_edgeFlags3 & 2) {
    output.z8_y8_x8_null4_edgeNum4 = z8_y8_x8_null8 | 0;
    Stream.Append(output);
  }
  if (input[0].z8_y8_x8_null5_edgeFlags3 & 4) {
    output.z8_y8_x8_null4_edgeNum4 = z8_y8_x8_null8 | 8;
    Stream.Append(output);
  }
}