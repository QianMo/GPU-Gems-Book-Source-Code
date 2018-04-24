// (PASSTHRU)

struct v2gConnector {
  float4 wsCoordAmbo : COORD1;
  float3 wsNormal    : NORM1;
};

struct g2fConnector {
  float4 worldCoordAmbo  : POSITION;     // .w = occlusion
  float4 worldNormal     : NORMAL;       
};

[maxvertexcount (1)]
void main(inout PointStream<g2fConnector> Stream, point v2gConnector input[1])
{
  g2fConnector output;
  output.worldCoordAmbo  = input[0].wsCoordAmbo;
  output.worldNormal     = float4(input[0].wsNormal, 0);
  Stream.Append(output);
}