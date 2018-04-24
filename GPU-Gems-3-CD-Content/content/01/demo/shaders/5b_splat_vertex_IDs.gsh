// note: the only reason this GS exists is because the VS 
//       can't specify SV_RenderTargetArrayIndex...
//       (i.e. only a GS can render to slices of a 3d texture)

struct v2gConnector {
  float4 projCoord       : POSITION;
  uint2  vertexID_and_slice : TEX2;
};

struct g2fConnector {
  float4 projCoord    : POSITION;
  uint   vertexID     : TEX2;
  uint   RTIndex      : SV_RenderTargetArrayIndex;
};

[maxvertexcount (1)]
void main( point v2gConnector input[1],
           inout PointStream<g2fConnector> Stream
         )
{
  g2fConnector g2f;
  g2f.projCoord = input[0].projCoord;
  g2f.vertexID  = input[0].vertexID_and_slice.x;
  g2f.RTIndex   = input[0].vertexID_and_slice.y;
  Stream.Append(g2f);
}
