// note: the only reason this GS exists is because the VS 
//       can't specify SV_RenderTargetArrayIndex...
//       (i.e. only a GS can render to slices of a 3d texture)

struct v2gConnector {
  float4 projCoord    : POSITION;
  float4 wsCoord      : TEXCOORD;
  float3 chunkCoord   : TEXCOORD1;
  uint   nInstanceID  : BLAH;
};

struct g2fConnector {
  float4 projCoord    : POSITION;
  float4 wsCoord      : TEXCOORD;
  float3 chunkCoord   : TEXCOORD1;
  uint   RTIndex      : SV_RenderTargetArrayIndex;
};

[maxvertexcount (3)]
void main( triangle v2gConnector input[3],
           inout TriangleStream<g2fConnector> Stream
         )
{
    for (int v=0; v<3; v++) 
    {
      g2fConnector g2f;
      g2f.projCoord    = input[v].projCoord;
      g2f.wsCoord      = input[v].wsCoord;
      g2f.RTIndex      = input[v].nInstanceID;
      g2f.chunkCoord   = input[v].chunkCoord;
      Stream.Append(g2f);
    }
    Stream.RestartStrip();
}
