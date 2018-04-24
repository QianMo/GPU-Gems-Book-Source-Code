struct v2gConnector {
  float4 POSITION    : POSITION;
  uint   nInstanceID : TEX2;
};

struct g2fConnector {
  float4 POSITION  : POSITION;
  uint   RTIndex   : SV_RenderTargetArrayIndex;
};

[maxvertexcount (3)]
void main( triangle v2gConnector input[3],
           inout TriangleStream<g2fConnector> Stream
         )
{
  for (int i=0; i<3; i++) 
  {
    g2fConnector g2f;
    g2f.POSITION = input[i].POSITION;
    g2f.RTIndex  = input[i].nInstanceID;
    Stream.Append(g2f);
  }
  Stream.RestartStrip();
}
