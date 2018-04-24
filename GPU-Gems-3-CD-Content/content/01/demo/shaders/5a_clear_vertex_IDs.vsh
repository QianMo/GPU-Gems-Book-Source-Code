struct a2vConnector {
  float3 objCoord : POSITION;
  float2 tex : TEXCOORD;
  uint   nInstanceID : SV_InstanceID;
};

struct v2fConnector {
  float4 POSITION    : POSITION;
  uint   nInstanceID : TEX2;
};

v2fConnector main(a2vConnector a2v)
{
    v2fConnector v2f;
    float4 projCoord = float4(a2v.objCoord.xyz, 1);
      projCoord.y *= -1;               // flip Y coords for DX
    
    v2f.POSITION = projCoord;
    v2f.nInstanceID = a2v.nInstanceID;

    return v2f;
}