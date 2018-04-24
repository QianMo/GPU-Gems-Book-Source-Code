struct a2vConnector {
  float3 objCoord    : POSITION;
};

struct v2fConnector {
  float4 projCoord   : POSITION;
  float3 worldCoord  : TEXCOORD;
  //float3 NoiseCoord1 : TEXCOORD2;
  //float3 NoiseCoord2 : TEXCOORD3;
  //float3 NoiseCoord3 : TEXCOORD4;
};

cbuffer ShaderCB {
  matrix view;
  matrix proj;
}

//cbuffer BkgNoiseTransforms {
//  float4x4 noiseXform1;
//  float4x4 noiseXform2;
//  float4x4 noiseXform3;
//}

float3 vecMul(const float4x4 mat, const float3 vec){
  return(float3(dot(vec, mat._11_12_13),
                dot(vec, mat._21_22_23),
                dot(vec, mat._31_32_33)));
}

v2fConnector main(a2vConnector a2v)
{
    v2fConnector v2f;

    float3 wsCoordNoTranslate = a2v.objCoord - view._41_42_43;
    float3 wsSphereCoord = normalize(wsCoordNoTranslate);
    v2f.worldCoord = wsSphereCoord;
    
    
    float3 esCoordNoTranslate = vecMul( view, wsCoordNoTranslate.xyz );
    v2f.projCoord = mul( proj, float4(esCoordNoTranslate,1) );
    v2f.projCoord.z = v2f.projCoord.w*0.999;

    //v2f.NoiseCoord1 = vecMul(noiseXform1, wsSphereCoord);
    //v2f.NoiseCoord2 = vecMul(noiseXform2, wsSphereCoord);
    //v2f.NoiseCoord3 = vecMul(noiseXform3, wsSphereCoord);

    return v2f;
}
