// This is a simple vertex shader for passing screen
// coordinates unchanged to the fragment shader

// holds nothing but 3d position
struct a2vConnector {
  float3 projCoord : POSITION;
  float2 tex       : TEXCOORD;
};

// holds nothing but 3d position
struct v2fConnector {
  float4 projCoord : POSITION;
  float2 tex       : TEXCOORD;
};

// moves the coord over and returns the new connector
v2fConnector main(a2vConnector a2v){
  v2fConnector v2f;
  v2f.projCoord = float4(a2v.projCoord, 1);
  v2f.tex       = a2v.tex;
  return v2f;
}