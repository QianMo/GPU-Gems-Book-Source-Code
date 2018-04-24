struct a2vConnector {
  float4 projCoord : POSITION;  // MUST BE A FLOAT4 OR DX10::DRAWTEXT WILL BREAK!
};


// holds nothing but 3d position
struct v2fConnector {
  float4 projCoord : POSITION;
};

// moves the coord over and returns the new connector
v2fConnector main(a2vConnector a2v){
  v2fConnector v2f;
  v2f.projCoord = float4(a2v.projCoord.xyz, 1.0);
  return v2f;
}
