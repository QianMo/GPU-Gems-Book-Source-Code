struct VsVertex
{
	float3 pos;
	float3 normal;
};

float3 getTrunkAxis()
{
	return float3(0,0,1);
}

float lerp3(float a, float b, float c, float t)
{
	return 
		lerp(
			lerp(a, b, saturate(t+1)),
			c,
			saturate(t));
}