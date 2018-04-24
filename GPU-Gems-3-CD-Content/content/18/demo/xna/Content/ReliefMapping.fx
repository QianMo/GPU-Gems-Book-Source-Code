float4x4 g_WorldViewProj;

texture g_ColorMap;
texture g_ReliefMap;

float3 g_LightPos;
float3 g_CameraPos;

bool g_DepthBias = true;
bool g_BorderClamp = true;
float g_DepthScale = 0.15;
float g_TextureTile = 1.0;

float4 g_ColorDiff = { 0.85, 0.85, 0.85, 1 };
float4 g_ColorSpec = { 0.6, 0.6, 0.6, 64 };
float3 g_ColorAmb = { 0.2, 0.2, 0.2 };

sampler ReliefSampler = 
sampler_state
{
    Texture = <g_ReliefMap>;
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

sampler ColorSampler = 
sampler_state
{
    Texture = <g_ColorMap>;
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

struct VS_OUTPUT
{
	float4 Position  : POSITION;
	float2 TexCoord  : TEXCOORD0;
	float3 ViewDir   : TEXCOORD1;
	float3 LightDir  : TEXCOORD2;
};

VS_OUTPUT MainVS( 
	float4 Pos : POSITION, 
	float3 Normal : NORMAL,
	float2 TexCoord : TEXCOORD0,
	float3 Tangent : TANGENT,
	float3 Binormal : BINORMAL)
{
	VS_OUTPUT Output;

	Output.Position = mul( Pos, g_WorldViewProj );
	Output.TexCoord = TexCoord * g_TextureTile;
	
	float3x3 TangentSpaceMatrix = float3x3(Tangent,Binormal,Normal);
	
	Output.ViewDir = mul( TangentSpaceMatrix, Pos.xyz - g_CameraPos );
	Output.LightDir = mul( TangentSpaceMatrix, g_LightPos - Pos.xyz );
	
	return Output;
}

// setup ray pos and dir based on view vector
// and apply depth bias and depth factor
void setup_ray(VS_OUTPUT IN,out float3 p,out float3 v)
{
	p = float3(IN.TexCoord,0);
	v = normalize(IN.ViewDir);
	
	v.z = abs(v.z);

	if (g_DepthBias)
	{
		float db=1.0-v.z; 
		db*=db; 
		db*=db; 
		db=1.0-db*db;
		v.xy*=db;
	}
	
	v.xy *= g_DepthScale;
}

// do normal mapping using given texture coordinate
// tangent space phong lighting with optional border clamp
// normal X and Y stored in red and green channels
float4 normal_mapping(
	sampler2D color_map,
	sampler2D normal_map,
	float2 texcoord,
	VS_OUTPUT IN)
{
	// color map
	float4 color = tex2D(color_map,texcoord);
	
	// normal map
	float4 normal = tex2D(normal_map,texcoord);
	normal.xy = 2*normal.xy - 1;
	normal.y = -normal.y;
	normal.z = sqrt(1.0 - normal.x*normal.x - normal.y*normal.y);

	// light and view in tangent space
	float3 l = normalize(IN.LightDir);
	float3 v = normalize(IN.ViewDir);

	// compute diffuse and specular terms
	float diff = saturate(dot(l,normal.xyz));
	float spec = saturate(dot(normalize(l-v),normal.xyz));

	// attenuation factor
	float att = 1.0 - max(0,l.z); 
	att = 1.0 - att*att;

	// border clamp
	float alpha=1;
	if (g_BorderClamp)
	{
		if (texcoord.x<0) alpha=0;
		if (texcoord.y<0) alpha=0;
		if (texcoord.x>g_TextureTile) alpha=0;
		if (texcoord.y>g_TextureTile) alpha=0;
	}
	
	// compute final color
	float4 finalcolor;
	finalcolor.xyz = g_ColorAmb*color.xyz +
		att*(color.xyz*g_ColorDiff.xyz*diff +
		g_ColorSpec.xyz*pow(spec,g_ColorSpec.w));
	finalcolor.w = g_ColorDiff.w*alpha;
	return finalcolor;
}

// ray intersect depth map using linear and binary searches
// depth value stored in alpha channel (black at is object surface)
void ray_intersect_relief(
	sampler2D relief_map,
	inout float3 p,
	inout float3 v)
{
	const int num_steps_lin=15;
	const int num_steps_bin=6;
	
	v /= v.z*num_steps_lin;
	
	int i;
	for( i=0;i<num_steps_lin;i++ )
	{
		float4 tex = tex2D(relief_map, p.xy);
		if (p.z<tex.w)
			p+=v;
	}
	
	for( i=0;i<num_steps_bin;i++ )
	{
		v *= 0.5;
		float4 tex = tex2D(relief_map, p.xy);
		if (p.z<tex.w)
			p+=v;
		else
			p-=v;
	}
}

// ray intersect depth map using relaxed cone stepping
// depth value stored in alpha channel (black is at object surface)
// and cone ratio stored in blue channel
void ray_intersect_relaxedcone(
	sampler2D relaxedcone_relief_map,
	inout float3 p,
	inout float3 v)
{
	const int cone_steps=15;
	const int binary_steps=8;
	
	float3 p0 = p;

	v /= v.z;
	
	float dist = length(v.xy);
	
	for( int i=0;i<cone_steps;i++ )
	{
		float4 tex = tex2D(relaxedcone_relief_map, p.xy);

		float height = saturate(tex.w - p.z);
		
		float cone_ratio = tex.z;
		
		p += v * (cone_ratio * height / (dist + cone_ratio));
	}

	v *= p.z*0.5;
	p = p0 + v;

	for( int i=0;i<binary_steps;i++ )
	{
		float4 tex = tex2D(relaxedcone_relief_map, p.xy);
		v *= 0.5;
		if (p.z<tex.w)
			p+=v;
		else
			p-=v;
	}
}

float4 NormalMappingPS(VS_OUTPUT IN) : COLOR
{
	return normal_mapping(ColorSampler,ReliefSampler,IN.TexCoord,IN);
}

float4 ReliefMappingPS(VS_OUTPUT IN) : COLOR
{
	float3 p,v;

	setup_ray(IN,p,v);

	ray_intersect_relief(ReliefSampler,p,v);

	return normal_mapping(ColorSampler,ReliefSampler,p.xy,IN);
}

float4 RelaxedConeMappingPS(VS_OUTPUT IN) : COLOR
{
	float3 p,v;

	setup_ray(IN,p,v);

	ray_intersect_relaxedcone(ReliefSampler,p,v);

	return normal_mapping(ColorSampler,ReliefSampler,p.xy,IN);
}

technique NormalMapping
{
	pass P0
	{          
		VertexShader = compile vs_1_1 MainVS( );
		PixelShader  = compile ps_2_0 NormalMappingPS( ); 
	}
}

technique ReliefMapping
{
	pass P0
	{          
		VertexShader = compile vs_1_1 MainVS( );
		PixelShader  = compile ps_2_a ReliefMappingPS( ); 
	}
}

technique RelaxedConeMapping
{
	pass P0
	{          
		VertexShader = compile vs_1_1 MainVS( );
		PixelShader  = compile ps_2_a RelaxedConeMappingPS( ); 
	}
}
