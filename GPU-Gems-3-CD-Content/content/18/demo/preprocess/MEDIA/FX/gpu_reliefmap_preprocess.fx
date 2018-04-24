float4x4 g_WorldViewProj;

texture g_ReliefMap : TEXUNIT0;
texture g_ColorMap : TEXUNIT1;

float3 g_ViewDir;
float3 g_Offset;

sampler ReliefSampler = 
sampler_state
{
    Texture = <g_ReliefMap>;
    MinFilter = POINT;
    MagFilter = POINT;
    AddressU = WRAP;
    AddressV = WRAP;
};

sampler ResultSampler = 
sampler_state
{
    Texture = <g_ColorMap>;
    MinFilter = POINT;
    MagFilter = POINT;
};

sampler ColorSampler = 
sampler_state
{
    Texture = <g_ColorMap>;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};

struct VS_OUTPUT
{
	float4 Position  : POSITION;
	float2 TexCoord  : TEXCOORD0;
	float2 TexCoordInvY : TEXCOORD1;
};

VS_OUTPUT vertexshader_common( 
	float4 Pos : POSITION, 
	float2 TexCoord : TEXCOORD0)
{
	VS_OUTPUT output;

	output.Position = mul( Pos, g_WorldViewProj );
	output.TexCoord = TexCoord;
	output.TexCoordInvY = TexCoord*float2(1,-1);
	
	return output;
}

float4 pixelshader_depth2cone( VS_OUTPUT input ) : COLOR
{
	float3 p;
	p.xy = input.TexCoord+g_Offset.xy;
	p.z = tex2D(ReliefSampler,p.xy).w;
	
	float d = tex2D(ReliefSampler,input.TexCoord).w;

	float r = length(g_Offset.xy);
	r = (p.z >= d) ? 1.0 : r / (d - p.z);
		
	float best_r = tex2D(ResultSampler,input.TexCoordInvY).x;
	if ( r > best_r )
		r = best_r;
		
	return float4(r,r,r,r);
}

float4 pixelshader_depth2quadcone( VS_OUTPUT input ) : COLOR
{
	float3 p;
	p.xy = input.TexCoord+g_Offset.xy;
	p.z = tex2D(ReliefSampler,p.xy).w;
	
	float d = tex2D(ReliefSampler,input.TexCoord).w;

	float r = length(g_Offset.xy);
	r = (p.z >= d) ? 1.0 : r / (d - p.z);

	int index=0;

	float2 abs_dir = abs(g_Offset.xy);
	if (abs_dir.x>abs_dir.y)
		index = g_Offset.x > 0 ? 0 : 1;
	else
		index = g_Offset.y > 0 ? 2 : 3;
		
	float4 best_r = tex2D(ResultSampler,input.TexCoordInvY);
	if ( r > best_r[index] )
		r = best_r[index];

	if (index==0)
		best_r.x = r;
	if (index==1)
		best_r.y = r;
	if (index==2)
		best_r.z = r;
	if (index==3)
		best_r.w = r;
		
	return best_r;
}

float4 pixelshader_depth2relaxedcone( VS_OUTPUT input ) : COLOR
{
	const int search_steps = 128;
	
	float3 p = float3(input.TexCoord,0);
	
	float3 o = g_Offset + p;
	o.z = tex2D(ReliefSampler,o.xy).w;
	
	float3 v = o - p;
	v /= v.z;
	v *= 1.0-o.z;
	v /= search_steps;

	p = o;

	for( int i=0;i<search_steps;i++ )
	{
		float d = tex2D(ReliefSampler,p.xy).w;

		if ( d <= p.z )
			p += v;
	}
	float d = tex2D(ReliefSampler,input.TexCoord).w;
	
	float r = length(p.xy-input.TexCoord);
	
	r = (p.z >= d) ? 1.0 : r / (d - p.z);		

	float best_r = tex2D(ResultSampler,input.TexCoordInvY).x;
	if ( r > best_r )
		r = best_r;
		
	return float4(r,r,r,r);
}

float4 viewtexture( VS_OUTPUT input ) : COLOR
{
	return tex2D(ColorSampler,input.TexCoord);
}

technique viewtexture
{
	pass P0
	{          
		VertexShader = compile vs_1_1 vertexshader_common( );
		PixelShader  = compile ps_2_0 viewtexture( ); 
	}
}

technique depth2cone
{
	pass P0
	{          
		VertexShader = compile vs_1_1 vertexshader_common( );
		PixelShader  = compile ps_2_0 pixelshader_depth2cone( ); 
	}
}

technique depth2quadcone
{
	pass P0
	{          
		VertexShader = compile vs_1_1 vertexshader_common( );
		PixelShader  = compile ps_2_0 pixelshader_depth2quadcone( ); 
	}
}

technique depth2relaxedcone
{
	pass P0
	{          
		VertexShader = compile vs_1_1 vertexshader_common( );
		PixelShader  = compile ps_3_0 pixelshader_depth2relaxedcone( ); 
	}
}
