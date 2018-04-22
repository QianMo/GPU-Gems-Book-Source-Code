float4 lightPos : Position
<   string Object = "PointLight";
    string Space = "World";
> = {100.0f, 100.0f, 100.0f, 0.0f};

float4 diffColor : Diffuse
<   string Desc = "Diffuse Color";
> = {0.6f, 0.9f, 0.6f, 1.0f};

float specExpon : Power
<   string gui = "slider";
    float uimin = 1.0;
    float uimax = 128.0;
    float uistep = 1.0;
    string Desc = "Specular Exponent";
> = 30.0;

float4x4 worldIT : WorldIT;
float4x4 wvp : WorldViewProjection;
float4x4 world : World;
float4x4 viewIT : ViewIT;

struct appdata {
    float3 Position    : POSITION;
    float4 Normal      : NORMAL;
};

struct vertexOutput {
    float4 HPosition   : POSITION;
    float3 LightVec    : TEXCOORD0;
    float3 WorldNormal : TEXCOORD1;
    float3 WorldPos    : TEXCOORD2;
    float3 WorldView   : TEXCOORD3;
};

struct pixelOutput {
  float4 col : COLOR;
};

vertexOutput myVertexShader(appdata IN,
    uniform float4x4 WorldViewProj,
    uniform float4x4 WorldIT,
    uniform float4x4 World,
    uniform float4x4 ViewIT,
    uniform float3 LightPos)
{
    vertexOutput OUT;
    OUT.WorldNormal = mul(WorldIT, IN.Normal).xyz;
    float4 Po = float4(IN.Position.x, IN.Position.y, IN.Position.z, 1.0);
    float3 Pw = mul(World, Po).xyz;
    OUT.WorldPos = Pw;
    OUT.LightVec = normalize(LightPos - Pw);
    OUT.WorldView = normalize(ViewIT[3].xyz - Pw);
    OUT.HPosition = mul(WorldViewProj, Po);
    return OUT;
}

pixelOutput myPixelShader(vertexOutput IN,
    uniform float4 DiffColor,
    uniform float SpecExpon)
{
    pixelOutput OUT; 
    float3 Ln = normalize(IN.LightVec);
    float3 Nn = normalize(IN.WorldNormal);
    float ldn = dot(Ln, Nn);
    float3 Vn = normalize(IN.WorldView);
    float3 Hn = normalize(Vn + Ln);
    float hdn = ldn * pow(max(0, dot(Hn, Nn)), SpecExpon);
    OUT.col = hdn*float4(1.0, 1.0, 1.0, 1.0) + max(0, ldn)*DiffColor;
    return OUT;
}

technique dx9
{
	pass p0 
	{		
        VertexShader = compile vs_2_0 myVertexShader(wvp, worldIT, world, viewIT, lightPos);
        ZEnable = true;
        ZWriteEnable = true;
        CullMode = CW; // Cinema 4D uses clockwise culling
        PixelShader = compile ps_2_0 myPixelShader(diffColor, specExpon);
	}
}