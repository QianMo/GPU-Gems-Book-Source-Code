float4x4 World;					///< World matrix for the current object
float4x4 WorldIT;				///< World matrix IT (inverse transposed) to transform surface normals of the current object 
float4x4 WorldView;
float4x4 WorldViewIT;
float4x4 WorldViewProj;
float3 referencePos;
float3 eyePos;
float3 F0; // Freshnel factor
float N0; // Refraction coefficient

int MAX_LIN_ITERATIONCOUNT;
int MIN_LIN_ITERATIONCOUNT;
int SECANT_ITERATIONCOUNT;
int MAX_RAY_DEPTH;

texture envCube1;
sampler envCubeSampler1 = sampler_state 
{
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = POINT;    
    Texture   = <envCube1>;
    AddressU  = WRAP;
    AddressV  = WRAP;
};

texture envCube2;
sampler envCubeSampler2 = sampler_state 
{
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = POINT;    
    Texture   = <envCube2>;
    AddressU  = WRAP;
    AddressV  = WRAP;
};

texture envCube3;
sampler envCubeSampler3 = sampler_state 
{
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = POINT;    
    Texture   = <envCube3>;
    AddressU  = WRAP;
    AddressV  = WRAP;
};

texture envCube4;
sampler envCubeSampler4 = sampler_state 
{
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;    
    Texture   = <envCube4>;
    AddressU  = WRAP;
    AddressV  = WRAP;
};

texture envCube5;
sampler envCubeSampler5 = sampler_state 
{
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;    
    Texture   = <envCube5>;
    AddressU  = WRAP;
    AddressV  = WRAP;
};

texture envCube6;
sampler envCubeSampler6 = sampler_state 
{
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;    
    Texture   = <envCube6>;
    AddressU  = WRAP;
    AddressV  = WRAP;
};


void linearSearch(  float3 x, float3 R, samplerCUBE mp,
                    out float3 p,
                    out float dl,
                    out float dp,
                    out float llp,
                    out float ppp)
{	
	p = 1;
	 	
	float a = length(x) / length(R);
	float3 s = normalize(x);
	float3 e = normalize(R);
	float dt = (-dot(s, e) + 1.0f) / 2.0f * ((float) MAX_LIN_ITERATIONCOUNT);
    dt = max(dt, MIN_LIN_ITERATIONCOUNT);
    dt = 1.0f / dt;
	bool   undershoot = false, overshoot = false;	
	
	
	float t = 0.01;
	while( t < 1 && !(overshoot && undershoot) ) {  // iteration 
		float dr = a * t / (1 - t);  	// ray parameter corresponding to t
	  	float3 r = x + R * dr;        	// point on the ray
	  	float ra =  texCUBElod(mp, float4(r, 0)).a; 	// |p'|: distance direction of p
	  	
	  	
	  	if (ra > 0) {		// valid texel, i.e. anything is visible
    		float rrp = length(r)/ra; //|r|/|r'|
   	 		if (rrp < 1) {     	// undershooting
      			dl = dr;    	// store last undershooting as l
      			llp = rrp;
      			undershoot = true;
    		} else {
      			dp = dr;    	// store last overshooting as p
      			ppp = rrp;
	    		overshoot = true;}
  		} else {			// nothing is visible: restart search
    		undershoot = false;	
    		overshoot = false;
		}	
	  	t += dt;                	// next texel
	}

 	if(!(overshoot && undershoot))
  	  p = float3(0,0,0); 

}

void secantSearch(float3 x, float3 R, samplerCUBE mp, 
	               float dl,
	               float dp,
	               float llp,
	               float ppp,
	               out float3 p)
{
  p = x + R * dp; // if no secant iteration
  for(int i= 0; i < SECANT_ITERATIONCOUNT; i++)
  {
   float dnew;
   dnew = dl + (dp - dl) * (1 - llp) / (ppp - llp);
   p = x + R * dnew;
   half pppnew = length(p) / texCUBElod(mp, float4(p, 0)).a;
   if(pppnew == 1.0)
	i = SECANT_ITERATIONCOUNT;
   else if(pppnew < 1.0f)
   {
    llp = pppnew;
    dl = dnew;
   }
   else
   {
    ppp = pppnew;
    dp = dnew;
   }
  }
}

float3 Hit(float3 x, float3 R, out float4 Il, out float3 Nl)
{
 float dl1 = 0, dp1, llp1, ppp1;
 float3 p1;
 linearSearch(x, R, envCubeSampler4, p1, dl1, dp1, llp1, ppp1);
 float dl2 = 0, dp2, llp2, ppp2;
 float3 p2;
 linearSearch(x, R, envCubeSampler5, p2, dl2, dp2, llp2, ppp2);
 
 bool valid1 = dot(p1,p1) != 0;
 bool valid2 = dot(p2,p2) != 0;
		
 float dl = 0, dp, llp, ppp;
 float3 p;
 
 if(!valid1 && !valid2)
 {
    linearSearch(x, R, envCubeSampler6, p, dl, dp, llp, ppp);
    secantSearch(x, R, envCubeSampler6, dl, dp, llp, ppp, p);
    Il = texCUBElod(envCubeSampler3, float4(p, 0));  
    Nl = texCUBElod(envCubeSampler6, float4(p, 0)).rgb;  
 }
 else
 {
    if( !valid2 || (valid1 && dp1 < dp2))
    {
     secantSearch(x, R, envCubeSampler4, dl1, dp1, llp1, ppp1, p1);
     Il = texCUBElod(envCubeSampler1, float4(p1, 0));  
	 Nl = texCUBElod(envCubeSampler4, float4(p1, 0)).rgb;  
     p = p1;
    }
    else
    {
     secantSearch(x, R, envCubeSampler5, dl2, dp2, llp2, ppp2, p2);
     Il = texCUBElod(envCubeSampler2, float4(p2, 0));  
     Nl = texCUBElod(envCubeSampler5, float4(p2, 0)).rgb;  
     p = p2;
    }
 }
 
 return p;
}




struct SpecularReflection_VS_OUT
{
	float4 hPos : POSITION; 	// clipping space
  	float3 x    : TEXCOORD1;	// cube map space
  	float3 N    : TEXCOORD2;	// normal
  	float3 V    : TEXCOORD3; 	// view
};

SpecularReflection_VS_OUT SpecularReflectionVS(    	
	float4 Pos  : POSITION,
  	float4 Norm : NORMAL)
{
	SpecularReflection_VS_OUT OUT;
  	OUT.hPos = mul(Pos, WorldViewProj);
  	float3 wPos = mul(Pos, World).xyz;
  	OUT.N    = mul(Norm, WorldIT).xyz;
  	OUT.V    = wPos - eyePos;
  	OUT.x    = wPos - referencePos;
  	return OUT;
}

float4 SingleReflectionPS( SpecularReflection_VS_OUT IN) : COLOR
{
//return 1;
	float3 V = normalize(IN.V); float3 N = normalize(IN.N);
  	float3 R;               // reflection dir.
  	float3 Nl;   			// normal vector at the hit point
  	float4 Il;   			// radiance at the hit point
  	float3 I = 1;
	
  	float3 F = F0 + pow(1-dot(N, -V), 5) * (1 - F0);
	if (N0 <= 0) // reflective material
	{
     	R = reflect(V, N);
      	I *= F; // Fresnel reflection
    }
    else //refractive material
    {
      	R = refract(V, N, N0);
      	if (dot(R, R) == 0)	// no refraction direction exits
      		R = reflect(V, N); // total reflection  			
		else
     		I *= (1 - F);      // Fresnel refraction
    }
    // ray hit l, radiance Il, normal Nl 
    float3 l = Hit(IN.x, R, Il, Nl);
   
    //return float4(Nl,1);
	return Il * float4(I,1) * (Il.a == 0);
}


technique SingleReflection
{																
			pass p0														
			{															
				VertexShader = compile vs_2_0 SpecularReflectionVS();		
				PixelShader  = compile ps_3_0 SingleReflectionPS();	
			}															
}

float4 MultipleReflectionPS( SpecularReflection_VS_OUT IN ) : COLOR
{
//return 1;
	float3 V = normalize(IN.V); float3 N = normalize(IN.N);
  	float3 x = IN.x;
  	
  	float4 I = float4(1,1,1,0); // radiance of the path
  	float3 Fp = F0;           // Fresnel at 90 degrees at first hit
  	float  n = N0;             // index of refraction of the first hit
  	int depth = 0;		// length of the path
  	while (depth < MAX_RAY_DEPTH) {
    	float3 R;      	// reflection or refraction dir
    	float3 F = Fp + pow(1-abs(dot(N, -V)), 5) * (1-Fp);  // Fresnel
        if (n <= 0) {
      		R = reflect(V, N);  // reflection
      		I.rgb *= F;             // Fresnel reflection
    	}
        else{              	// refraction
      		if (dot(V,N) > 0) { // coming from inside
        			n = 1.0 / n;
        			N = -N;
      		}
      		R = refract(V, N, n);
      		if (dot(R, R) == 0)	// no refraction direction exits
      			R = reflect(V, N); // total reflection  			
			else
     			I.rgb *= (1-F);        // Fresnel refraction
    	}
    	
    	float4 Il;      	// radiance at the hit point
  		float3 Nl;		// normal vector at the hit point
    	// Trace ray x+R*d and obtain hit l, radiance Il, normal Nl 
 	    float3 l = Hit(x, R, Il, Nl);
 	    n = Il.a;
 	    if(n != 0)// hit point is on specular surface 	    
 	    {    	
 			Fp = Il.rgb;
      		depth += 1;      		
    	}
    	else
    	{         	// hit point is on diffuse surface
      		I.rgb *= Il.rgb;	// multiply with the radiance      		
 			depth = MAX_RAY_DEPTH;   // terminate
 			I.a = 1;
    	}
    	N = Nl; V = R; x = l; // hit point is the next shaded point
  	}
  	return I * I.a;
}

technique MultipleReflection
{																
			pass p0														
			{															
				VertexShader = compile vs_2_0 SpecularReflectionVS();		
				PixelShader  = compile ps_3_0 MultipleReflectionPS();	
			}															
}
