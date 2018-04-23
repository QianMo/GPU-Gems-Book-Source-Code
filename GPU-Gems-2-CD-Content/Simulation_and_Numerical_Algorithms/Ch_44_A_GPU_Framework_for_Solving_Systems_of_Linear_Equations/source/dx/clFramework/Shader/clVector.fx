/************************************************************
 *															*
 * decr     : shaders for the clVector Classes				*
 * version  : 1.4											*
 * author   : Jens Krüger									*
 * date     : 15.04.2003									*
 * modified	: 18.02.2004									*
 * e-mail   : jens.krueger@in.tum.de						*
 *															*
 ************************************************************/

/////////////////////////////////////////////////////
// VARIABLES
/////////////////////////////////////////////////////

texture tVector;
texture tVector2;
texture tLastPass;
texture tMultiply;
float fMultiply;
float fMultiply2;
float fReduceStep;
float4 f4ReduceStep;
float4 f4TexShift;
float fShift;
float4 fSize;

/////////////////////////////////////////////////////
// STRUCTURES
/////////////////////////////////////////////////////

struct app2Vertex {
	float4 Position   : POSITION;
	float4 TexCoords  : TEXCOORD0;
};

struct vertex2pixel {
	float4 Position   : POSITION;
	float4 TexCoords  : TEXCOORD0;
};

sampler sVector = sampler_state {
	Texture = (tVector);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Wrap;
	AddressV  = Wrap;
};

sampler sVector2 = sampler_state {
	Texture = (tVector2);

	MinFilter = Point;
	MagFilter = Point;
	AddressU  = Wrap;
	AddressV  = Wrap;
};

sampler sLastPass = sampler_state {
	Texture = (tLastPass);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Wrap;
	AddressV  = Wrap;
};

sampler sMultiply = sampler_state {
	Texture = (tMultiply);

	MinFilter = Point;		// Point, Linear
	MagFilter = Point;
	AddressU  = Wrap;
	AddressV  = Wrap;
};

/////////////////////////////////////////////////////
// HELPER FUNCTIONS
/////////////////////////////////////////////////////
float2 computeShiftedCoordsPosOnly(float2 f2InCoords, float fShift) {
	float  temp;
	float2 f2ShiftCoords;

    temp=f2InCoords.x*fSize.x+fShift; 

#define ORIG 0
#if ORIG	
	f2ShiftCoords.x = temp%fSize.x;
	modf(temp/fSize.x,f2ShiftCoords.y);
#else
    f2ShiftCoords.x = fmod(temp, fSize.x); 
    temp = temp / fSize.x;
    f2ShiftCoords.y = floor(temp); // assumes positive temp
#endif

    f2ShiftCoords.y = f2InCoords.y*fSize.y+f2ShiftCoords.y;

	// NEEDED FOR NV30 BECAUSE OF THE LACK OF FPTEXTURE-WRAP
	//f2ShiftCoords = (f2ShiftCoords>=fSize) ? f2ShiftCoords-fSize : f2ShiftCoords;

	return f2ShiftCoords;
}

float4 computeShiftedCoordsPosOnlyP1(float2 f2InCoords, float fShift) {
	float2 temp;
	temp.x =f2InCoords.x*fSize.x+fShift; 
	temp.y =temp.x+1; 
	
	float4 f4ShiftCoords;

#if ORIG	
	f4ShiftCoords.xz = temp.xy%fSize.x;
	modf(temp.xy/fSize.x,f4ShiftCoords.yw);
#else
    f4ShiftCoords.xz = fmod(temp.xy, fSize.x);//f4ShiftCoords.xz = temp.xy%fSize.x;
    temp.xy = temp.xy/fSize.x;
    f4ShiftCoords.yw = floor(temp.xy); // assumes positive temp
#endif
    
    f4ShiftCoords.yw = f2InCoords.y*fSize.y+f4ShiftCoords.yw;
    
	// NEEDED FOR NV30 BECAUSE OF THE LACK OF FPTEXTURE-WRAP
	//	f4ShiftCoords = (f4ShiftCoords>=fSize.xyxy) ? f4ShiftCoords-fSize.xyxy : f4ShiftCoords;

	return f4ShiftCoords;
}

float2 computeShiftedCoordsAllCases(float2 f2InCoords, float fShift) {
    float  temp;
    float2 f2ShiftCoords;

    temp=f2InCoords.x*fSize.x+fShift; 

#define ORIG 0
#if ORIG	
	f2ShiftCoords.x = (fSize.x+temp)%fSize.x;
	
	float a;	modf(temp/fSize.x,a);
	float b;	modf(-(-temp+fSize.x)/fSize.x,b);
#else
	f2ShiftCoords.x = fmod((fSize.x+temp),fSize.x);
	
	float a = floor(temp / fSize.x);
	float b = floor(-(-temp+fSize.x)/fSize.x) + 1; // we know temp is negative here so floor+1 is correct.
#endif

	f2ShiftCoords.y = (temp >= 0) ? a : b;
	f2ShiftCoords.y = f2InCoords.y*fSize.y+f2ShiftCoords.y;

	// NEEDED FOR NV30 BECAUSE OF THE LACK OF FPTEXTURE-WRAP
	//	f2ShiftCoords = (f2ShiftCoords>=fSize) ? f2ShiftCoords-fSize : f2ShiftCoords;
	//	f2ShiftCoords = (f2ShiftCoords<0) ? f2ShiftCoords+fSize : f2ShiftCoords;

	return f2ShiftCoords;
}

float4 computeShiftedCoordsAllCasesP1(float2 f2InCoords, float fShift) {
	float2  temp;
	temp.x = f2InCoords.x*fSize.x+fShift;
	temp.y = temp.x + 1;	
	float4 f4ShiftCoords;

#define ORIG 0
#if ORIG
	f4ShiftCoords.xz = (fSize.x+temp.xy)%fSize.x;
	
	float2 a;	modf(temp/fSize.x,a);
	float2 b;	modf(-(-temp+fSize.x)/fSize.x,b);
#else
   	f4ShiftCoords.xz = fmod((fSize.x+temp.xy),fSize.x);
	
	float2 a = floor(temp/fSize.x);
	float2 b = floor(-(-temp+fSize.x)/fSize.x) + 1; // we know temp is negative here so floor+1 is correct.
#endif	

	f4ShiftCoords.yw = (temp >= 0) ? a : b;
	f4ShiftCoords.yw = f2InCoords.y*fSize.y+f4ShiftCoords.yw;

	// NEEDED FOR NV30 BECAUSE OF THE LACK OF FPTEXTURE-WRAP
	//	f4ShiftCoords = (f4ShiftCoords>=fSize.xyxy) ? f4ShiftCoords-fSize.xyxy : f4ShiftCoords;
	//	f4ShiftCoords = (f4ShiftCoords<0) ? f4ShiftCoords+fSize.xyxy : f4ShiftCoords;

	return f4ShiftCoords;
}


/////////////////////////////////////////////////////
// PIXEL- AND VERTEX-SHADER
/////////////////////////////////////////////////////

vertex2pixel vsAll(app2Vertex IN) {
	return IN;
}

vertex2pixel vsReduce(app2Vertex IN) {
    vertex2pixel OUT;
    OUT.Position = IN.Position;
    OUT.TexCoords = float4(IN.TexCoords.xy, IN.TexCoords.xy + fReduceStep);
    return OUT;
}

float4 psMultily(vertex2pixel v) : COLOR0 {
	return tex2D(sVector,v.TexCoords)*fMultiply;
}


float4 psReduceAddFirst(vertex2pixel v) : COLOR0 {
	float2 coords1 = v.TexCoords;
	float2 coords2 = v.TexCoords.zy;//float2(v.TexCoords.x+fReduceStep,v.TexCoords.y);
	float2 coords3 = v.TexCoords.xw;//float2(v.TexCoords.x,v.TexCoords.y+fReduceStep);
	float2 coords4 = v.TexCoords.zw;//v.TexCoords+fReduceStep;

	return	  tex2D(sVector,coords1)*tex2D(sVector2,coords1)
			+ tex2D(sVector,coords2)*tex2D(sVector2,coords2)
			+ tex2D(sVector,coords3)*tex2D(sVector2,coords3)
			+ tex2D(sVector,coords4)*tex2D(sVector2,coords4);
}

float4 psReduceAddRestX(vertex2pixel v) : COLOR0 {
	float2 coords1 = v.TexCoords;
	float2 coords2 = v.TexCoords.zy;//float2(v.TexCoords.x+fReduceStep,v.TexCoords.y);

	return tex2D(sLastPass,coords1)+ tex2D(sLastPass,coords2);
}

float4 psReduceAddRestY(vertex2pixel v) : COLOR0 {
	float2 coords1 = v.TexCoords;
	float2 coords2 = v.TexCoords.xw;//float2(v.TexCoords.x,v.TexCoords.y+fReduceStep);

	return tex2D(sLastPass,coords1)+tex2D(sLastPass,coords2);
}

float4 psReduceAddRest(vertex2pixel v) : COLOR0 {
	float2 coords1 = v.TexCoords;
	float2 coords2 = v.TexCoords.zy;//float2(v.TexCoords.x+fReduceStep,v.TexCoords.y);
	float2 coords3 = v.TexCoords.xw;//float2(v.TexCoords.x,v.TexCoords.y+fReduceStep);
	float2 coords4 = v.TexCoords.zw;//v.TexCoords+fReduceStep;

	return	  tex2D(sLastPass,coords1)
	        + tex2D(sLastPass,coords2)
			+ tex2D(sLastPass,coords3)
			+ tex2D(sLastPass,coords4);
}

float4 psReduceAddLast(vertex2pixel v) : COLOR0 {
	float2 coords1 = v.TexCoords;
	float2 coords2 = v.TexCoords.zy;//float2(v.TexCoords.x+f4ReduceStep.x,v.TexCoords.y);
	float2 coords3 = v.TexCoords.xw;//float2(v.TexCoords.x,v.TexCoords.y+f4ReduceStep.y);
	float2 coords4 = v.TexCoords.zw;//v.TexCoords+f4ReduceStep.xy;

	return	  tex2D(sLastPass,coords1)
	        + tex2D(sLastPass,coords2)
			+ tex2D(sLastPass,coords3)
			+ tex2D(sLastPass,coords4);
}

float4 psReduceAddLastRGBA(vertex2pixel v) : COLOR0 {
	float2 coords1 = v.TexCoords;
	float2 coords2 = v.TexCoords.zy;//float2(v.TexCoords.x+f4ReduceStep.x,v.TexCoords.y);
	float2 coords3 = v.TexCoords.xw;//float2(v.TexCoords.x,v.TexCoords.y+f4ReduceStep.y);
	float2 coords4 = v.TexCoords.zw;//v.TexCoords+f4ReduceStep.xy;

	return dot(float4(1,1,1,1),tex2D(sLastPass,coords1)+tex2D(sLastPass,coords2)+tex2D(sLastPass,coords3)+tex2D(sLastPass,coords4));
}

float4 psVectorMultiply(vertex2pixel v) : COLOR0 {
	return tex2D(sVector,v.TexCoords)*tex2D(sLastPass,v.TexCoords)*fMultiply;
}

float4 psVectorMultiplyClFloat(vertex2pixel v) : COLOR0 {
	return tex2D(sVector,v.TexCoords)*tex2D(sLastPass,v.TexCoords)*tex2D(sMultiply,float2(0,0)).x;
}

float4 psVectorAdd(vertex2pixel v) : COLOR0 {
	return (tex2D(sVector,v.TexCoords)*fMultiply)+(tex2D(sLastPass,v.TexCoords)*fMultiply2);
}

float4 psVectorAddClFloat(vertex2pixel v) : COLOR0 {
	return (tex2D(sVector,v.TexCoords)*fMultiply)+(tex2D(sLastPass,v.TexCoords)*tex2D(sMultiply,float2(0,0)).x);
}

float4 psVectorSubClFloat(vertex2pixel v) : COLOR0 {
	return (tex2D(sVector,v.TexCoords)*fMultiply)-(tex2D(sLastPass,v.TexCoords)*tex2D(sMultiply,float2(0,0)).x);
}

/*
 * Case "0" and all clUnpackedVector
 */

float4 psVectorMultiplyMatPosOnly0(vertex2pixel v) : COLOR0 {
	float2 f2ShiftCoords = computeShiftedCoordsPosOnly(v.TexCoords,fSize.z);//fShift);
	return tex2D(sLastPass,v.TexCoords)+(tex2D(sVector,v.TexCoords)*tex2D(sVector2,f2ShiftCoords/fSize.xy)*fSize.w);//fMultiply);
}

float4 psVectorMultiplyMatAllCases0(vertex2pixel v) : COLOR0 {
	float2 f2ShiftCoords = computeShiftedCoordsAllCases(v.TexCoords,fSize.z);//fShift);
	return tex2D(sLastPass,v.TexCoords)+(tex2D(sVector,v.TexCoords)*tex2D(sVector2,f2ShiftCoords/fSize.xy)*fSize.w);//fMultiply);
}

/*
 * Case "1"
 */
float4 psVectorMultiplyMatPosOnly1(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsPosOnlyP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;

	float4 f4Result    = tex2D(sLastPass,v.TexCoords);
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.rgb += f4MatVector.rgb * tex2D(sVector2,f4ShiftCoords.xy).gba;
	f4Result.a   += f4MatVector.a   * tex2D(sVector2,f4ShiftCoords.zw).r;

	return f4Result;
}

float4 psVectorMultiplyMatAllCases1(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsAllCasesP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result    = tex2D(sLastPass,v.TexCoords);
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.rgb += f4MatVector.rgb * tex2D(sVector2,f4ShiftCoords.xy).gba;
	f4Result.a   += f4MatVector.a   * tex2D(sVector2,f4ShiftCoords.zw).r;

	return f4Result;
}

/*
 * Case "2"
 */
float4 psVectorMultiplyMatPosOnly2(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsPosOnlyP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result    = tex2D(sLastPass,v.TexCoords);
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.rg += f4MatVector.rg * tex2D(sVector2,f4ShiftCoords.xy).ba;
	f4Result.ba += f4MatVector.ba * tex2D(sVector2,f4ShiftCoords.zw).rg;

	return f4Result;
}

float4 psVectorMultiplyMatAllCases2(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsAllCasesP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result    = tex2D(sLastPass,v.TexCoords);
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.rg += f4MatVector.rg * tex2D(sVector2,f4ShiftCoords.xy).ba;
	f4Result.ba += f4MatVector.ba * tex2D(sVector2,f4ShiftCoords.zw).rg;

	return f4Result;
}

/*
 * Case "3"
 */
float4 psVectorMultiplyMatPosOnly3(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsPosOnlyP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result    = tex2D(sLastPass,v.TexCoords);
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.r   += f4MatVector.r * tex2D(sVector2,f4ShiftCoords.xy).a;
	f4Result.gba += f4MatVector.gba * tex2D(sVector2,f4ShiftCoords.zw).rgb;

	return f4Result;
}

float4 psVectorMultiplyMatAllCases3(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsAllCasesP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result    = tex2D(sLastPass,v.TexCoords);
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.r   += f4MatVector.r * tex2D(sVector2,f4ShiftCoords.xy).a;
	f4Result.gba += f4MatVector.gba * tex2D(sVector2,f4ShiftCoords.zw).rgb;

	return f4Result;
}

//////////

/*
 * Case "0" and all clUnpackedVector
 */

float4 psVectorMultiplyMatPosOnly0_noadd(vertex2pixel v) : COLOR0 {
	float2 f2ShiftCoords = computeShiftedCoordsPosOnly(v.TexCoords,fSize.z/*fshift*/);
	return tex2D(sVector,v.TexCoords)*tex2D(sVector2,f2ShiftCoords/fSize.xy)*fSize.w;//fMultiply;
}

float4 psVectorMultiplyMatAllCases0_noadd(vertex2pixel v) : COLOR0 {
	float2 f2ShiftCoords = computeShiftedCoordsAllCases(v.TexCoords,fSize.z/*fshift*/);
	return tex2D(sVector,v.TexCoords)*tex2D(sVector2,f2ShiftCoords/fSize.xy)*fSize.w;//fMultiply;
}

/*
 * Case "1"
 */
float4 psVectorMultiplyMatPosOnly1_noadd(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsPosOnlyP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result;
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.rgb = f4MatVector.rgb * tex2D(sVector2,f4ShiftCoords.xy).gba;
	f4Result.a   = f4MatVector.a   * tex2D(sVector2,f4ShiftCoords.zw).r;

	return f4Result;
}

float4 psVectorMultiplyMatAllCases1_noadd(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsAllCasesP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result;
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.rgb = f4MatVector.rgb * tex2D(sVector2,f4ShiftCoords.xy).gba;
	f4Result.a   = f4MatVector.a   * tex2D(sVector2,f4ShiftCoords.zw).r;

	return f4Result;
}

/*
 * Case "2"
 */
float4 psVectorMultiplyMatPosOnly2_noadd(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsPosOnlyP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result;
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.rg = f4MatVector.rg * tex2D(sVector2,f4ShiftCoords.xy).ba;
	f4Result.ba = f4MatVector.ba * tex2D(sVector2,f4ShiftCoords.zw).rg;

	return f4Result;
}

float4 psVectorMultiplyMatAllCases2_noadd(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsAllCasesP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result;
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.rg = f4MatVector.rg * tex2D(sVector2,f4ShiftCoords.xy).ba;
	f4Result.ba = f4MatVector.ba * tex2D(sVector2,f4ShiftCoords.zw).rg;

	return f4Result;
}

/*
 * Case "3"
 */
float4 psVectorMultiplyMatPosOnly3_noadd(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsPosOnlyP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result;
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.r   = f4MatVector.r * tex2D(sVector2,f4ShiftCoords.xy).a;
	f4Result.gba = f4MatVector.gba * tex2D(sVector2,f4ShiftCoords.zw).rgb;

	return f4Result;
}

float4 psVectorMultiplyMatAllCases3_noadd(vertex2pixel v) : COLOR0 {
	float4 f4ShiftCoords   = computeShiftedCoordsAllCasesP1(v.TexCoords,fSize.z/*fShift*/)/fSize.xyxy;
	
	float4 f4Result;
	float4 f4MatVector = tex2D(sVector,v.TexCoords)*fSize.w;//fMultiply;

	f4Result.r   = f4MatVector.r * tex2D(sVector2,f4ShiftCoords.xy).a;
	f4Result.gba = f4MatVector.gba * tex2D(sVector2,f4ShiftCoords.zw).rgb;

	return f4Result;
}

//////////

float4 psUnpackVector(vertex2pixel v) : COLOR{

	float4 result;
	float2 coords = float2(v.TexCoords.x/2,v.TexCoords.y);
	
	if (round((v.TexCoords.y*fSize.y*2)%2)==1) coords.x += 0.5;
	
	float4 vecValue = tex2D(sVector,coords);
	
	if (round((v.TexCoords.x*fSize.x*2)%4) == 3)
		result = float4(vecValue.a,0,0,0);
	else if (round((v.TexCoords.x*fSize.x*2)%4) == 2)
		result = float4(vecValue.b,0,0,0);
	else if (round((v.TexCoords.x*fSize.x*2)%4) == 1)
		result = float4(vecValue.g,0,0,0);
	else 
		result = float4(vecValue.r,0,0,0);	
	
	return result;
}


float4 psPackVector(vertex2pixel v) : COLOR {

	// remark: using different constant definitions in this function
	// fShift  = packed size in X
	// fSize.x = unpacked size in X
	// fSize.y = unpacked size in Y
	// fSize.b = half the packed size in X (=fSize.x/2)
	// fSize.a = 1/(unpacked size in X)

	float4 result;

	float2 coords = float2(round((v.TexCoords.x*fShift)%fSize.b)*4.0, floor(round(v.TexCoords.x*fShift)/fSize.b)+(v.TexCoords.y*fSize.y));

	// rescale back to [0..1] in unpacked coords
	coords /= fSize.xy;

	result.r = tex2D(sVector,coords).r;	coords.x += fSize.a;
	result.g = tex2D(sVector,coords).r;	coords.x += fSize.a;
	result.b = tex2D(sVector,coords).r;	coords.x += fSize.a;
	result.a = tex2D(sVector,coords).r;

	return result;
}

/////////////////////////////////////////////////////
// TECHNIQUES
/////////////////////////////////////////////////////

technique tMultiplyScal {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psMultily();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tReduceAddFirst {
	pass P0 {
		VertexShader = compile vs_1_1 vsReduce();
		PixelShader  = compile PS_PROFILE psReduceAddFirst();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tReduceAddRest {
	pass P0 {
		VertexShader = compile vs_1_1 vsReduce();
		PixelShader  = compile PS_PROFILE psReduceAddRest();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tReduceAddLast {
	// for r-only encoded (unpacked)
	pass P0 {
		VertexShader = compile vs_1_1 vsReduce();
		PixelShader  = compile PS_PROFILE psReduceAddLast();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
    
    // for RGBA encoded (packed)
	pass P1 {
		VertexShader = compile vs_1_1 vsReduce();
		PixelShader  = compile PS_PROFILE psReduceAddLastRGBA();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

}

technique tReduceAddRestX {
	pass P0 {
		VertexShader = compile vs_1_1 vsReduce();
		PixelShader  = compile PS_PROFILE psReduceAddRestX();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;
    }   
}

technique tReduceAddRestY {
	pass P0 {
		VertexShader = compile vs_1_1 vsReduce();
		PixelShader  = compile PS_PROFILE psReduceAddRestY();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tVectorAdd {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorAdd();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P1 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorAddClFloat();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P2 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorSubClFloat();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tVectorMultiply {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiply();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
    
	pass P1 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyClFloat();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tVectorMultiplyMatPosOnly {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatPosOnly0();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P1 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatPosOnly1();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P2 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatPosOnly2();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P3 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatPosOnly3();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tVectorMultiplyMatAllCases {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatAllCases0();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P1 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatAllCases1();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P2 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatAllCases2();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P3 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatAllCases3();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}


technique tVectorMultiplyMatAllCases_noadd {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatAllCases0_noadd();

		CullMode			= NONE; 
		ZWriteEnable		= FALSE;
		ZEnable				= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P1 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatAllCases1_noadd();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P2 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatAllCases2_noadd();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P3 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatAllCases3_noadd();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}


technique tVectorMultiplyMatPosOnly_noadd {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatPosOnly0_noadd();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P1 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatPosOnly1_noadd();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P2 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatPosOnly2_noadd();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   

	pass P3 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psVectorMultiplyMatPosOnly3_noadd();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tUnpackVector {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psUnpackVector();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}

technique tPackVector {
	pass P0 {
		VertexShader = compile vs_1_1 vsAll();
		PixelShader  = compile PS_PROFILE psPackVector();

		CullMode			= NONE; 
		ZEnable				= FALSE;
		ZWriteEnable		= FALSE;
		AlphaBlendEnable	= FALSE;				
    }   
}