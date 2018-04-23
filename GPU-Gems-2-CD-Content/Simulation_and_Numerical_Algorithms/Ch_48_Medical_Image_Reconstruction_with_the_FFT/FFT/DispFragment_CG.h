/***************************************************************************
*        FILE NAME:  DispFragment_CG.h
*
* ONE LINE SUMMARY:
*        This file contains the shader for displaying the final image
*        
*        Thilaka Sumanaweera
*        Siemens Medical Solutions USA, Inc.
*        1230 Shorebird Way
*        Mountain View, CA 94039
*        USA
*        Thilaka.Sumanaweera@siemens.com
*
* DESCRIPTION:
*
*****************************************************************************
* REVISION HISTORY:
* Rev     When      Who         What
* V1      15Dec2004 Thilaka     Created.
****************************************************************************/
const char *DispFragment_CG = "\n"
"void FragmentProgram(in float4 TexCoord : TEXCOORD0,\n"
"                     out float4 sColor0 : COLOR0,\n"
"                     uniform float4 InvEnergy,\n"
"                     const uniform samplerRECT Real1,\n"
"                     const uniform samplerRECT Imag1,\n"
"                     const uniform samplerRECT Real2,\n"
"                     const uniform samplerRECT Imag2)\n"
"{\n"
"	float4 R1 = texRECT(Real1, TexCoord.xy)*InvEnergy.xxxx;\n"
"	float4 I1 = texRECT(Imag1, TexCoord.xy)*InvEnergy.yyyy;\n"
"	float4 R2 = texRECT(Real2, TexCoord.xy)*InvEnergy.zzzz;\n"
"	float4 I2 = texRECT(Imag2, TexCoord.xy)*InvEnergy.wwww;\n"
"\n"
"	float4 val = float4(R1.x, I1.x, R2.x, I2.x);\n"
"\n"
"	sColor0.x = sqrt(dot(val, val));\n"
"	sColor0.y = sColor0.x;\n"
"	sColor0.z = sColor0.x;\n"
"	sColor0.w = 1.0;\n"
"}\n"
"\n";

