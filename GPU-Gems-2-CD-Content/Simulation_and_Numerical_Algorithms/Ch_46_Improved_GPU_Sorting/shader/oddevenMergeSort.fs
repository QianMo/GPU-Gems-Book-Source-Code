// -*- C++ -*- automatisch in C++mode wechseln (emacs)

uniform vec3      Param1;
uniform vec3      Param2;
uniform sampler2DRect Data;

#define OwnPos         gl_TexCoord[0]

// contents of the uniform data fields
#define TwoStage       Param1.x
#define Pass_mod_Stage Param1.y
#define TwoStage_PmS_1 Param1.z
#define Width          Param2.x
#define Height         Param2.y
#define Pass           Param2.z

void main(void)
{
    // get self
    vec4 self = texture2DRect(Data, OwnPos.xy);
    float i = floor(OwnPos.x) + floor(OwnPos.y) * Width; 

    // my position within the range to merge
    float j = floor(mod(i,TwoStage));

    float compare;

    if ( (j<Pass_mod_Stage) || (j>TwoStage_PmS_1) ) 
	// must copy -> compare with self
	compare = 0.0;
    else
	// must sort
	if ( mod((j+Pass_mod_Stage) / Pass,2.0) < 1.0)
	    // we are on the left side -> compare with partner on the right
	    compare = 1.0;
	else
	    // we are on the right side -> compare with partner on the left
	    compare = -1.0;

    // get the partner
    float adr = i + compare*Pass;
    vec4 partner = texture2DRect(Data, vec2( floor(mod(adr,Width)),
					     floor(adr / Width) ) );

    // on the left its a < operation, on the right its a >= operation
    gl_FragColor = (self.x*compare < partner.x*compare) ? self : partner;
}
