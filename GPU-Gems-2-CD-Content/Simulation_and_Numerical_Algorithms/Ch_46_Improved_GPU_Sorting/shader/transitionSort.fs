// -*- C++ -*- automatisch in C++mode wechseln (emacs)

uniform vec3      Param;
uniform sampler2DRect Data;

#define OwnPos   gl_TexCoord[0]

// contents of the uniform data fields
#define Width    Param.x
#define Height   Param.y
#define Odd      Param.z

void main(void)
{
    // get self
    vec4 self = texture2DRect(Data, OwnPos.xy);
    float i = floor(OwnPos.x) + floor(OwnPos.y) * Width; 
    bool selfisodd = bool(mod(i,2.0));

    float compare;

    // invert the compare if we're on the "odd" sorting pass
    if (selfisodd)
	// self is odd -> compare with right key
	compare = Odd;
    else
	// self is even -> compare with left key
	compare = -Odd;

    // correct the special case that the "odd" pass copies the first and the last key
    if ( (Odd > 0.0) && ((i==0.0) || (i==((Width*Height)-1.0))) ) 
	// must copy -> compare with self
	compare = 0.0;

    // get the partner
    float adr = i + compare;
    vec4 partner = texture2DRect(Data, vec2( floor(mod(adr,Width)),
					     floor(adr / Width)) );

    // on the left its a < operation, on the right its a >= operation
    gl_FragColor = (self.x*compare < partner.x*compare) ? self : partner;
}
