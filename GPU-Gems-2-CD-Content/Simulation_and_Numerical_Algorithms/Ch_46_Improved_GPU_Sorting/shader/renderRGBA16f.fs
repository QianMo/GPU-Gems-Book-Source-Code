// -*- C++ -*- automatisch in C++mode wechseln (emacs)

uniform float MaxValue;
uniform sampler2DRect Data;

#define OwnPos   gl_TexCoord[0]

void main(void)
{
    // get self
    vec4 self = texture2DRect(Data, OwnPos.xy);

    // rescale 
    float level = self.x / MaxValue;

    // turn key into color
    gl_FragColor = vec4(level,level,1.0,1.0);
}
