// -*- C++ -*- automatisch in C++mode wechseln (emacs)

uniform sampler2DRect PackedData;

// contents of the texcoord data
#define OwnPos          gl_TexCoord[0].xy
#define SearchDir       gl_TexCoord[0].z
#define CompOp          gl_TexCoord[0].w
#define Distance        gl_TexCoord[1].x
#define Stride          gl_TexCoord[1].y
#define Height          gl_TexCoord[1].z
#define HalfStrideMHalf gl_TexCoord[1].w

void main(void)
{
    // get self
    vec4 self = texture2DRect(PackedData, OwnPos);

    // restore sign of search direction and assemble vector to partner
    vec2 adr = vec2 ( (SearchDir < 0.0) ? -Distance : Distance , 0.0);

    // get the partner
    vec4 partner = texture2DRect(PackedData, OwnPos + adr);

    // switch ascending/descending sort for every other row
    // by modifying comparison flag
    float compare = CompOp * -(mod(floor(gl_TexCoord[0].y), Stride) - HalfStrideMHalf);

    // x and y are the keys of the two items
    // --> multiply with comparison flag
    vec4 keys = compare * vec4( self.x, self.y, partner.x, partner.y);

    // compare the keys and store accordingly
    // z and w are the indices
    // --> just copy them accordingly
    vec4 result;
    result.xz = (keys.x < keys.z) ? self.xz : partner.xz;
    result.yw = (keys.y < keys.w) ? self.yw : partner.yw;

    gl_FragColor = result;

    // uncomment one of these to see the data created by the rasterizer
//     gl_FragColor = (SearchDir < 0.0) ? vec4(1,0,0,1) : vec4(0,1,0,1);
//     gl_FragColor = (CompOp < 0.0) ? vec4(1,0,0,1) : vec4(0,1,0,1);
//     gl_FragColor = vec4(gl_TexCoord[0].x,gl_TexCoord[0].y,0,1);
//     gl_FragColor = vec4(gl_FragCoord.x/Height, gl_FragCoord.y/Height, 0,1);
}
