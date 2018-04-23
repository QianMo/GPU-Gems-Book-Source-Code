//gl2
//No: bp_v1_cGL2_pb

uniform sampler2D gBuffers;
uniform vec2 textureCoordScale;
uniform vec2 windowDimension;

void main (void) {

	// Texture Center and Offset
	vec2 center = vec2(gl_FragCoord.x / windowDimension.x, gl_FragCoord.y / windowDimension.y);
	vec2 off    = vec2((1.0/windowDimension.x)*textureCoordScale.x, (1.0/windowDimension.y)*textureCoordScale.y);

	// North west texture access
	vec4 tex_nw = vec4(center.x+off.x, center.y+off.y, 1.0, 1.0);
	tex_nw = gl_TextureMatrix[0] * tex_nw;
	vec4 val_nw = texture2D(gBuffers, tex_nw.xy);
	val_nw.xyz = (val_nw.xyz *2.0)-1.0;
	
	// North east texture access
	vec4 tex_ne = vec4(center.x-off.x, center.y+off.y, 1.0, 1.0);
	tex_ne = gl_TextureMatrix[0] * tex_ne;
	vec4 val_ne = texture2D(gBuffers, tex_ne.xy);
	val_ne.xyz = (val_ne.xyz *2.0)-1.0;

	// South west texture access
	vec4 tex_sw = vec4(center.x+off.x, center.y-off.y, 1.0, 1.0);
	tex_sw = gl_TextureMatrix[0] * tex_sw;
	vec4 val_sw = texture2D(gBuffers, tex_sw.xy);
	val_sw.xyz = (val_sw.xyz *2.0)-1.0;

	// South east texture access
	vec4 tex_se = vec4(center.x-off.x, center.y-off.y, 1.0, 1.0);
	tex_se = gl_TextureMatrix[0] * tex_se;
	vec4 val_se = texture2D(gBuffers, tex_se.xy);
	val_se.xyz = (val_se.xyz *2.0)-1.0;
	
	// Calculate discontinuities
	vec3 discontinuity = vec3(0.0, 0.0, 0.0);

	// (north west DOT south east) AND (north east DOT south west)
	float dot0 = dot(val_nw.xyz, val_se.xyz);
	float dot1 = dot(val_ne.xyz, val_sw.xyz);
	discontinuity.x = 0.5*(dot0+dot1);

	// (north west DEPTH DISCONT. south east) AND (north east DEPTH DISCONT. south west)
	float depth_discont0 = 1.0-abs(val_nw.w - val_se.w);
	float depth_discont1 = 1.0-abs(val_ne.w - val_sw.w);
	discontinuity.y = depth_discont0*depth_discont1;
	
	discontinuity.z = discontinuity.x*discontinuity.y;

    gl_FragColor = vec4(discontinuity, 1.0);
}
