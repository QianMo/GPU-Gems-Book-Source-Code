//gl2
//No: bp_v1_cGL2_pb

uniform sampler2D edgeMap;
uniform sampler2D depthMap;
uniform vec2 windowDimension;
uniform float bluishValue;

void main (void) {
	vec4 tex = vec4(gl_FragCoord.x / windowDimension.x, gl_FragCoord.y / windowDimension.y, 1.0, 1.0);
	tex = gl_TextureMatrix[0] * tex;

	// Depth texture access
	float depthValue    = float(texture2D(depthMap, tex.xy).xyz);

	// Discard in advance
	if(depthValue == 1.0) {
		discard;
	} 

	// Output depth for depth sprite rendering
	gl_FragDepth = depthValue;

	vec3 edgeIntensity = texture2D(edgeMap, tex.xy).xyz;
	//if(edgeIntensity.z>0.95) {
	//    discard;
	//}

	vec3 color = edgeIntensity.z * vec3(0.93, 0.93, bluishValue);

	gl_FragColor = vec4(color, edgeIntensity.z);
}
