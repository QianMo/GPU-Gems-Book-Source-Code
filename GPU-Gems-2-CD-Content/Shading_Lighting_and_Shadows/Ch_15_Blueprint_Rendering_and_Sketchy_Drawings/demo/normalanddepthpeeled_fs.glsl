//gl2
//No: bp_v1_cGL2_pb

uniform sampler2D preDepthLayer;
uniform vec2 windowDimension;

varying vec3 perVertexNormals;
varying float depthInCamera;

void main (void) {

	// Determine previous depth value 
	vec4 center = gl_TextureMatrix[0] * vec4(gl_FragCoord.x / windowDimension.x, gl_FragCoord.y / windowDimension.y, 0.0, 0.0);
	float preDepth = float(texture2D(preDepthLayer, center.xy));

	// Apply additional depth test to peel away depth layers
	if(gl_FragCoord.z<=preDepth) {
		discard;
	}
	
	// Normalized per fragment normals
	vec3 normal = normalize(perVertexNormals);
	
	// Encode normals: [-1,1] => [0,1]
    normal = (normal+1.0)*0.5;

	// Output color and depth 
    gl_FragColor = vec4(normal, depthInCamera);
    gl_FragDepth = gl_FragCoord.z;
}
