//gl2
//No: bp_v1_cGL2_pb

varying vec3 perVertexNormals;
varying float depthInCamera;

void main (void) {
	// Normalized per fragment normals
	vec3 normal = normalize(perVertexNormals);
	
	// Encode normals: [-1,1] => [0,1]
    normal = (normal+1.0)*0.5;

	// Output color and depth 
    gl_FragColor = vec4(normal, depthInCamera);
    gl_FragDepth = gl_FragCoord.z;
}
