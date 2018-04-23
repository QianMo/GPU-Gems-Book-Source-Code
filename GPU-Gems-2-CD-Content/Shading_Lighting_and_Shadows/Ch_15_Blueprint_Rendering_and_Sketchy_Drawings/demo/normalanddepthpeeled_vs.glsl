//gl2
//No: bp_v1_cGL2_pb

varying vec3 perVertexNormals;
varying float depthInCamera;

void main(void) {
	// Compute and submit per vertex normals
    perVertexNormals = normalize(gl_NormalMatrix * gl_Normal);
    
    // Compute and submit eye space depth in the range [0,1]
    vec4 position    = gl_ModelViewMatrix * gl_Vertex;
    depthInCamera    = (position.z - gl_DepthRange.near) / (gl_DepthRange.diff);
    
    // Submit transformed vertices
    gl_Position = ftransform();
}
