//Copyright and Disclaimer:
//This code is copyright Vienna University of Technology, 2004.

#include "SceneData.h"
#include <Mathematic/Vector3.h>
#include <Mathematic/Vector4.h>
#include <Mathematic/perlin.h>
#include <Mathematic/MathTools.h>
#include <Engine/Renderer/State.h>

using Math::Vector3f;

GLenum texHeight = 0;

void Scene::makeHeightTexture(void) {
	const unsigned SIZE = 8;
	GLubyte heightTexture[SIZE*3];

	/* Setup RGB image for the texture. */
	GLubyte *loc = (GLubyte*) heightTexture;
	// deep water. 
	loc[0] = 0x00;
	loc[1] = 0x00;
	loc[2] = 0xff;
	loc += 3;

	loc[0] = 0x00;
	loc[1] = 0xaf;
	loc[2] = 0xff;
	loc += 3;

	// sand.
	loc[0] = 0xa0;
	loc[1] = 0xff;
	loc[2] = 0xa0;
	loc += 3;

	// green. 
	loc[0] = 0x00;
	loc[1] = 0x8f;
	loc[2] = 0x00;
	loc += 3;

	// green. 
	loc[0] = 0x00;
	loc[1] = 0xaf;
	loc[2] = 0x00;
	loc += 3;

	// gray. 
	loc[0] = 0x6f;
	loc[1] = 0x6f;
	loc[2] = 0x6f;
	loc += 3;

	loc[0] = 0xaf;
	loc[1] = 0xaf;
	loc[2] = 0xaf;

	glTexImage1D(GL_TEXTURE_1D,0,GL_RGB,SIZE,0,GL_RGB,GL_UNSIGNED_BYTE,heightTexture);
	glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_WRAP_S,GL_CLAMP);
}

inline unsigned Scene::getID(const unsigned u, const unsigned v, const unsigned mapSize) const  { 
//	assert(u < mapSize);
//	assert(v < mapSize);
	return u*mapSize+v;
}

void Scene::createFaceTriangleStrip(ArrID& vertexID, const unsigned mapSize,
							 const unsigned U, const unsigned V,
							 const unsigned deltaU, const unsigned deltaV) {
	unsigned U2 = U+deltaU;
	if(mapSize < U2)
		U2 = mapSize;
	unsigned V2 = V+deltaV;
	if(mapSize < V2)
		V2 = mapSize;
	for(unsigned u = U; u < U2-1; u++) {
		//to join the row triangle-strips send first and last vertex of each row twice 
		//-> 4 degenerate triangels are created
		//don't send first vertex twice, because of orientation
		if(U < u) vertexID.append(getID(u+1,V,mapSize));
		for(unsigned v = V; v < V2; v++) {
			vertexID.append(getID(u+1,v,mapSize));
			vertexID.append(getID(u,v,mapSize));
		}
		vertexID.append(getID(u,V2-1,mapSize));
	}
}


void Scene::createHeightFieldObjects(const DynVector3& vertex, const unsigned mapSize, const unsigned objVertexDelta) {
	//create triangle indices
	const unsigned delta = Math::clamp(objVertexDelta,2u,mapSize);
	const unsigned step = delta-1;
	const unsigned big = mapSize-1;
	for(unsigned u = 0; u < big; u += step) {
		for(unsigned v = 0; v < big; v += step) {
			ArrID vertexID;
			createFaceTriangleStrip(vertexID,mapSize,u,v,delta,delta);
			SmpGeometry hf(new HeightField(
				heightFieldState
//				heightFieldStateShader
				,vertexID,vertex));
			scene.push_back(hf);
			caster.push_back(hf);
		}
	}
}

void Scene::createHeightField(const DynByte& heightMap, const unsigned mapSize, const float& scale, const float& sY) {
	//create vertices
	vertex.resize(mapSize*mapSize);
	texCoord.resize(vertex.size());
	const float f = 1.0/(mapSize-1);
	const float fY = 1.0/255;
	for(unsigned u = 0; u < mapSize; u++) {
		for(unsigned v = 0; v < mapSize; v++) {
			const unsigned id = u*mapSize+v;
			const float x = u*f-0.5;
			const float z = v*f-0.5;
			const float y = Math::clamp<float>(fY*heightMap[id],0.0,1.0);
			texCoord[id] = y;
			vertex[id] = Vector3f(scale*x,sY*(y-0.5),scale*z);
		}
	}

	ArrID vertexID;
	createFaceTriangleStrip(vertexID,mapSize,0,0,mapSize,mapSize);
    glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3,GL_FLOAT,0,&vertex);
	glTexCoordPointer(1,GL_FLOAT,0,&texCoord);

	normal.resize(vertex.size());
	normal.fillWith(Vector3f::ZERO);
	for(unsigned i = 0; i < vertexID.size()-2; i ++) {
		const Math::Vector3f a = vertex[vertexID[i]]-vertex[vertexID[i+1]];
		const Math::Vector3f b = vertex[vertexID[i+1]]-vertex[vertexID[i+2]];
		Vector3f fn;
		fn.unitCross(a,b);
		if(fn.dot(Vector3f::UNIT_Y) < 0.0)
			fn.invert();
		normal[vertexID[i]] += fn;
		normal[vertexID[i+1]] += fn;
		normal[vertexID[i+2]] += fn;
	}
	for(unsigned i = 0; i < normal.size(); i++) {
		normal[i].normalize();
	}
	glNormalPointer(GL_FLOAT,0,&normal);
}

void Scene::createTrees(const DynByte& heightMap, const DynVector3& v, const unsigned count, const unsigned treeSize, const unsigned MAP_SIZE, const float terrainSize) {
	if(v.empty()) return;
	if(heightMap.size() == 0) return;
	const double identMap = (1.0/RAND_MAX);
	const unsigned step1 = Math::clamp<unsigned>(1,MAP_SIZE/(terrainSize/treeSize),MAP_SIZE/2);
	const unsigned step2 = MAP_SIZE/Math::sqrt(double(count));
	const unsigned step = (step1 < step2)? step1:step2;

	ArrID treeVertexID;
	srand(heightMap[0]);
	const int maxVar = step/4;
	for(unsigned u = step; u < MAP_SIZE-step; u += step) {
		for(unsigned v = step; v < MAP_SIZE-step; v += step) {
			const int var = Math::randRange(-maxVar,maxVar);
			treeVertexID.append(getID(u+var,v+var,MAP_SIZE));
		}
	}
	Math::randomizeArray(treeVertexID);
	if(step1 < step2) 
		treeVertexID.resize(count);
	for(unsigned i = 0; i < treeVertexID.size(); i++) {
		SmpGeometry tree(new Tree(treeState,v[treeVertexID[i]],(rand()+1)*identMap*treeSize,baseTree));
		scene.push_back(tree);
		caster.push_back(tree);
	}
}


Scene::Scene(): sceneAABox(V3::ZERO,V3::ZERO), baseTree(treeState) {
	const float HEIGHT = 15.0;
	const float SIZE = 200.0;
	const float TREE_SIZE = 5.0;

	//height map extent
	const unsigned MAP_SIZE = 1024;

	glGenTextures(1,&texHeight);
	glBindTexture(GL_TEXTURE_1D,texHeight);
	makeHeightTexture();

	DynByte heightMap; // Holds The Height Map Data
	heightMap.resize(MAP_SIZE*MAP_SIZE);
	std::cout << "creating noise textures\n";
	Perlin::Tuppel4 *t = Perlin::createFieldWithOctave123(MAP_SIZE,MAP_SIZE,7);
	for(unsigned i = 0; i < heightMap.size(); i++) {
		Perlin::BYTE* noise = t[i];
		float v = abs(noise[0]/256.0);
		v += abs(noise[1]/256.0*0.5);
		v += abs(noise[2]/256.0*0.25);
		v += abs(noise[3]/256.0*0.125);
		v *= 1.1;
		v -= 0.5;
		heightMap[i] = v*256;
	}
	delete []t;

	std::cout << "creating height field\n";
	createHeightField(heightMap,MAP_SIZE,SIZE,HEIGHT);
	std::cout << "creating height field objects\n";
    createHeightFieldObjects(vertex,MAP_SIZE,30);
	std::cout << "creating trees\n";
	createTrees(heightMap,vertex,10000,TREE_SIZE,MAP_SIZE,SIZE);
	
	sceneAABox = scene.front()->getBoundingBox().convert2<double>();
	triangleCount = 0;
	for(Objects::iterator i = scene.begin(); i != scene.end(); i++) {
		sceneAABox += (*i)->getBoundingBox().convert2<double>();
		triangleCount += (*i)->triangleCount();
	}
	std::cout << "scene bounding box:" << sceneAABox << '\n';

	std::cout << "creating KD-Tree\n";
	int delta = glutGet(GLUT_ELAPSED_TIME);
	cull = new Occlusion(scene);
	delta = glutGet(GLUT_ELAPSED_TIME)-delta;
	std::cout << "kdtree build time:" << delta << "ms for: " << scene.size() << " objects\n";
}

Scene::~Scene() {
	delete cull;
	scene.clear();
	caster.clear();
	vertex.clear();
	normal.clear();
	texCoord.clear();
}


void Scene::draw(FrameState& fs) {
	switch(fs.mode) {
		case 0: cull->viewFrustumCulling(fs); break;
		case 1: cull->stopAndWait(fs); break;
		case 2: cull->CHCtraversal(fs); break;
	};
	fs.stateManager.setState(State::EMPTY);
}

void Scene::drawKDTree(FrameState& fs) {
	cull->KDTreeOnly(fs);
	fs.stateManager.setState(State::EMPTY);
}

void Scene::drawShadowCasters(FrameState& fs) {
	for(Objects::iterator i = caster.begin(); i != caster.end(); i++) {
		(*i)->drawGeometryCulled(fs);
	}
}

const Math::Vector3f::ElementType Scene::getHeight(const Math::Vector3f& pos) const {
	const unsigned MAP_SIZE = Math::sqrt(double(vertex.size()));
	const unsigned MAP_SIZE_1 = MAP_SIZE-1;
	//with scenebb get u,v of equidistant grid
	const Vector3f min = sceneAABox.getMin().convert2<float>();
	const Vector3f max = sceneAABox.getMax().convert2<float>();
	const float x_norm = (pos[0]-min[0])/(max[0]-min[0]);
	const float z_norm = (pos[2]-min[2])/(max[2]-min[2]);
	const float uf = x_norm*MAP_SIZE_1;
	const float vf = z_norm*MAP_SIZE_1;
	const unsigned u0 = Math::clamp<unsigned>(Math::floor(uf),0,MAP_SIZE_1);
	const unsigned v0 = Math::clamp<unsigned>(Math::floor(vf),0,MAP_SIZE_1);
	const unsigned u1 = Math::clamp<unsigned>(Math::ceil(uf),0,MAP_SIZE_1);
	const unsigned v1 = Math::clamp<unsigned>(Math::ceil(vf),0,MAP_SIZE_1);
	//lookup with getID in vertex arrray
	const unsigned id0 = getID(u0,v0,MAP_SIZE);
	const unsigned id1 = getID(u0,v1,MAP_SIZE);
	const unsigned id2 = getID(u1,v0,MAP_SIZE);
	const unsigned id3 = getID(u1,v1,MAP_SIZE);
	const Vector3f& vert0 = vertex[id0];
	const Vector3f& vert1 = vertex[id1];
	const Vector3f& vert2 = vertex[id2];
	const Vector3f& vert3 = vertex[id3];
	//bilinearly interpolate
	const float fractV = Math::fract<float>(vf);
	const float fractU = Math::fract<float>(uf);
	const float a = Math::lerp<float>(vert0[1],vert1[1],fractV);
	const float b = Math::lerp<float>(vert2[1],vert3[1],fractV);
	return Math::lerp(a,b,fractU);
}
