// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes for lights, materials, and objects

#if !defined(LIGHTS_MATERIALS_OBJECTS_H)
#define LIGHTS_MATERIALS_OBJECTS_H

class BaseDocument;
class BaseObject;
class BaseTag;
class FXWrapper;
class BaseMaterial;
class FXEmulator;
class ShadowMaps;
class CrawlHierarchy;
class PolygonObject;
struct Matrix;

class Lights
{
public:
	Lights(BaseDocument* doc);
	~Lights(void); // do not inherit
	int GetNumLights(void) {return numLights;}
	int GetNumShadowLights(void) {return numShadowLights;}
	const float* GetPos(int n) {return pos[n];}
	const float* GetDir(int n) {return dir[n];}
	const float* GetUp(int n) {return up[n];}
	const float* GetSide(int n) {return side[n];}
	int GetLightFromShadow(int s) {return lightFromShadow[s];}
	int GetShadowFromLight(int n) {return shadowFromLight[n];}
	BaseObject* GetLight(int n) {return lights[n];}
	bool HasShadowMap(int n) {return hasShadowMap[n];}
private:
	BaseObject* CrawlScene(BaseObject* b);
	void GetLights(void);
	BaseDocument* document;
	int numLights;
	int numShadowLights;
	BaseObject** lights;
	float (*pos)[4];
	float (*dir)[4];
	float (*up)[4];
	float (*side)[4];
	int* lightFromShadow;
	int* shadowFromLight;
	bool* hasShadowMap;
	bool valid;
};

class Materials
{
public:
	Materials(BaseDocument* doc, Lights* lights, int mapSize, ShadowMaps* sm);
	~Materials(void); // do not inherit
	bool GetAndCheckFXWrapper(BaseMaterial* bm, FXWrapper* &fx);
	void BeginRendering(void);
	void EndRendering(void);
private:
	BaseDocument* document;
	int size;
	FXWrapper** materialList;
	BaseMaterial** baseMaterialList;
	FXEmulator** emulatorMaterialList;
	int numMaterials;
	unsigned char* texturePic;
	Lights* lig;
	ShadowMaps* shadows;
};

class ObjectIterator
{
	friend CrawlHierarchy;
public:
	ObjectIterator(BaseDocument* doc, Materials* m, void(*f)(ObjectIterator*));
	~ObjectIterator(void); // do not inherit
	BaseDocument* GetDocument(void) {return document;}
	BaseMaterial* GetMaterial(void) {return material;}
	bool GetObjectName(char* s) const;
	bool GetPolys(float* &vert, long& numVert, long* &poly, long& numPoly);
	bool GetNormalsEtc(float* normals, float* tangents, float* binormals, float* texCoords, bool useOrthoFrame) const;
	bool GetWorldMatrix(float* m) const;
	bool GetAndCheckFXWrapper(FXWrapper* &fx) const;
	bool ReadFX(void);
	void Print(const char* s1, const char* s2) const;
	bool GetParamDirectionOrPosition(bool isPosition, bool isWorld, int index, float* v) const;
	bool CastsShadow(void) const;
private:
	BaseDocument* document;
	PolygonObject* object;
	Matrix* gMatrix;
	void (*func)(ObjectIterator*);
	BaseMaterial* material;
	BaseTag* compositing;
	float* vert;
	long numVert;
	long* poly;
	long numPoly;
	Materials* mats;
};

#endif