// from C4Dfx by Jörn Loviscach, www.l7h.cn
// functions hiding Cinema 4D functionality

#if !defined(C4DWRAPPER_H)
#define C4DWRAPPER_H

class FXWrapper;
class PolygonObject;
struct Matrix;
class BaseMaterial;
class CrawlHierarchy;
class BaseDocument;
class NodeData;
class FXEmulator;
class BaseObject;
class Description;
class BaseContainer;
class ShadowMaps;
class GeDialog;

namespace C4DWrapper  
{
	void MsgBox(const char* s);
	void MsgBox(const char* s1, const char* s2);

	void Print(const char* s);
	void Print(const char* s1, const char* s2);

	void SetFrame(BaseDocument* doc, double f);
	long GetRenderWidth(BaseDocument* doc);
	long GetRenderHeight(BaseDocument* doc);
	long GetFps(BaseDocument* doc);
	long GetFpsRender(BaseDocument* doc);
	long GetStartFrame(BaseDocument* doc);
	long GetEndFrame(BaseDocument* doc);
	bool CheckFormat(BaseDocument* doc);
	long GetRenderPathLength(BaseDocument* doc);

	BaseDocument* CloneDoc(BaseDocument* doc);
	void FreeDoc(BaseDocument* doc);

	BaseContainer* CloneContainer(BaseContainer* c);
	void DeleteContainer(BaseContainer* c);

	bool GetMaterialBool(BaseMaterial* material, long id, bool* val);
	bool GetMaterialVector(BaseMaterial* material, long id, float* vec);
	bool GetMaterialFloat(BaseMaterial* material, long id, float* val);
	bool GetMaterialTexture(BaseMaterial* material, long id, unsigned char* pic, int size);
	bool GetMaterialSpecShapeTexture(BaseMaterial* material, unsigned char* pic, int size);
	bool GetMaterialNormalTexture(BaseMaterial* material, unsigned char* pic, int size);
	bool GetMaterialEnvTexture(BaseMaterial* material, unsigned char* pic, int size, int dir);
	long GetChannelIdColor();

	bool GetObjectLong(BaseObject* obj, long id, long* val);
	bool GetObjectBool(BaseObject* obj, long id, bool* val);
	bool GetObjectFloat(BaseObject* obj, long id, float* val);
	bool GetObjectVector(BaseObject* obj, long id, float* vec);

	bool GetViewMatrix(BaseDocument* doc, float* m);
	bool GetProjMatrix(BaseDocument* doc, float* m);
	bool GetParamFloat(BaseMaterial* m, int i, float &a);
	bool GetParamVector(BaseMaterial* m, int i, float* v);
	bool GetParamFilename(BaseMaterial* m, int i, char* file);

	void RemoveData(int i, BaseContainer* data);
	bool BuildUI_Technique(char** techniqueNames, unsigned int numTechniques, bool init, Description* description, BaseContainer* data);
	bool BuildUI_Float(double a, double minVal, double maxVal, double stepVal, bool hasSlider, const char* name, int index, bool init, Description* description, BaseContainer* data);
	bool BuildUI_String(const char* s, const char* name, int index, bool init, Description* description, BaseContainer* data);
	bool BuildUI_Hidden(int index, Description* description, BaseContainer* data);
	bool BuildUI_Color(float* col, const char* name, int index, bool init, Description* description, BaseContainer* data);
	bool BuildUI_Filename(const char* file, const char* name, int index, bool init, Description* description, BaseContainer* data);
	bool BuildUI_Link(const char* name, int index, bool init, Description* description, BaseContainer* data);
	bool BuildUI_Vector(float* ved, const char* name, int index, bool init, Description* description, BaseContainer* data);
	bool GetTechnique(int& tech, BaseContainer* data);
	bool GetParamFloat(int i, float& f, BaseContainer* data);
	bool GetParamVector(int i, float* v, BaseContainer* data);
	bool GetParamFilename(int i, char* file, BaseContainer* data);
	bool IsParamFloat(int i, BaseContainer* data);
	bool IsParamVector(int i, BaseContainer* data);
	bool IsParamFilename(int i, BaseContainer* data);
	bool IsParamLink(int i, BaseContainer* data);
};

#endif
