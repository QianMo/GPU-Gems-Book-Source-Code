// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes for shadows, shaders, and parameters

#if !defined(FXWRAPPER_H)
#define FXWRAPPER

// be sure to include this file after the CgFX files
#if !defined(__ICGFX_EFFECT_H)
struct ICgFXEffect;
struct CgFXEFFECT_DESC;
struct CgFXPARAMETER_DESC;
#endif

class BaseDocument;
class BaseMaterial;
class BaseObject;
class BaseContainer;
class Description;
class ParamWrapper;
class ObjectIterator;
class Lights;
class Materials;
struct HDC__;
struct HPBUFFERARB__;
struct HGLRC__;

class ShadowMaps
{
public:
	ShadowMaps(HDC__* hDC, Lights* lig, int mapSize);
	~ShadowMaps(void); // don't inherit
	void Render(BaseDocument* doc, Materials* mat);
	int Bind(int which); // returns texture ref number
	void Release(int which);
private:
	void Begin(int which);
	void End(void);
	HDC__* dc;
	Lights* lights;
	int number;
	int size;
	int current;
	HPBUFFERARB__** depthBuf;
	HDC__** depthHDC;
	HGLRC__** depthHRC;
	bool* valid;
	bool* hasActiveShadow;
	unsigned int* textureNumbers;
};

class FXWrapper
{
public:
	FXWrapper(void);
	virtual ~FXWrapper(void);
	void ReleaseEffect(void);
	bool PrepareAndValidateTechnique(int t);
	virtual bool BeginRenderingObject(ObjectIterator* oi) = 0;
	virtual void EndRenderingObject(void) = 0;
	ICgFXEffect* GetEffect(void) const;
	bool IsValid(void) const;
	virtual void BeginRendering(BaseDocument* d, BaseMaterial* m) = 0;
	virtual void EndRendering(void) = 0;
	virtual bool UseOrthoFrame(void) = 0;
protected:
	bool valid;
	ICgFXEffect* effect;
	CgFXEFFECT_DESC* effectDesc;
	int numTechniques;
};

class FXEmulator : public FXWrapper
{
public:
	FXEmulator(BaseDocument* doc, BaseMaterial* mat, Lights* li, int mapSize,
		unsigned char* const texturePic, ShadowMaps* sm);
	virtual ~FXEmulator(void);
	bool Load(const char** errors);
	virtual bool BeginRenderingObject(ObjectIterator* oi);
	virtual void EndRenderingObject(void);
	bool IsValid(void) const;
	virtual void BeginRendering(BaseDocument* d, BaseMaterial* m);
	virtual void EndRendering(void);
	virtual bool UseOrthoFrame(void) {return false;}
private:
	BaseDocument* document;
	BaseMaterial* baseMaterial;
	Lights* lit;
	unsigned int diffTextureNumber, normTextureNumber, enviTextureNumber, specShapeTextureNumber;
	int size;
	unsigned char* const pic;
	bool useDiff, useBump, useEnvi, useLumi, useSpec, useSpecColor;
	ShadowMaps* shadows;
};

class FXFromFile : public FXWrapper
{
public:
	FXFromFile(void);
	virtual ~FXFromFile(void);
	bool Load(char* file, const char** errors);
	bool BuildUI_Technique(bool init, bool setToDefault, Description* description, BaseContainer* data) const;
	bool BuildUI_Parameters(bool init, bool setToDefault, Description* description, BaseContainer* data) const; // sets data to valid values, too
	virtual bool BeginRenderingObject(ObjectIterator* oi);
	virtual void EndRenderingObject(void);
	bool IsParamEnabled(int i);
	virtual void BeginRendering(BaseDocument* d, BaseMaterial* m);
	virtual void EndRendering(void);
	virtual bool UseOrthoFrame(void) {return true;}
private:
	ParamWrapper** paramList;
	int numParameters;
	char** techniqueNames;
};

class ParamWrapper
{
public:
	static ParamWrapper* BuildParamWrapper(ICgFXEffect* effect, int i); // returns 0 if parameter is not of this type
	virtual bool BeginRenderingObject(ObjectIterator* oi); // get value from UI and set in effect or load texture or load matrix
	virtual void EndRenderingObject(void); // do some cleanup
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data); // also for initialization, including name
	virtual bool IsEnabled(void);
	bool Check(void) const {return ok;} // returns false if parameter is not of this type
	bool GetAnnotation(const char* name, float &a) const;
	bool GetAnnotation(const char* name, const char* &s) const;
	ParamWrapper(ICgFXEffect* effect, int index);
	ParamWrapper(const ParamWrapper* pw);
	virtual ~ParamWrapper(void);
	virtual void BeginRendering(BaseDocument* d, BaseMaterial* m);
	virtual void EndRendering(void);
protected:
	ICgFXEffect* effect;
	CgFXEFFECT_DESC* effectDesc;
	CgFXPARAMETER_DESC* paramDesc;
	int index;
	bool ok;
	char* uiName;
};

class UnknownParam : public ParamWrapper
{
public:
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	UnknownParam(const ParamWrapper* pw);
	virtual ~UnknownParam(void);
};

class FloatParam : public ParamWrapper
{
public:
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	FloatParam(const ParamWrapper* pw); // sets ok to false if incorrect type
	virtual ~FloatParam();
	virtual void BeginRendering(BaseDocument* d, BaseMaterial* m);
private:
	float defaultValue;
	float minVal, maxVal, stepVal;
	bool hasSlider;
};

class StringParam : public ParamWrapper
{
public:
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	StringParam(const ParamWrapper* pw);
	virtual ~StringParam(void);
private:
	char* str;
};

class MatrixParam : public ParamWrapper
{
public:
	virtual bool BeginRenderingObject(ObjectIterator* oi);
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	MatrixParam(const ParamWrapper* pw);
	virtual ~MatrixParam(void);
	virtual void BeginRendering(BaseDocument* d, BaseMaterial* m);
private:
	bool containsWorld, containsView, containsProjection, isTransposed, isInverted;
};

class ColorParam : public ParamWrapper
{
public:
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	ColorParam(const ParamWrapper* pw);
	virtual ~ColorParam(void);
	void BeginRendering(BaseDocument* d, BaseMaterial* m);
private:
	float defaultValue[4];
};

class TextureParam : public ParamWrapper
{
public:
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	TextureParam(const ParamWrapper* pw);
	virtual ~TextureParam(void);
	virtual void BeginRendering(BaseDocument* d, BaseMaterial* m);
	virtual void EndRendering(void);
private:
	int textureType;
	unsigned int textureNumber; // not fixed; only used for rendering
	char* file;
};

class SuppressedParam : public ParamWrapper
{
public:
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	SuppressedParam(const ParamWrapper* pw);
	virtual ~SuppressedParam(void);
};

class DirectionOrPositionParam : public ParamWrapper
{
public:
	virtual bool BeginRenderingObject(ObjectIterator* oi);
	virtual void AfterRender(void);
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	DirectionOrPositionParam(const ParamWrapper* pw);
	virtual ~DirectionOrPositionParam(void);
private:
	bool isPosition;
	bool isWorld;
};

class VectorParam : public ParamWrapper
{
public:
	virtual bool BuildUI(bool init, bool setToDefault, Description* description, BaseContainer* data);
	virtual bool IsEnabled(void);
	VectorParam(const ParamWrapper* pw);
	virtual ~VectorParam(void);
	virtual void BeginRendering(BaseDocument* d, BaseMaterial* m);
private:
	float defaultValue[4];
};

#endif