// from C4Dfx by Jörn Loviscach, www.l7h.cn
// functions hiding Cinema 4D functionality

#include "c4d.h"
#include "C4DWrapper.h"
#include "IDnumbers.h"
#include "Mfxmaterial.h"
#include "customgui_filename.h"
#include <math.h>
#include <assert.h>

void C4DWrapper::MsgBox(const char* s)
{
	MessageDialog(String("C4Dfx: ") + s);
}

void C4DWrapper::MsgBox(const char* s1, const char* s2)
{
	MessageDialog(String("C4Dfx: ") + s1 + " " + s2);
}

void C4DWrapper::Print(const char* s)
{
	GePrint(String("C4Dfx: ") + s);
}

void C4DWrapper::Print(const char* s1, const char* s2)
{
	GePrint(String("C4Dfx: ") + s1 + " " + s2);
}

void C4DWrapper::SetFrame(BaseDocument* doc, double f)
{
	BaseDocument* d = doc;
	d->SetTime(BaseTime(f));
	d->AnimateDocument(NULL, TRUE, TRUE);
}

long C4DWrapper::GetRenderWidth(BaseDocument* doc)
{
	BaseContainer* c = doc->GetActiveRenderData()->GetDataInstance();
	return c->GetLong(RDATA_XRES, 42);
}

long C4DWrapper::GetRenderHeight(BaseDocument* doc)
{
	BaseContainer* c = doc->GetActiveRenderData()->GetDataInstance();
	return c->GetLong(RDATA_YRES, 42);
}

long C4DWrapper::GetFps(BaseDocument* doc)
{
	return doc->GetFps();
}

long C4DWrapper::GetFpsRender(BaseDocument* doc)
{
	BaseContainer* c = doc->GetActiveRenderData()->GetDataInstance();
	return c->GetLong(RDATA_FRAMERATE);
}

long C4DWrapper::GetStartFrame(BaseDocument* doc)
{
	BaseContainer* c = doc->GetActiveRenderData()->GetDataInstance();
	return c->GetTime(RDATA_FRAMEFROM).GetFrame(doc->GetFps());	
}

long C4DWrapper::GetEndFrame(BaseDocument* doc)
{
	BaseContainer* c = doc->GetActiveRenderData()->GetDataInstance();
	return c->GetTime(RDATA_FRAMETO).GetFrame(doc->GetFps());
}

bool C4DWrapper::CheckFormat(BaseDocument* doc)
{
	BaseContainer* c = doc->GetActiveRenderData()->GetDataInstance();
	return FILTER_AVI_USER == c->GetLong(RDATA_FORMAT);
}

long C4DWrapper::GetRenderPathLength(BaseDocument* doc)
{
	BaseContainer* c = doc->GetActiveRenderData()->GetDataInstance();
	return 	c->GetFilename(RDATA_PATH).GetString().GetCStringLen(St8bit);
}

BaseDocument* C4DWrapper::CloneDoc(BaseDocument* doc)
{
	return (BaseDocument*)(doc->GetClone(0L, NULL));
}

void C4DWrapper::FreeDoc(BaseDocument* doc)
{
	BaseDocument::Free(doc);
}

BaseContainer* C4DWrapper::CloneContainer(BaseContainer* c)
{
	return c->GetClone(NULL);
}

void C4DWrapper::DeleteContainer(BaseContainer* c)
{
	gDelete(c);
}

bool C4DWrapper::GetMaterialBool(BaseMaterial* material, long id, bool* val)
{
	GeData d;
	Bool retr = material->GetParameter(DescID(id), d, 0L);
	*val = (d.GetLong() != 0);
	return retr != 0;
}

bool C4DWrapper::GetMaterialVector(BaseMaterial* material, long id, float* vec)
{
	BaseContainer* b = material->GetDataInstance();
	Vector v = b->GetVector(id);
	vec[0] = v.x;
	vec[1] = v.y;
	vec[2] = v.z;
	return true;
}

bool C4DWrapper::GetMaterialFloat(BaseMaterial* material, long id, float* val)
{
	BaseContainer* b = material->GetDataInstance();
	*val = b->GetReal(id);
	return true;
}

bool C4DWrapper::GetMaterialTexture(BaseMaterial* material, long id, unsigned char* pic, int size)
{
	BaseChannel* c = material->GetChannel(id);
	if(c == NULL)
		return false;

	BaseDocument* doc = material->GetDocument();
	if(doc == NULL)
		return false;

	InitRenderStruct is;
	is.doc = doc;
	Filename fn(doc->GetDocumentPath());
	is.docpath = &fn;
	is.errorlist = NULL;
	is.flags = INITRENDERFLAG_TEXTURES;
	is.fps = doc->GetFps();
	String name(material->GetName()); 
	is.matname = &name;
	is.thread = NULL;
	is.time = doc->GetTime();
	is.vd = NULL;
	is.version = GetC4DVersion();
	if(c->InitTexture(&is) != LOAD_OK)
		return false;

	int x, y;
	Vector col;
	Vector p;
	double rSize = 1.0/size;
	Vector delta(rSize, rSize, rSize);
	Vector n(0.0, 0.0, 1.0); // This is a white lie.
	double t = is.time.Get();

	--pic;
	for(y=0; y<size; ++y)
	{
		p.y = y*rSize;
		for(x=0; x<size; ++x)
		{
			p.x = x*rSize;
			col = c->Sample(NULL, &p, &delta, &n, t, 0L, 0.0f, 0.0f);
			*(++pic) = (unsigned char)(255.99*col.x);
			*(++pic) = (unsigned char)(255.99*col.y);
			*(++pic) = (unsigned char)(255.99*col.z);
		}
	}

	c->FreeTexture();

	return true;
}

bool C4DWrapper::GetMaterialSpecShapeTexture(BaseMaterial* material, unsigned char* pic, int size)
{
	BaseContainer* b = material->GetDataInstance();
	float width = b->GetReal(MATERIAL_SPECULAR_WIDTH);
	float falloff = b->GetReal(MATERIAL_SPECULAR_FALLOFF);
	float innerw = b->GetReal(MATERIAL_SPECULAR_INNERWIDTH);
	
	float expo = pow(100.0, falloff + 0.125);
	float fact = 2.0/3.14159265/width/(1.0001 - width*innerw);
	float subt = innerw/(1.0001 - width*innerw);
	float twbw = 2.0/width;

	float d = 1.0/(size-1);
	int x;
	*pic = 0; // angle is >= 90°
	++pic;
	for(x=1; x<size; ++x)
	{
		*pic = (unsigned char)(0.5+255*pow(1.0 - pow(FCut(acos(x*d)*fact - subt, 0.0, 1.0), expo), twbw));
		++pic;
	}

	return true;
}

bool C4DWrapper::GetMaterialNormalTexture(BaseMaterial* material, unsigned char* pic, int size)
{
	BaseChannel* c = material->GetChannel(CHANNEL_BUMP);
	if(c == NULL)
		return false;

	BaseDocument* doc = material->GetDocument();
	if(doc == NULL)
		return false;

	InitRenderStruct is;
	is.doc = doc;
	Filename fn(doc->GetDocumentPath());
	is.docpath = &fn;
	is.errorlist = NULL;
	is.flags = INITRENDERFLAG_TEXTURES;
	is.fps = doc->GetFps();
	String name(material->GetName()); 
	is.matname = &name;
	is.thread = NULL;
	is.time = doc->GetTime();
	is.vd = NULL;
	is.version = GetC4DVersion();
	if(c->InitTexture(&is) != LOAD_OK)
		return false;

	int x, y;
	Vector col;
	Vector p;
	double rSize = 1.0/size;
	Vector delta(rSize, rSize, rSize);
	Vector n(0.0, 0.0, 1.0); // a quite arbitrary value ...
	double t = is.time.Get();

	float* bumpMap = bNew float[size*size];
	if(bumpMap == NULL)
		return false;
	
	float* pBump = bumpMap;
	--pBump;
	for(y=0; y<size; ++y)
	{
		p.y = y*rSize;
		for(x=0; x<size; ++x)
		{
			p.x = x*rSize;
			col = c->Sample(NULL, &p, &delta, &n, t, TEX_BUMP, 0.0f, 0.0f);
			*(++pBump) = VectorGray(col);
		}
	}

	int d = size/25;

	--pic;
	for(y=0; y<size; ++y)
	{
		for(x=0; x<size; ++x)
		{
			double nx = bumpMap[y*size + ((x+size-1)%size)] - bumpMap[y*size + ((x+1)%size)];
			double ny = bumpMap[((y+size-1)%size)*size + x] - bumpMap[((y+1)%size)*size + x];
			// note: bump intensity parameter is scaled by size elsewhere

			*(++pic) = 128 + (unsigned char)(127.99*nx);
			*(++pic) = 128 + (unsigned char)(127.99*ny);
			*(++pic) = 0.42; //dummy
		}
	}

	bDelete(bumpMap);
	c->FreeTexture();

	return true;
}

bool C4DWrapper::GetMaterialEnvTexture(BaseMaterial* material, unsigned char* pic, int size, int dir)
{
	BaseChannel* c = material->GetChannel(CHANNEL_ENVIRONMENT);
	if(c == NULL)
		return false;

	BaseDocument* doc = material->GetDocument();
	if(doc == NULL)
		return false;

	InitRenderStruct is;
	is.doc = doc;
	Filename fn(doc->GetDocumentPath());
	is.docpath = &fn;
	is.errorlist = NULL;
	is.flags = INITRENDERFLAG_TEXTURES;
	is.fps = doc->GetFps();
	String name(material->GetName()); 
	is.matname = &name;
	is.thread = NULL;
	is.time = doc->GetTime();
	is.vd = NULL;
	is.version = GetC4DVersion();
	if(c->InitTexture(&is) != LOAD_OK)
		return false;

	int x, y;
	Vector col;
	Vector p;
	double rSize = 1.0/size;
	Vector delta(rSize, rSize, rSize); // or something else ...
	double t = is.time.Get();

	Vector majorAxis, sAxis, tAxis;
	switch(dir)
	{
	case 0:
		majorAxis = Vector(1.0, 0.0, 0.0);
		sAxis = Vector(0.0, 0.0, -1.0);
		tAxis = Vector(0.0, -1.0, 0.0);
		break;
	case 1:
		majorAxis = Vector(-1.0, 0.0, 0.0);
		sAxis = Vector(0.0, 0.0, 1.0);
		tAxis = Vector(0.0, -1.0, 0.0);
		break;
	case 2:
		majorAxis = Vector(0.0, 1.0, 0.0);
		sAxis = Vector(1.0, 0.0, 0.0);
		tAxis = Vector(0.0, 0.0, 1.0);
		break;
	case 3:
		majorAxis = Vector(0.0, -1.0, 0.0);
		sAxis = Vector(1.0, 0.0, 0.0);
		tAxis = Vector(0.0, 0.0, -1.0);
		break;
	case 4:
		majorAxis = Vector(0.0, 0.0, 1.0);
		sAxis = Vector(1.0, 0.0, 0.0);
		tAxis = Vector(0.0, -1.0, 0.0);
		break;
	case 5:
		majorAxis = Vector(0.0, 0.0, -1.0);
		sAxis = Vector(-1.0, 0.0, 0.0);
		tAxis = Vector(0.0, -1.0, 0.0);
		break;
	}

	--pic;
	for(y=0; y<size; ++y)
	{
		for(x=0; x<size; ++x)
		{
			Vector r = majorAxis + ((2.0*x)/(double)size - 1.0)*sAxis + ((2.0*y)/(double)size - 1.0)*tAxis;
			r = !r; // normalize	
			Vector n(-r);

			//inversion of:
			//u = p.x;
			//v = p.y;
			//x = sin(v*pi)*cos(u*pi2);
			//y = cos(v*pi);
			//z = sin(v*pi)*sin(u*pi2);
			p.x = atan2(r.z, r.x)*(1.0/pi2);
			if(p.x<0.0)
				p.x += 1.0;
			p.y = acos(r.y)*(1.0/pi);

			col = c->Sample(NULL, &p, &delta, &n, t, 0L, 0.0f, 0.0f);
			*(++pic) = (unsigned char)(255.99*col.x);
			*(++pic) = (unsigned char)(255.99*col.y);
			*(++pic) = (unsigned char)(255.99*col.z);
		}
	}

	c->FreeTexture();

	return true;
}

long C4DWrapper::GetChannelIdColor(void)
{
	return CHANNEL_COLOR;
}

bool C4DWrapper::GetObjectLong(BaseObject* obj, long id, long* val)
{
	GeData d;
	obj->GetParameter(DescID(id), d, 0L);
	*val = d.GetLong();
	return true;
}

bool C4DWrapper::GetObjectBool(BaseObject* obj, long id, bool* val)
{
	GeData d;
	obj->GetParameter(DescID(id), d, 0L);
	*val = (d.GetLong() != 0 ? true : false);
	return true;
}

bool C4DWrapper::GetObjectFloat(BaseObject* obj, long id, float* val)
{
	GeData d;
	obj->GetParameter(DescID(id), d, 0L);
	*val = d.GetReal();
	return true;
}

bool C4DWrapper::GetObjectVector(BaseObject* obj, long id, float* vec)
{
	GeData d;
	obj->GetParameter(DescID(id), d, 0L);
	Vector v = d.GetVector();
	vec[0] = v.x;
	vec[1] = v.y;
	vec[2] = v.z;
	return true;
}


// in column-major order
bool C4DWrapper::GetViewMatrix(BaseDocument*doc, float* m)
{
	BaseDraw* b = doc->GetRenderBaseDraw();
	if(b == NULL)
		return false;

	//	Matrix mat = b->GetMi(); would be fine, but doesn't work

	BaseObject* cam = b->GetSceneCamera(doc);
	if(cam == NULL) // no scene camera used
	{
		cam = b->GetEditorCamera();
		if(cam == NULL)
			return false;
	}

	Matrix mat = ! cam->GetMg();
	m[0] = mat.v1.x;
	m[1] = mat.v1.y;
	m[2] = mat.v1.z;
	m[3] = 0.0;
	m[4] = mat.v2.x;
	m[5] = mat.v2.y;
	m[6] = mat.v2.z;
	m[7] = 0.0;
	m[8] = mat.v3.x;
	m[9] = mat.v3.y;
	m[10] = mat.v3.z;
	m[11] = 0.0;
	m[12] = mat.off.x;
	m[13] = mat.off.y;
	m[14] = mat.off.z;
	m[15] = 1.0;
	return true;
}

// in column-major order
bool C4DWrapper::GetProjMatrix(BaseDocument* doc, float* m)
{
	long width = C4DWrapper::GetRenderWidth(doc);
	long height = C4DWrapper::GetRenderHeight(doc);

	if(width == 0 || height == 0)
		return false;

	BaseDraw* bd = doc->GetRenderBaseDraw();
	if(bd == NULL)
		return false;

	BaseObject* cam = bd->GetSceneCamera(doc);
	if(cam == NULL) // no scene camera used
	{
		cam = bd->GetEditorCamera();
		if(cam == NULL)
			return false;
	}
  
	BaseContainer* data = cam->GetDataInstance();
	LONG projection = data->GetLong(CAMERA_PROJECTION, Pperspective);
    if (projection != Pperspective)
	{
		Print("Camera is not in perspective mode.");
		return false;
	}
	
	double focus = data->GetReal(CAMERA_FOCUS, 36.0);
	double aperture = data->GetReal(CAMERAOBJECT_APERTURE, 36.0);
	double f = 2.0*focus/aperture;
	double aspectRatio = width/(double)height;
	double zFar = 20000.0;
	double zNear = 100.0;

	m[0] = f;
	m[1] = 0.0;
	m[2] = 0.0;
	m[3] = 0.0;
	m[4] = 0.0;
	m[5] = f*aspectRatio;
	m[6] = 0.0;
	m[7] = 0.0;
	m[8] = 0.0;
	m[9] = 0.0;
	m[10] = -(zNear+zFar)/(zNear-zFar);
	m[11] = 1.0;
	m[12] = 0.0;
	m[13] = 0.0;
	m[14] = 2.0*zFar*zNear/(zNear-zFar);
	m[15] = 0.0;
	return true;
}

bool C4DWrapper::GetParamFloat(BaseMaterial* m, int i, float &a)
{
	BaseContainer* bc = m->GetDataInstance();
	return C4DWrapper::GetParamFloat(i, a, bc);
}

bool C4DWrapper::GetParamVector(BaseMaterial* m, int i, float* v)
{
	BaseContainer* bc = m->GetDataInstance();
	return C4DWrapper::GetParamVector(i, v, bc);
}

bool C4DWrapper::GetParamFilename(BaseMaterial* m, int i, char* file)
{
	BaseContainer* bc = m->GetDataInstance();
	return C4DWrapper::GetParamFilename(i, file, bc);
}

void C4DWrapper::RemoveData(int i, BaseContainer* data)
{
	data->RemoveData(i);
}

bool C4DWrapper::BuildUI_Technique(char** techniqueNames, unsigned int numTechniques, bool init, Description* description, BaseContainer* data)
{
	if(init)
		data->SetLong(FX_TECHNIQUE, 0L);

	BaseContainer bcSt;

	unsigned int i;
	for(i=0; i<numTechniques; ++i)
	{
		bcSt.SetString(i, techniqueNames[i]);
	}

	AutoAlloc<AtomArray> ar;
	if (!ar)
		return false;

	BaseContainer* bc = description->GetParameterI(DescLevel(FX_TECHNIQUE), ar);
	if (bc)
	{
		bc->SetContainer(DESC_CYCLE, bcSt);
		return true;
	}
	else
		return true; // under some circumstances, bc == NULL
}

bool C4DWrapper::BuildUI_Float(double a, double minVal, double maxVal, double stepVal, bool hasSlider, const char* name, int index, bool init, Description* description, BaseContainer* data)
{
	BaseContainer bc = GetCustomDataTypeDefault(DTYPE_REAL);
	bc.SetLong(DESC_ANIMATE, DESC_ANIMATE_ON);
	bc.SetString(DESC_SHORT_NAME, name);
	bc.SetString(DESC_NAME, name);
	if(hasSlider)
		bc.SetLong(DESC_CUSTOMGUI, CUSTOMGUI_REALSLIDER);
	else
		bc.SetLong(DESC_CUSTOMGUI, CUSTOMGUI_REAL);
	bc.SetReal(DESC_MIN, (float)minVal);
	bc.SetReal(DESC_MAX, (float)maxVal);
	bc.SetReal(DESC_STEP, (float)stepVal);

	if(! description->SetParameter(DescLevel(20000+index, DTYPE_REAL, ID_MATERIAL_PLUGIN), bc, DescLevel(ID_FX_PARAMS_GRP)))
		return false;

	if(init)
		data->SetReal(20000+index, (float) a);

	return true;
}

bool C4DWrapper::BuildUI_String(const char* s, const char* name, int index, bool init, Description* description, BaseContainer* data)
{
	BaseContainer bc = GetCustomDataTypeDefault(DTYPE_STRING);
	bc.SetLong(DESC_ANIMATE, DESC_ANIMATE_OFF);
	bc.SetString(DESC_SHORT_NAME, name);
	bc.SetString(DESC_NAME, name);
	bc.SetLong(DESC_CUSTOMGUI, CUSTOMGUI_STRING);
	
	if(! description->SetParameter(DescLevel(20000+index, DTYPE_STRING, ID_MATERIAL_PLUGIN), bc, DescLevel(ID_FX_PARAMS_GRP)))
		return false;

	if(init)
		data->SetString(20000+index, s);
	
	return true;
}

bool C4DWrapper::BuildUI_Hidden(int index, Description* description, BaseContainer* data)
{
	BaseContainer bc = GetCustomDataTypeDefault(DTYPE_STRING);
	bc.SetLong(DESC_ANIMATE, DESC_ANIMATE_OFF);
	bc.SetLong(DESC_CUSTOMGUI, CUSTOMGUI_STRING);
	bc.SetBool(DESC_HIDE, TRUE);

	if(! description->SetParameter(DescLevel(20000+index, DTYPE_STRING, ID_MATERIAL_PLUGIN), bc, DescLevel(ID_FX_PARAMS_GRP)))
		return false;

	data->RemoveData(20000+index); // clear what may have been there	

	return true;
}

bool C4DWrapper::BuildUI_Color(float* col, const char* name, int index, bool init, Description* description, BaseContainer* data)
{
	BaseContainer bc = GetCustomDataTypeDefault(DTYPE_COLOR);
	bc.SetLong(DESC_ANIMATE, DESC_ANIMATE_ON);
	bc.SetString(DESC_SHORT_NAME, name);
	bc.SetString(DESC_NAME, name);
	bc.SetLong(DESC_CUSTOMGUI, CUSTOMGUI_COLOR);
	if(! description->SetParameter(DescLevel(20000+index, DTYPE_COLOR, ID_MATERIAL_PLUGIN), bc, DescLevel(ID_FX_PARAMS_GRP)))
		return false;

	if(init)
		data->SetVector(20000+index, Vector(col[0], col[1], col[2]));

	return true;
}

bool C4DWrapper::BuildUI_Filename(const char* file, const char* name, int index, bool init, Description* description, BaseContainer* data)
{
	BaseContainer bc = GetCustomDataTypeDefault(DTYPE_FILENAME);
	bc.SetLong(DESC_ANIMATE, DESC_ANIMATE_ON);
	bc.SetString(DESC_SHORT_NAME, name);
	bc.SetString(DESC_NAME, name);
	bc.SetLong(DESC_CUSTOMGUI, CUSTOMGUI_FILENAME);
	if(! description->SetParameter(DescLevel(20000+index, DTYPE_FILENAME, ID_MATERIAL_PLUGIN), bc, DescLevel(ID_FX_PARAMS_GRP)))
		return false;
	
	if(init)
	{
		data->SetFilename(20000+index, Filename(file));
	}

	return true;
}

bool C4DWrapper::BuildUI_Link(const char* name, int index, bool init, Description* description, BaseContainer* data)
{
	BaseContainer bc = GetCustomDataTypeDefault(DTYPE_BASELISTLINK);
	bc.SetLong(DESC_ANIMATE, DESC_ANIMATE_ON);
	bc.SetString(DESC_SHORT_NAME, name);
	bc.SetString(DESC_NAME, name);
	bc.SetLong(DESC_CUSTOMGUI, CUSTOMGUI_LINKBOX);
	if(! description->SetParameter(DescLevel(20000+index, DTYPE_BASELISTLINK, ID_MATERIAL_PLUGIN), bc, DescLevel(ID_FX_PARAMS_GRP)))
		return false;

	if(init)
		data->SetLink(20000+index, 0);

	return true;
}

bool C4DWrapper::BuildUI_Vector(float* vec, const char* name, int index, bool init, Description* description, BaseContainer* data)
{
	BaseContainer bc = GetCustomDataTypeDefault(DTYPE_VECTOR);
	bc.SetLong(DESC_ANIMATE, DESC_ANIMATE_ON);
	bc.SetString(DESC_SHORT_NAME, name);
	bc.SetString(DESC_NAME, name);
	bc.SetLong(DESC_CUSTOMGUI, CUSTOMGUI_VECTOR);
	if(! description->SetParameter(DescLevel(20000+index, DTYPE_VECTOR, ID_MATERIAL_PLUGIN), bc, DescLevel(ID_FX_PARAMS_GRP)))
		return false;

	if(init)
		data->SetVector(20000+index, Vector(vec[0], vec[1], vec[2]));

	return true;
}

bool C4DWrapper::GetTechnique(int& tech, BaseContainer* data)
{
	tech = data->GetLong(FX_TECHNIQUE);
	return true;
}

bool C4DWrapper::GetParamFloat(int i, float& f, BaseContainer* data)
{
	f = data->GetReal(20000+i);
	return true;
}

bool C4DWrapper::GetParamVector(int i, float* v, BaseContainer* data)
{
	Vector w = data->GetVector(20000+i);
	v[0] = w.x;
	v[1] = w.y;
	v[2] = w.z;
	v[3] = 1.0;
	return true;
}

bool C4DWrapper::GetParamFilename(int i, char* file, BaseContainer* data)
{
	Filename f = data->GetFilename(20000+i);
	f.GetString().GetCString(file, 255, St8bit);
	return true;
}

bool C4DWrapper::IsParamFloat(int i, BaseContainer* data)
{
	return data->GetType(20000+i) == DA_REAL;
}

bool C4DWrapper::IsParamVector(int i, BaseContainer* data)
{
	return data->GetType(20000+i) == DA_VECTOR;
}

bool C4DWrapper::IsParamFilename(int i, BaseContainer* data)
{
	return data->GetType(20000+i) == DA_FILENAME;
}

bool C4DWrapper::IsParamLink(int i, BaseContainer* data)
{
	return data->GetType(20000+i) == DA_ALIASLINK;
}