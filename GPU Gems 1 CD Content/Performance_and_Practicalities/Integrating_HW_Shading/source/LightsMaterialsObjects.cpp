// from C4Dfx by Jörn Loviscach, www.l7h.cn
// classes for lights, materials, and objects

#include "c4d.h"
#include "LightsMaterialsObjects.h"
#include "C4DWrapper.h"
#include "IDnumbers.h"
#include "FXMaterial.h"
#include "Tlight.h"
#include <math.h>
#include <assert.h>

Lights::Lights(BaseDocument* doc)
: document(doc), lights(NULL), pos(NULL), dir(NULL), up(NULL), side(NULL),
lightFromShadow(NULL), shadowFromLight(NULL), hasShadowMap(NULL), valid(false)
{
	GetLights(); // only count, because lights == NULL

	lights = bNew BaseObject*[numLights];
	if(lights == NULL)
		goto error;
	ClearMem(lights, sizeof(BaseObject*)*numLights, 0);

	pos = bNew float[numLights][4];
	if(pos == NULL)
		goto error;

	dir = bNew float[numLights][4];
	if(dir == NULL)
		goto error;

	up = bNew float[numLights][4];
	if(up == NULL)
		goto error;

	side = bNew float[numLights][4];
	if(side == NULL)
		goto error;

	lightFromShadow = bNew int[numShadowLights];
	if(lightFromShadow == NULL)
		goto error;

	shadowFromLight = bNew int[numLights];
	if(shadowFromLight == NULL)
		goto error;
	ClearMem(shadowFromLight, sizeof(int)*numLights, -1);

	hasShadowMap = bNew bool[numLights];
	if(hasShadowMap == NULL)
		goto error;
	ClearMem(hasShadowMap, sizeof(bool)*numLights, 0);

	GetLights();
	valid = true;
	return;

error:
	C4DWrapper::Print("Shadows: Allocation error.");
	return;
}

Lights::~Lights(void)
{
	bDelete(hasShadowMap);
	bDelete(shadowFromLight);
	bDelete(lightFromShadow);
	bDelete(side)
	bDelete(up);
	bDelete(dir);
	bDelete(pos);
	bDelete(lights);
}

BaseObject* Lights::CrawlScene(BaseObject* b)
{
	BaseObject* c = b->GetDown();
	if(c != NULL)
		return c;
	c = b->GetNext();
	if(c != NULL)
		return c;
	do
	{
		b = b->GetUp();
		if(b == NULL)
		{
			return NULL;
		}
		c = b->GetNext();
	}
	while(c == NULL);
	return c;
}

// lights == 0 means: only return number, do not fill array
void Lights::GetLights(void)
{
	numLights = 0;
	numShadowLights = 0;
	BaseObject* b = document->GetFirstObject();
	while(b != NULL)
	{
		if(b->GetType() == Olight)
		{
			BaseTag* t = b->GetTag(ID_LIGHT);
			bool enableLight = true;
			bool enableShadow = true;
			if(t != NULL)
			{
				BaseContainer* c = t->GetDataInstance();
				enableLight = (c->GetBool(INCLUDELIGHT) ? true : false);
				enableShadow = (c->GetBool(INCLUDESHADOW) ? true : false);
			}
			if(enableLight)
			{
				if(lights != 0) // second pass: we're not only counting but also filling the arrays
				{
					lights[numLights] = b;
					Vector v = b->GetMg().off;
					pos[numLights][0] = v.x;
					pos[numLights][1] = v.y;
					pos[numLights][2] = v.z;
					pos[numLights][3] = 1.0f; // position

					v = !(b->GetMg().v3);
					dir[numLights][0] = v.x;
					dir[numLights][1] = v.y;
					dir[numLights][2] = v.z;
					dir[numLights][3] = 0.0f; // direction

					v = !(b->GetMg().v2);
					up[numLights][0] = v.x;
					up[numLights][1] = v.y;
					up[numLights][2] = v.z;
					up[numLights][3] = 0.0f; // direction

					v = !(b->GetMg().v1);
					side[numLights][0] = v.x;
					side[numLights][1] = v.y;
					side[numLights][2] = v.z;
					side[numLights][3] = 0.0f; // direction

					if(enableShadow)
					{
						lightFromShadow[numShadowLights] = numLights;
						shadowFromLight[numLights] = numShadowLights;
						hasShadowMap[numLights] = true;
					}
				}

				if(enableShadow)
				{
					++numShadowLights;
				}

				++numLights;
			}
		}
		b = CrawlScene(b);
	}
}

Materials::Materials(BaseDocument* doc, Lights* lights, int mapSize, ShadowMaps* sm)
: document(doc), lig(lights), size(mapSize), materialList(0), baseMaterialList(0), emulatorMaterialList(0), texturePic(0), shadows(sm)
{
	numMaterials = 0;
	BaseMaterial* m = 0;
	int n = 0;
	for(m = document->GetFirstMaterial(); m != NULL; m = m->GetNext())
		++numMaterials;

	materialList = bNew FXWrapper*[numMaterials];
	if(materialList == 0)
		goto error;
	ClearMem(materialList, numMaterials*sizeof(FXWrapper*));

	baseMaterialList = bNew BaseMaterial*[numMaterials];
	if(baseMaterialList == 0)
		goto error;
	ClearMem(baseMaterialList, numMaterials*sizeof(BaseMaterial*));

	emulatorMaterialList = bNew FXEmulator*[numMaterials];
	if(emulatorMaterialList == 0)
		goto error;
	ClearMem(baseMaterialList, numMaterials*sizeof(FXEmulator*));

	texturePic = bNew unsigned char[size*size*3]; // reused lots of times to save time for allocation
	if(texturePic == 0)
		goto error;

	m = 0;
	for(m = document->GetFirstMaterial(); m != NULL; m = m->GetNext(), ++n)
	{
		if(m->GetType() == Mmaterial)
		{
			baseMaterialList[n] = m;
			FXEmulator* fxwrap = emulatorMaterialList[n]
				= gNew FXEmulator(document, m, lig, size, texturePic, shadows); // texturePic: workspace
			if(fxwrap == 0)
				goto error;
			materialList[n] = fxwrap;

			const char* errors;
			if(! fxwrap->Load(&errors))
			{
				GePrint("C4Dfx: material " + m->GetName());
				GePrint("error on loading emulator fx");
				GePrint(errors);
				continue;
			}

			if(! fxwrap->PrepareAndValidateTechnique(0))
			{
				GePrint("C4Dfx: material " + m->GetName());
				GePrint("error on setting and validating emulator technique");
				continue;
			}
		}
		else if(m->GetType() == ID_MATERIAL_PLUGIN)
		{
			baseMaterialList[n] = m;
			FXMaterial* fxm = (FXMaterial*)(m->GetNodeData());
			if(fxm == 0 || ! fxm->ReadForRendering(m))
			{
				GePrint("C4Dfx: material " + m->GetName());
				GePrint("error on reading");
				continue;
			}

			BaseContainer* bc = m->GetDataInstance();
			int tech;
			if(!C4DWrapper::GetTechnique(tech, bc))
			{
				GePrint("C4Dfx: material " + m->GetName());
				GePrint("error on retrieving technique");
				continue;
			}

			FXWrapper* fxwrap = fxm->GetFXWrapper();
			if(fxwrap == 0)
			{
				GePrint("C4Dfx: material " + m->GetName());
				GePrint("error on retrieving FXWrapper");
				continue;
			}

			if(! fxwrap->PrepareAndValidateTechnique(tech))
			{
				GePrint("C4Dfx: material " + m->GetName());
				GePrint("error on setting and validating technique");
				continue;
			}

			materialList[n] = fxwrap;
		}
	}

	return;

error:
	C4DWrapper::Print("Materials: Allocation error.");
	return;
}

Materials::~Materials(void)
{
	bDelete(texturePic);

	if(emulatorMaterialList != 0)
	{
		int i;
		for(i=0; i<numMaterials; ++i)
		{
			gDelete(emulatorMaterialList[i]);
		}
	}

	bDelete(emulatorMaterialList);
	bDelete(baseMaterialList);
	bDelete(materialList);
}

bool Materials::GetAndCheckFXWrapper(BaseMaterial* bm, FXWrapper* &fx)
{
	if(bm == 0)
		return false;

	int n;
	for(n=0; n<numMaterials; ++n)
	{
		if(baseMaterialList[n] == bm)
		{
			fx = materialList[n];
			return fx != 0 && fx->IsValid();
		}
	}
	return false;
}

void Materials::BeginRendering(void)
{
	int n;
	for(n=0; n<numMaterials; ++n)
	{
		FXWrapper* fxw = materialList[n];
		if(fxw != 0)
			fxw->BeginRendering(document, baseMaterialList[n]);
	}
}

void Materials::EndRendering(void)
{
	int n;
	for(n=0; n<numMaterials; ++n)
	{
		FXWrapper* fxw = materialList[n];
		if(fxw != 0)
			fxw->EndRendering();
	}
}

class CrawlHierarchy : public Hierarchy
{
public:
	virtual void *Alloc(void) { return gNew HierarchyParentState; }
	virtual void Free(void *data) { HierarchyParentState *es=(HierarchyParentState*)data; gDelete(es); }
	virtual void CopyTo(void *src, void *dst) { *((HierarchyParentState*)dst)=*((HierarchyParentState*)src); }
	virtual Bool Do(void *data, BaseObject *op, const Matrix &mg, Bool controlobject);
	struct HierarchyParentState
	{
		ObjectIterator* oi;
		LONG parent_state;
		BaseMaterial* parent_material;
		BaseTag* parent_compositing;
	};
};

Bool CrawlHierarchy::Do(void *data, BaseObject *op, const Matrix &mg, Bool controlobject)
{
	HierarchyParentState *d = (HierarchyParentState*) data;

	d->oi->object = (PolygonObject*)op;
	*(d->oi->gMatrix) = mg;
	d->oi->material = d->parent_material;
	d->oi->compositing = d->parent_compositing;

	BaseTag* t = d->oi->object->GetTag(Ttexture);
	if(t != NULL)
	{
		BaseMaterial* m = ((TextureTag*)t)->GetMaterial();
		if(m != NULL && (m->GetType()==ID_MATERIAL_PLUGIN || m->GetType()==Mmaterial))
		{
			d->oi->material = m;
			d->parent_material = m;
		}
	}

	BaseTag* ct = d->oi->object->GetTag(Tcompositing);
	if(ct != NULL)
	{
		d->oi->compositing = ct;
		d->parent_compositing = ct;
	}

	LONG mode = op->GetRenderMode();

	if (mode != MODE_UNDEF) // if mode is MODE_ON or MODE_OFF    
	  d->parent_state = mode;

	if (mode == MODE_OFF || mode == MODE_UNDEF && d->parent_state == MODE_OFF) return TRUE;
	if (controlobject) return TRUE; // this object is not visible, has been used by generator

	if (op->GetType() != Opolygon) return TRUE; // We can't render this

	if(d->oi->material == NULL)
	{
		d->oi->Print("No standard or C4Dfx material.", "");
		return TRUE;
	}

	d->oi->func(d->oi);

	return TRUE;
}

// crawl through the scene using f on every tesselated object
ObjectIterator::ObjectIterator(BaseDocument* doc, Materials* m, void(*f)(ObjectIterator*))
{
	document = doc;
	mats = m;
	gMatrix = gNew Matrix;
	func = f;
	CrawlHierarchy h;
	CrawlHierarchy::HierarchyParentState s;
	s.oi = this;
	s.parent_state = MODE_ON;
	s.parent_material = NULL;
	s.parent_compositing = NULL;
	h.Run(document, FALSE, 2.0, VFLAG_EXTERNALRENDERER, &s, NULL);
}

ObjectIterator::~ObjectIterator(void)
{
	gDelete(gMatrix);
}

// caller has to supply s
bool ObjectIterator::GetObjectName(char* s) const
{
	object->GetName().GetCString(s, 255, St8bit); // better use dynamic size
	return true;
}

bool ObjectIterator::GetPolys(float* &vert, long& numVert, long* &poly, long& numPoly)
{
	PolygonObject* po = object;
	this->vert = vert = (float*)po->GetPoint();
	this->numVert = numVert = po->GetPointCount();
	this->poly = poly = (long*)po->GetPolygon();
	this->numPoly = numPoly = po->GetPolygonCount();
	return vert!=0 && poly!=0 && numVert!=0 && numPoly!=0;
}


// Must only be called _after_ GetPolys
// useOrthoFrame: true for external .fx files, false for bump map emulation (no binormal, but v direction vector)
bool ObjectIterator::GetNormalsEtc(float* normals, float* tangents, float* binormals, float* texCoords, bool useOrthoFrame) const
{
	UVWTag* t = (UVWTag*)(object->GetTag(Tuvw));
	if(t == 0)
		return false;

	int i; // for polygons
	int j; // for vertices
	Vector* faceNormals = new Vector[numPoly];
	for(i=0; i<numPoly; ++i)
	{
		faceNormals[i] = CalcFaceNormal((Vector*)vert, ((Polygon*)poly)[i]);
	}

	for(i=0; i<numPoly; ++i)
	{
		UVWStruct uvw = t->Get(i);
		const long* p = poly+4*i;
		long q = 3*p[0];
		texCoords[q] = uvw.a.x; // The texCoords may be overwritten lots of times.
		texCoords[++q] = uvw.a.y;
		texCoords[++q] = uvw.a.z;
		q = 3*p[1];
		texCoords[q] = uvw.b.x;
		texCoords[++q] = uvw.b.y;
		texCoords[++q] = uvw.b.z;
		q = 3*p[2];
		texCoords[q] = uvw.c.x;
		texCoords[++q] = uvw.c.y;
		texCoords[++q] = uvw.c.z;
		q = 3*p[3];
		texCoords[q] = uvw.d.x;
		texCoords[++q] = uvw.d.y;
		texCoords[++q] = uvw.d.z;
	}

	Neighbor neigh;
	if(! neigh.Init(numVert, (Polygon*)poly, numPoly, NULL)) // use BaseSelect instead of NULL for Phong breaks
		return false;

	for(j=0; j<numVert; ++j)
	{
		LONG *dadr = NULL;
		LONG dcnt = 0;
		neigh.GetPointPolys(j, &dadr, &dcnt);

		int k;
		Vector n1; // initialized to zero
		for(k=0; k<dcnt; ++k)
		{
			n1 += faceNormals[dadr[k]];
		}

		Vector n = ((Vector*)normals)[j] = !n1;

		Vector ukmu0xkmx0; // initialized to zero
		Vector vkmv0xkmx0; // initialized to zero
		float ukmu0vkmv0 = 0.0f;
		float ukmu02 = 0.0f;
		float vkmv02 = 0.0f;

		Vector x0 = ((Vector*)vert)[j];
		float u0 = texCoords[3*j];
		float v0 = texCoords[3*j+1];

		for(k=0; k<dcnt; ++k) // march through neighboring polys
		{
			LONG* p = (LONG*) &(((Polygon*)poly)[dadr[k]]);
			int j2;
			bool triangle = (p[2]==p[3]);
			for(j2=0; j2<(triangle?3:4); ++j2) // march thru vertices
			{
				LONG vertex = p[j2];
				if(vertex == j) // skip the central one; direct neighbours have double weight because they are collected twice
					continue;

				Vector xk = ((Vector*)vert)[vertex];
				float uk = texCoords[3*vertex];
				float vk = texCoords[3*vertex+1];
		
				Vector xkmx0 = xk-x0;
				Vector xkmxo = xkmx0;
				float ukmu0 = uk-u0;
				float vkmv0 = vk-v0;

				ukmu0xkmx0 += ukmu0*xkmxo; 
				vkmv0xkmx0 += vkmv0*xkmxo;
				ukmu0vkmv0 += ukmu0*vkmv0;
				ukmu02 += ukmu0*ukmu0;
				vkmv02 += vkmv0*vkmv0;
			}
		}
		if(useOrthoFrame)
		{
			Vector t1 = ukmu0xkmx0*vkmv02 - vkmv0xkmx0*ukmu0vkmv0;
			Vector t = ((Vector*)tangents)[j] = ! (t1 - (t1*n)*n);
			((Vector*)binormals)[j] = n%t;
		}
		else
		{
			Vector drdu = ukmu0xkmx0*vkmv02 - vkmv0xkmx0*ukmu0vkmv0;
			Vector drdv = vkmv0xkmx0*ukmu02 - ukmu0xkmx0*ukmu0vkmv0;
			((Vector*)tangents)[j] = ! (drdu - (drdu*n)*n);
			((Vector*)binormals)[j] = ! (drdv - (drdv*n)*n);
		}
	}

	delete[] faceNormals;
	faceNormals = 0;

	return true;
}

// in column-major order
bool ObjectIterator::GetWorldMatrix(float* m) const
{
	Matrix mat = *gMatrix;
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

bool ObjectIterator::GetAndCheckFXWrapper(FXWrapper* &fx) const
{
	return mats->GetAndCheckFXWrapper(material, fx);
}

bool ObjectIterator::ReadFX(void)
{
	NodeData* d = material->GetNodeData();
	if(d == 0)
		return false;
	return 0 != ((FXMaterial*)d)->ReadForRendering(material);
}		

void ObjectIterator::Print(const char* s1, const char* s2) const
{
	char name[256];
	if(! GetObjectName(name))
		name[0] = 0;
	GePrint(String("C4Dfx: ") + String(name));
	GePrint(String(s1) + String(s2));
}

bool ObjectIterator::GetParamDirectionOrPosition(bool isPosition, bool isWorld, int index, float* v) const
{
	BaseContainer* bc = material->GetDataInstance();

	Vector vec;

	BaseObject* linkedObj = (BaseObject*)bc->GetLink(20000+index, document, Obase);

	if(linkedObj == 0)
		return false;

	if(isPosition)
	{
		vec = linkedObj->GetMg().off;	
	}
	else
	{ // get z direction
		vec = !(linkedObj->GetMg().v3);
	}

	if(!isWorld)
	{
		Matrix m = *gMatrix;
		if(isPosition)
		{
			m.off.x = m.off.y = m.off.z = 0.0f;
		}
		Vector vec1 = (!m) * vec;
		vec = vec1;
	}

	v[0] = vec.x;
	v[1] = vec.y;
	v[2] = vec.z;
	v[3] = 1.0f;
	return true;
}

bool ObjectIterator::CastsShadow(void) const
{
	if(compositing == NULL)
		return true;

	GeData d;
	if(! compositing->GetParameter(DescID(DescLevel(COMPOSITINGTAG_CASTSHADOW)), d, NULL))
	{
		Print("Couldn't retrieve Cast Shadow setting", "");
		return true;
	}

	return d.GetLong() != 0;
}