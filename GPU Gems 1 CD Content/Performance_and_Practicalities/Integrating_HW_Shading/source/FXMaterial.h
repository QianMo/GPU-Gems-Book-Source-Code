// from C4Dfx by Jörn Loviscach, www.l7h.cn
// a class for the C4Dfx material

#if !defined(FXMATERIAL_H)
#define FXMATERIAL_H

#include "c4d.h"
#include "FXWrapper.h"

class FXMaterial : public MaterialData
{
	INSTANCEOF(FXMaterial, MaterialData)

	public:
		virtual Bool Read(GeListNode* node, HyperFile* hf, LONG level);
		Bool ReadForRendering(GeListNode* node);
		virtual Bool GetDDescription(GeListNode* node, Description* description, LONG& flags);
		virtual Bool GetDEnabling(GeListNode* node, const DescID& id, GeData& t_data, LONG flags, const BaseContainer* itemdesc);
		virtual Bool Message(GeListNode* node, LONG type, void* data);
		virtual Bool Init (GeListNode *node);
		virtual	void CalcSurface(PluginMaterial *mat, VolumeData *vd);
		static NodeData *Alloc(void);
		FXWrapper* GetFXWrapper(void);
		virtual Bool CopyTo(NodeData* dest, GeListNode* snode, GeListNode* dnode, LONG flags, AliasTrans* trn);
	private:
		FXMaterial(void);
		Filename path;
		FXFromFile wrapper;
		bool doInit; // material has to be loaded from .fx file
		bool setToDefault; // material has to be set to default values
		bool forRendering;
		bool error;
};

#endif