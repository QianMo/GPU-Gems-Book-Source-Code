// from C4Dfx by Jörn Loviscach, www.l7h.cn
// the class for the C4Dfx tag controlling lights

#include "c4d.h"
#include "c4d_symbols.h"
#include "Tlight.h"
#include "IDnumbers.h"

class LightTag : public TagData
{
public:
	static NodeData *Alloc(void) { return gNew LightTag; }
	virtual Bool GetDEnabling(GeListNode* node, const DescID& id, GeData& t_data, LONG flags, const BaseContainer* itemdesc);
};

Bool RegisterLightTag(void)
{
	String name = GeLoadString(IDS_LIGHTTAG); if (!name.Content()) return TRUE;
	return RegisterTagPlugin(ID_LIGHT, name, TAG_VISIBLE, LightTag::Alloc, "Tlight", "fx24.tif", 0);
}

Bool LightTag::GetDEnabling(GeListNode* node, const DescID& id, GeData& t_data, LONG flags, const BaseContainer* itemdesc)
{
	switch(id[0].id)
	{
		case INCLUDELIGHT:
			return TRUE;
		case INCLUDESHADOW: // if this is off disable INCLUDELIGHT 
			{
				BaseContainer* bc = ((BaseTag*)node)->GetDataInstance();
				if(bc != NULL && bc->GetBool(INCLUDELIGHT))
					return TRUE;
				return FALSE;
			}
	}
	return TagData::GetDEnabling(node, id, t_data, flags, itemdesc);
}
