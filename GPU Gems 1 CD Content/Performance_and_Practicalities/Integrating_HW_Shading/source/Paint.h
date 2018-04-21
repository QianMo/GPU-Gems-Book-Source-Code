// from C4Dfx by Jörn Loviscach, www.l7h.cn
// a function to render a single object and a function to render the scene

#if !defined(PAINT_H)
#define PAINT_H

class ObjectIterator;
class Materials;
class BaseDocument;
class ShadowMaps;

extern void RenderSingleObject(ObjectIterator* oi);
extern void Paint(BaseDocument* doc, Materials* mat, ShadowMaps* shadow);

#endif