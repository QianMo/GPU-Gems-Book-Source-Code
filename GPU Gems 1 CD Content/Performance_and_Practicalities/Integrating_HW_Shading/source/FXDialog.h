// from C4Dfx by Jörn Loviscach, www.l7h.cn
// the class for the main dialog of C4Dfx

#if !defined(FX_DIALOG_H)
#define FX_DIALOG_H

#include "c4d.h"
#include "RenderDialog.h"

class FXDialog : public GeDialog
{
public:
	FXDialog(void);
	~FXDialog(void); // do not inherit
	virtual Bool CreateLayout(void);
	virtual Bool InitValues(void);
	virtual Bool Command(LONG id, const BaseContainer& msg);
	virtual Bool CoreMessage(LONG id, const BaseContainer& msg);
	virtual void DestroyWindow(void);
	virtual void Timer(const BaseContainer& msg);
private:
	static LONG ComputeSize(LONG button);
	Bool autoUpdate;
	LONG antialiasing;
	LONG mapSize;
	BaseContainer windowSet;
	RenderWinWindow rww;
};

#endif