//
// widgets.h
// Last Updated:		05.01.07
// 
// Mark Colbert & Jaroslav Krivanek
// colbert@cs.ucf.edu
//
// Copyright (c) 2007.
//
// The following code is freely distributed "as is" and comes with 
// no guarantees or required support by the authors.  Any use of 
// the code for commercial purposes requires explicit written consent 
// by the authors.
//


#include <Cg/cg.h>

#include "guiwidget.h"
#include "brdfgraph.h"
#include "lobepainter.h"

// Widgets
enum BRDFType {LAFORTUNE=0, WARD, SV_LAFORTUNE};
extern BRDFType type;

extern BRDFGraph	*brdfGraph;

/// Tone Mapping Widgets
extern WidgetGroup	*toneMapGrp;
extern SliderBar	*exposureKey;
extern SliderBar	*gamma;

/// Lafortune Widgets
extern WidgetGroup *lobeGrp;
extern SliderBar	*cxy;
extern SliderBar	*cz;
extern SliderBar	*n;
extern SliderBar	*ps_r, *ps_g, *ps_b;

extern WidgetGroup *diffuseGrp;
extern SliderBar	*pd_r, *pd_g, *pd_b;

/// Ward Widgets
extern WidgetGroup	*wardLobeGrp;
extern SliderBar	*ax;
extern SliderBar	*ay;
extern SliderBar	*wps_r, *wps_g, *wps_b;
extern WidgetGroup	*wardDiffuseGrp;

/// Sequence Type Selector
extern const char *seqOptions[];
extern ScrollSelector seqSelector;

/// Number of samples selector
extern const char *smplOptions[];
extern ScrollSelector smplSelector;

extern GLuint hdrTexId[6];

template<class T> 
class SetOptionButton : public Button {
	public:
		SetOptionButton(const char *name, int x, int y, int w, int h, T _mode, T *_data) :
						Button(name, x,y,w,h), mode(_mode), data(_data) {}

		virtual void OnClick() { (*data) = mode; }
	private:
		T *data;
		T mode;
};

/// Button selectors
extern SetOptionButton<int> probeButtons[3];
extern SetOptionButton<int> meshButtons[3];

extern LobePainter *lobePainter;

// actual external variables and functions
extern int currMesh;
extern int currProbe;
extern CGparameter keyGammaParam;
extern int wWidth, wHeight;
extern void updateProbeSH(int probe);

extern void initWidgets();
extern void initSamplingWidgets();
extern void initToneMappingWidgets();
extern void initLafortuneWidgets();
extern void initWardWidgets();
extern void initProbeGroup();
extern void initMeshGroup();
extern void initBRDFGroup();

extern void destroyWidgets();

extern void displayInterface();
extern void clickInterface(int button, int state, int x, int y);
extern bool motionInterface(int x, int y);
extern void reshapeInterface(int w, int h);
