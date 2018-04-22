// from C4Dfx by Jörn Loviscach, www.l7h.cn
// a function to render a single object and a function to render the scene

#include <windows.h>
#include <glh/glh_extensions.h>
#include <GL/glu.h>
#include "CgFX/ICgFXEffect.h"
#include "C4DWrapper.h"
#include "LightsMaterialsObjects.h"
#include "FXWrapper.h"
#include "MatrixMath.h"
#include "Paint.h"
#include <assert.h>

void RenderSingleObject(ObjectIterator* oi)
{
	FXWrapper* fxwrap = 0;
	if(! oi->GetAndCheckFXWrapper(fxwrap))
	{
		oi->Print("No standard material or faulty C4Dfx material.", "");
		return;
	}

	ICgFXEffect* effect = 0;

	float* vert = 0;
	long numVert;
	long* poly = 0;
	long numPoly;

	float* normals = 0;
	float* tangents = 0;
	float* binormals = 0;
	float* texCoords = 0;

	if(! oi->GetPolys(vert, numVert, poly, numPoly))
	{
		oi->Print("Isn't a polygon object.", "");
		goto error;
	}
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, vert);

	normals = new float[3*numVert];
	tangents = new float[3*numVert];
	binormals = new float[3*numVert];
	texCoords = new float[3*numVert];
	if(normals == 0 || tangents == 0 || binormals == 0 || texCoords == 0)
	{
		oi->Print("Couldn't allocate memory.", "");
		goto error;
	}
	if(! oi->GetNormalsEtc(normals, tangents, binormals, texCoords, fxwrap->UseOrthoFrame()))
	{
		oi->Print("uvw tag inaccessible.", "");
		goto error;
	}

	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, 0, normals);

	glClientActiveTextureARB(GL_TEXTURE0_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(3, GL_FLOAT, 0, texCoords);

	glClientActiveTextureARB(GL_TEXTURE1_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(3, GL_FLOAT, 0, tangents);

	glClientActiveTextureARB(GL_TEXTURE2_ARB);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(3, GL_FLOAT, 0, binormals);

	effect = fxwrap->GetEffect();

    UINT numPasses;
    if(FAILED(effect->Begin(&numPasses, 0)))
	{
		oi->Print("Effect initialization failed.", "");
		goto error;
	}

	if(! fxwrap->BeginRenderingObject(oi))
		goto error1;

	unsigned int pass;
    for (pass = 0; pass < numPasses; ++pass)
	{
        if(FAILED(effect->Pass(pass)))
		{
			oi->Print("CgFX: pass failed.", "");
			goto error1;
		}
		glDrawElements(GL_QUADS, 4*numPoly, GL_UNSIGNED_INT, poly);
    }

	if(FAILED(effect->End()))
		goto error1;
error1:
	fxwrap->EndRenderingObject();
error:
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glClientActiveTextureARB(GL_TEXTURE2_ARB);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glClientActiveTextureARB(GL_TEXTURE1_ARB);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glClientActiveTextureARB(GL_TEXTURE0_ARB);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	delete[] texCoords;
	texCoords = 0;
	delete[] binormals;
	binormals = 0;
	delete[] tangents;
	tangents = 0;
	delete[] normals;
	normals = 0;
}

void Paint(BaseDocument* doc, Materials* mat, ShadowMaps* shadow)
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glColor4d(0.0, 0.0, 0.0, 1.0);
	glBegin(GL_QUADS);
		glVertex2d(-1.0, -1.0);
		glVertex2d( 1.0, -1.0);
		glVertex2d( 1.0,  1.0);
		glVertex2d(-1.0,  1.0);
	glEnd();

//	to test shadow maps
//	shadow->Bind(0);
//	glColor4d(1.0, 0.0, 0.0, 1.0);
//	glBegin(GL_QUADS);
//		glTexCoord2d(0.0, 0.0);
//		glVertex2d(-1.0, -1.0);
//		glTexCoord2d(1.0, 0.0);
//		glVertex2d( 1.0, -1.0);
//		glTexCoord2d(1.0, 1.0);
//		glVertex2d( 1.0,  1.0);
//		glTexCoord2d(0.0, 1.0);
//		glVertex2d(-1.0,  1.0);
//	glEnd();
//	glFinish();
//	shadow->Release(0);

	ObjectIterator oi(doc, mat, RenderSingleObject);
}