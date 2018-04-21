/**
  @file Effects.cpp

  @maintainer Kevin Egan, (ktegan@cs.brown.edu)
  @cite Portions written by Seth Block, (smblock@cs.brown.edu)

*/


#include <G3DAll.h>
#include <SDL.h>
#include "BasicCamera.h"
#include "BasicModel.h"
#include "DemoSettings.h"
#include "Effects.h"
#include "DemoSettings.h"
#include "Renderer.h"
#include "Viewport.h"


TextureRef loadFontTexture(
    const std::string &                     filename)
{
    CImage* image;
    try {
        image = new CImage(filename);
    } catch (const CImage::Error & e) {
        debugAssertM(false, std::string("could not load texture ") + filename);
        e; // to avoid unreferenced local variable warning
    }
    unsigned char* src = (unsigned char*) image->byte();
    unsigned char* copy = new unsigned char[image->width * image->height * 4];

    // manipulate data so that we have white text with a transparent background
    int x;
    int y;
    for (y = 0; y < image->height; y++) {
        for (x = 0; x < image->width; x++) {
            int srcpos = (y * image->width + x) * 3;
            int copypos = (y * image->width + x) * 4;
            
            copy[copypos + 0] = 255;
            copy[copypos + 1] = 255;
            copy[copypos + 2] = 255;
            copy[copypos + 3] = 255 - src[srcpos + 0];
        }
    }

	TextureRef texture = Texture::fromMemory(filename,
		(const unsigned char**) (& copy), TextureFormat::RGBA8,
		image->width, image->height,
		1, TextureFormat::RGBA8, Texture::TILE,
		Texture::TRILINEAR_MIPMAP, Texture::DIM_2D);

    delete image;
    delete[] copy;

    return texture;
}



static void drawFontChar(
    char                                    ch,
    const TextureRef                        tex,
    int                                     charWidth,
    int                                     charHeight,
    float                                   x,
    float                                   y)
{
    float s = (ch % 16) * charWidth;
    float t = (ch / 16) * charHeight;

    glTexCoord2f(s, t);
    glVertex2f(x, y);

    glTexCoord2f(s, t + charHeight);
    glVertex2f(x, y + charHeight);
    
    glTexCoord2f(s + charWidth, t + charHeight);
    glVertex2f(x + charWidth, y + charHeight);
   
    glTexCoord2f(s + charWidth, t);
    glVertex2f(x + charWidth, y);
}



void drawFontString(
    const std::string &                     str,
    const TextureRef                        tex,
    int                                     charWidth,
    int                                     charHeight,
    int                                     kerning,
    float                                   x,
    float                                   y,
    int                                     winWidth,
    int                                     winHeight)
{

    int i;

    glPushAttrib(GL_ALL_ATTRIB_BITS);

	glDepthFunc(GL_ALWAYS);
    glDepthMask(0);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    // make (0, 0) the upper left corner and
    // (width, height) the lower right corner
    gluOrtho2D(0, winWidth, winHeight, 0);

    // set the texture matrix so that texture coordinates
    // s and t range from 0 to width/height
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    float texMatrix[16];
    memset(texMatrix, 0, sizeof(texMatrix));
    texMatrix[0 + 0 * 4] = 1.0 / tex->getTexelWidth();
    texMatrix[1 + 1 * 4] = 1.0 / tex->getTexelHeight();
    texMatrix[2 + 2 * 4] = 1.0;
    texMatrix[3 + 3 * 4] = 1.0;
    glLoadMatrixf(texMatrix);

    // setup GL so that we can render transparent textures
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glBindTexture(GL_TEXTURE_2D, tex->getOpenGLID());

    // render letters as small 2D quads using the font texture
    glBegin(GL_QUADS);
        for (i = 0; i < str.length(); i++) {
            drawFontChar(str[i], tex, charWidth, charHeight,
                    x + i * (charWidth + kerning), y);
        }
    glEnd();

	incrementPolyCount(str.length() * 2, POLY_COUNT_VISIBLE);

    // set GL back to it's normal state
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
}


void drawSkyBox(
    const BasicCamera&                           camera,
    const Array<TextureRef>&                skybox)
{

    glPushAttrib(GL_ALL_ATTRIB_BITS);

	glDepthFunc(GL_ALWAYS);
    glDepthMask(0);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    // draw the sky box around the eye point
    CoordinateFrame orientation = camera.getWorldToCamera();
    orientation.translation = Vector3(0, 0, 0);
    glLoadMatrix(orientation);

	// Draw the sky box
	double s = 10.0;


    glBindTexture(GL_TEXTURE_2D, skybox[0]->getOpenGLID());
    glBegin(GL_QUADS);
		glTexCoord(Vector2(0, 0));
		glVertex(Vector3(-s, +s, -s));
		glTexCoord(Vector2(0, 1));
		glVertex(Vector3(-s, -s, -s));
		glTexCoord(Vector2(1, 1));
		glVertex(Vector3(+s, -s, -s));
		glTexCoord(Vector2(1, 0));
		glVertex(Vector3(+s, +s, -s));
	glEnd();

    glBindTexture(GL_TEXTURE_2D, skybox[1]->getOpenGLID());
    glBegin(GL_QUADS);
		glTexCoord(Vector2(0, 0));
		glVertex(Vector3(-s, +s, +s));
		glTexCoord(Vector2(0, 1));
		glVertex(Vector3(-s, -s, +s));
		glTexCoord(Vector2(1, 1));
		glVertex(Vector3(-s, -s, -s));
		glTexCoord(Vector2(1, 0));
		glVertex(Vector3(-s, +s, -s));
	glEnd();


    glBindTexture(GL_TEXTURE_2D, skybox[2]->getOpenGLID());
    glBegin(GL_QUADS);
		glTexCoord(Vector2(0, 0));
		glVertex(Vector3(+s, +s, +s));
		glTexCoord(Vector2(0, 1));
		glVertex(Vector3(+s, -s, +s));
		glTexCoord(Vector2(1, 1));
		glVertex(Vector3(-s, -s, +s));
		glTexCoord(Vector2(1, 0));
		glVertex(Vector3(-s, +s, +s));
	glEnd();

    glBindTexture(GL_TEXTURE_2D, skybox[3]->getOpenGLID());
    glBegin(GL_QUADS);
		glTexCoord(Vector2(1, 0));
		glVertex(Vector3(+s, +s, +s));
		glTexCoord(Vector2(0, 0));
		glVertex(Vector3(+s, +s, -s));
		glTexCoord(Vector2(0, 1));
		glVertex(Vector3(+s, -s, -s));
		glTexCoord(Vector2(1, 1));
		glVertex(Vector3(+s, -s, +s));
	glEnd();

    glBindTexture(GL_TEXTURE_2D, skybox[4]->getOpenGLID());
    glBegin(GL_QUADS);
		glTexCoord(Vector2(1, 1));
		glVertex(Vector3(+s, +s, +s));
		glTexCoord(Vector2(1, 0));
		glVertex(Vector3(-s, +s, +s));
		glTexCoord(Vector2(0, 0));
		glVertex(Vector3(-s, +s, -s));
		glTexCoord(Vector2(0, 1));
		glVertex(Vector3(+s, +s, -s));
	glEnd();

    glBindTexture(GL_TEXTURE_2D, skybox[5]->getOpenGLID());
    glBegin(GL_QUADS);
		glTexCoord(Vector2(0, 0));
		glVertex(Vector3(+s, -s, -s));
		glTexCoord(Vector2(0, 1));
		glVertex(Vector3(-s, -s, -s));
		glTexCoord(Vector2(1, 1));
		glVertex(Vector3(-s, -s, +s));
		glTexCoord(Vector2(1, 0));
		glVertex(Vector3(+s, -s, +s));
	glEnd();

	incrementPolyCount(6 * 2, POLY_COUNT_VISIBLE);

    glPopAttrib();
}



void finalPass(
        Array<BasicModel*>&                 modelArray,
        const BasicCamera&                       camera,
        const Viewport&                     view,
		TextureRef						    fontTexture,
        DemoSettings&						vars)
{
    int i;
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glDisable(GL_LIGHTING);
    glStencilMask(0);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    const Light& curLight = vars.m_lightArray[vars.m_lightModify];

    Vector3 ur;
    Vector3 lr;
    Vector3 ll;
    Vector3 ul;
    getViewportCorners(camera, view, ur, lr, ll, ul);


	// draw lights as dots on the screen
    glEnable(GL_POINT_SMOOTH);
    glPointSize(4.0);
    for (i = 0; i < vars.m_lightArray.size(); ++i) {
        if (vars.m_lightArray[i].m_on) {
            glLoadMatrix(camera.getWorldToCamera());
            if (i == vars.m_lightModify) {
                glColor(Color4(1, 0, 0, 1));
            } else {
                glColor(Color4(0, 0, 1, 1));
            }
            glEnable(GL_POINT_SMOOTH);
            glPointSize(4.0);
            const Vector4& light = vars.m_lightArray[i].m_position;
            glBegin(GL_POINTS);
                glVertex(Vector3(light[0], light[1], light[2]));
            glEnd();
        }
    }


	// draw shadow volumes visually in frame buffer
    if (vars.m_drawShadowVolumes) {
        double viewMatrix[16];
        Array<Plane> frustumPlanes;

        dirtyAllExtrusions(modelArray);

        view.getInfiniteFrustumMatrix(viewMatrix);
        makeFrustumPlanes(camera.m_transformation.translation,
            ur, lr, ll, ul, frustumPlanes);
        
        if (vars.m_shadowOptimizations &&
                (!isDirectionalLight(curLight.m_position)) &&
                (curLight.m_radius > 0) && vars.m_lightAttenuation) {
            int x, y, width, height;
            // set scissor region optimization
            getScreenBoundingRectangle(vector4to3(curLight.m_position),
                curLight.m_radius, camera, view,
                vars.m_winWidth, vars.m_winHeight,
                x, y, width, height);

            glEnable(GL_SCISSOR_TEST);
            glScissor(x, y, width, height);
        }

        for (i = 0; i < modelArray.size(); ++i) {
            if (modelArray[i]->doesCastShadow()) {
                bool frontCapInFrustum, extrusionInFrustum, backCapInFrustum;
                int polyCount;

                modelInFrustum(*modelArray[i], frustumPlanes,
                        curLight.m_position, vars.m_shadowOptimizations,
                        frontCapInFrustum, extrusionInFrustum, backCapInFrustum);

                // set transparency of quads
                if (vars.m_volumeTransparent) {
                    glDepthMask(0);
                    glEnable(GL_BLEND);
                } else {
                    glDepthMask(1);
                    glDisable(GL_BLEND);
                }
                glEnable(GL_CULL_FACE);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glLoadMatrix(camera.getWorldToCamera() *
                    modelArray[i]->m_transformation);

                const Vector4& lightPos = curLight.m_position;
                Vector4 modVector =
                        modelArray[i]->m_transformation.toObjectSpace(lightPos);

                // draw front-facing polygons (red)
                glCullFace(GL_BACK);
                glColor(Color4(1, 0, 0, 0.1));
                modelArray[i]->drawShadowVolume(modVector,
                        false, extrusionInFrustum, false, polyCount,
                        vars.m_shadowOptimizations);
                incrementPolyCount(polyCount, POLY_COUNT_VISIBLE);
                incrementPolyCount(polyCount, POLY_COUNT_TOTAL);

                // draw back-facing polygons (blue)
                glCullFace(GL_FRONT);
                glColor(Color4(0, 0, 1, 0.1));
                modelArray[i]->drawShadowVolume(modVector,
                        false, extrusionInFrustum, false, polyCount,
                        vars.m_shadowOptimizations);
                incrementPolyCount(polyCount, POLY_COUNT_VISIBLE);
                incrementPolyCount(polyCount, POLY_COUNT_TOTAL);


                if (!vars.m_volumeTransparent) {
                    // render polygon outlines
                    glDisable(GL_CULL_FACE);
                    glDisable(GL_BLEND);
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                    glColor(Color4(0, 0, 0, 1));
                    modelArray[i]->drawShadowVolume(modVector,
                            false, extrusionInFrustum, false, polyCount,
                            vars.m_shadowOptimizations);
                    incrementPolyCount(polyCount, POLY_COUNT_VISIBLE);
                    incrementPolyCount(polyCount, POLY_COUNT_TOTAL);
                }
            }
        }
    }

	// highlight objects that use z-fail
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glColor(Color4(0, 1, 0, 1));
    glDisable(GL_BLEND);
    glDepthMask(0);
    glDepthFunc(GL_LEQUAL);

    if (vars.m_highlightOccluders) {
        for (i = 1; i < modelArray.size(); ++i) {
            int polyCount;
            eShadowMethod method = calculateShadowMethod(*modelArray[i],
                    camera, curLight.m_position, curLight.m_radius,
                    vars.m_lightAttenuation, vars.m_shadowOptimizations,
                    ur, lr, ll, ul);
            if (method == SHADOW_Z_FAIL) {
                 glLoadMatrix(camera.getWorldToCamera() *
                    modelArray[i]->m_transformation);
                 modelArray[i]->useTextures(false);
                 modelArray[i]->drawFaces(polyCount);
                 incrementPolyCount(polyCount, POLY_COUNT_VISIBLE);
                 incrementPolyCount(polyCount, POLY_COUNT_TOTAL);
            }
        }
    }

    glPopAttrib();


    // draw fps counter, and update time array
    char buf[64];
    int lineOffset = 30;
    int nextTimeIndex = ((vars.m_curTimeIndex + 1) % vars.m_numTimeSamples);
    int fps = vars.m_numTimeSamples * 1000.0 /
        (vars.m_timeArray[vars.m_curTimeIndex] -
         vars.m_timeArray[nextTimeIndex]);
    vars.m_curTimeIndex = nextTimeIndex;
    vars.m_timeArray[vars.m_curTimeIndex] = SDL_GetTicks();
    sprintf(buf, "FPS: %d", fps);

	/*
    drawFontString(buf, fontTexture, 25, 30, -12, 50, lineOffset,
            vars.m_winWidth, vars.m_winHeight);
    lineOffset += 30;


    // draw Shadows label
    if (vars.m_drawShadows) {
        sprintf(buf, "Shadows: ON");
    } else {
        sprintf(buf, "Shadows: OFF");
    }
    drawFontString(buf, fontTexture, 25, 30, -12, 50, lineOffset,
            vars.m_winWidth, vars.m_winHeight);
    lineOffset += 30;


    // draw Optimizations label
    if (vars.m_shadowOptimizations) {
        sprintf(buf, "Optimizations: ON");
    } else {
        sprintf(buf, "Optimizations: OFF");
    }
    drawFontString(buf, fontTexture, 25, 30, -12, 50, lineOffset,
            vars.m_winWidth, vars.m_winHeight);
    lineOffset += 30;
	*/

    // draw visible polygon count

	glColor3f(0,0,0);

	for(int i = 0; i < 2; i++) {
		int count = getPolyCount(POLY_COUNT_TOTAL);
		sprintf(buf, "%dk polys rendered", count/1000);
		//drawFontString(buf, fontTexture, 25, 30, -12, 50, lineOffset,
		 //       vars.m_winWidth, vars.m_winHeight);

		drawFontString(buf, fontTexture, 25, 30, -12, 10-i, 10-i,

				vars.m_winWidth, vars.m_winHeight);
		lineOffset += 30;


		// draw total polygon count 
		count = getPolyCount(POLY_COUNT_VISIBLE);
		sprintf(buf, "%dk poly scene", count/1000);
	   // drawFontString(buf, fontTexture, 25, 30, -12, 50, lineOffset,

		//        vars.m_winWidth, vars.m_winHeight);

		drawFontString(buf, fontTexture, 25, 30, -12, 410-i, 10-i,

				vars.m_winWidth, vars.m_winHeight);

		glColor3f(1,1,1);

	}

	


    lineOffset += 30;
	
}


