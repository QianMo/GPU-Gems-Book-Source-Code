/*
s_p_oneil@hotmail.com
Copyright (c) 2000, Sean O'Neil
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of this project nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include "Master.h"
#include "GameApp.h"
#include "GameEngine.h"
#include "GLUtil.h"


CGameEngine::CGameEngine()
{
	m_bUseHDR = true;

	//GetApp()->MessageBox((const char *)glGetString(GL_EXTENSIONS));
	GLUtil()->Init();
	m_fFont.Init(GetGameApp()->GetHDC());
	m_nPolygonMode = GL_FILL;

	m_pBuffer.Init(1024, 1024, 0);
	m_pBuffer.MakeCurrent();
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_CULL_FACE);

	//glEnable(GL_MULTISAMPLE_ARB);

	// Read last camera position and orientation from registry
	CVector vPos(0, 0, 25);
	const char *psz = GetApp()->GetProfileString("Camera", "Position", NULL);
	if(psz)
		sscanf(psz, "%f, %f, %f", &vPos.x, &vPos.y, &vPos.z);
	m_3DCamera.SetPosition(CDoubleVector(vPos));
	CQuaternion qOrientation(0.0f, 0.0f, 0.0f, 1.0f);
	psz = GetApp()->GetProfileString("Camera", "Orientation", NULL);
	if(psz)
		sscanf(psz, "%f, %f, %f, %f", &qOrientation.x, &qOrientation.y, &qOrientation.z, &qOrientation.w);
	qOrientation.Normalize();
	m_3DCamera = qOrientation;

	m_vLight = CVector(0, 0, 1000);
	m_vLightDirection = m_vLight / m_vLight.Magnitude();
	CTexture::InitStaticMembers(238653, 256);

	m_nSamples = 3;		// Number of sample rays to use in integral equation
	m_Kr = 0.0025f;		// Rayleigh scattering constant
	m_Kr4PI = m_Kr*4.0f*PI;
	m_Km = 0.0010f;		// Mie scattering constant
	m_Km4PI = m_Km*4.0f*PI;
	m_ESun = 20.0f;		// Sun brightness constant
	m_g = -0.990f;		// The Mie phase asymmetry factor
	m_fExposure = 2.0f;

	m_fInnerRadius = 10.0f;
	m_fOuterRadius = 10.25f;
	m_fScale = 1 / (m_fOuterRadius - m_fInnerRadius);

	m_fWavelength[0] = 0.650f;		// 650 nm for red
	m_fWavelength[1] = 0.570f;		// 570 nm for green
	m_fWavelength[2] = 0.475f;		// 475 nm for blue
	m_fWavelength4[0] = powf(m_fWavelength[0], 4.0f);
	m_fWavelength4[1] = powf(m_fWavelength[1], 4.0f);
	m_fWavelength4[2] = powf(m_fWavelength[2], 4.0f);

	m_fRayleighScaleDepth = 0.25f;
	m_fMieScaleDepth = 0.1f;
	m_pbOpticalDepth.MakeOpticalDepthBuffer(m_fInnerRadius, m_fOuterRadius, m_fRayleighScaleDepth, m_fMieScaleDepth);

	m_shSkyFromSpace.Load("SkyFromSpace");
	m_shSkyFromAtmosphere.Load("SkyFromAtmosphere");
	m_shGroundFromSpace.Load("GroundFromSpace");
	m_shGroundFromAtmosphere.Load("GroundFromAtmosphere");
	m_shSpaceFromSpace.Load("SpaceFromSpace");
	m_shSpaceFromAtmosphere.Load("SpaceFromAtmosphere");


	CPixelBuffer pb;
	pb.Init(256, 256, 1);
	pb.MakeGlow2D(40.0f, 0.1f);
	m_tMoonGlow.Init(&pb);

	pb.LoadJPEG("earthmap1k.jpg");
	m_tEarth.Init(&pb);
}

CGameEngine::~CGameEngine()
{
	// Write the camera position and orientation to the registry
	char szBuffer[256];
	sprintf(szBuffer, "%f, %f, %f", m_3DCamera.GetPosition().x, m_3DCamera.GetPosition().y, m_3DCamera.GetPosition().z);
	GetApp()->WriteProfileString("Camera", "Position", szBuffer);
	sprintf(szBuffer, "%f, %f, %f, %f", m_3DCamera.x, m_3DCamera.y, m_3DCamera.z, m_3DCamera.w);
	GetApp()->WriteProfileString("Camera", "Orientation", szBuffer);

	m_pBuffer.Cleanup();
	GLUtil()->Cleanup();
}

void CGameEngine::RenderFrame(int nMilliseconds)
{
	// Determine the FPS
	static char szFrameCount[20] = {0};
	static int nTime = 0;
	static int nFrames = 0;
	nTime += nMilliseconds;
	if(nTime >= 1000)
	{
		m_fFPS = (float)(nFrames * 1000) / (float)nTime;
		sprintf(szFrameCount, "%2.2f FPS", m_fFPS);
		nTime = nFrames = 0;
	}
	nFrames++;

	// Move the camera
	HandleInput(nMilliseconds * 0.001f);

	m_pBuffer.MakeCurrent();
	glViewport(0, 0, 1024, 1024);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();
	glLoadMatrixf(m_3DCamera.GetViewMatrix());

	C3DObject obj;
	glMultMatrixf(obj.GetModelMatrix(&m_3DCamera));

	CVector vCamera = m_3DCamera.GetPosition();
	CVector vUnitCamera = vCamera / vCamera.Magnitude();

	CShaderObject *pSpaceShader = NULL;
	if(vCamera.Magnitude() < m_fOuterRadius)
		pSpaceShader = &m_shSpaceFromAtmosphere;
	else if(vCamera.z > 0.0f)
		pSpaceShader = &m_shSpaceFromSpace;

	if(pSpaceShader)
	{
		pSpaceShader->Enable();
		pSpaceShader->SetUniformParameter3f("v3CameraPos", vCamera.x, vCamera.y, vCamera.z);
		pSpaceShader->SetUniformParameter3f("v3LightPos", m_vLightDirection.x, m_vLightDirection.y, m_vLightDirection.z);
		pSpaceShader->SetUniformParameter3f("v3InvWavelength", 1/m_fWavelength4[0], 1/m_fWavelength4[1], 1/m_fWavelength4[2]);
		pSpaceShader->SetUniformParameter1f("fCameraHeight", vCamera.Magnitude());
		pSpaceShader->SetUniformParameter1f("fCameraHeight2", vCamera.MagnitudeSquared());
		pSpaceShader->SetUniformParameter1f("fInnerRadius", m_fInnerRadius);
		pSpaceShader->SetUniformParameter1f("fInnerRadius2", m_fInnerRadius*m_fInnerRadius);
		pSpaceShader->SetUniformParameter1f("fOuterRadius", m_fOuterRadius);
		pSpaceShader->SetUniformParameter1f("fOuterRadius2", m_fOuterRadius*m_fOuterRadius);
		pSpaceShader->SetUniformParameter1f("fKrESun", m_Kr*m_ESun);
		pSpaceShader->SetUniformParameter1f("fKmESun", m_Km*m_ESun);
		pSpaceShader->SetUniformParameter1f("fKr4PI", m_Kr4PI);
		pSpaceShader->SetUniformParameter1f("fKm4PI", m_Km4PI);
		pSpaceShader->SetUniformParameter1f("fScale", 1.0f / (m_fOuterRadius - m_fInnerRadius));
		pSpaceShader->SetUniformParameter1f("fScaleDepth", m_fRayleighScaleDepth);
		pSpaceShader->SetUniformParameter1f("fScaleOverScaleDepth", (1.0f / (m_fOuterRadius - m_fInnerRadius)) / m_fRayleighScaleDepth);
		pSpaceShader->SetUniformParameter1f("g", m_g);
		pSpaceShader->SetUniformParameter1f("g2", m_g*m_g);
		pSpaceShader->SetUniformParameter1i("s2Test", 0);
	}

	m_tMoonGlow.Enable();
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex3f(-4.0f, 4.0f, -50.0f);
	glTexCoord2f(0, 1);
	glVertex3f(-4.0f, -4.0f, -50.0f);
	glTexCoord2f(1, 1);
	glVertex3f(4.0f, -4.0f, -50.0f);
	glTexCoord2f(1, 0);
	glVertex3f(4.0f, 4.0f, -50.0f);
	glEnd();
	m_tMoonGlow.Disable();

	if(pSpaceShader)
		pSpaceShader->Disable();

	CShaderObject *pGroundShader;
	if(vCamera.Magnitude() >= m_fOuterRadius)
		pGroundShader = &m_shGroundFromSpace;
	else
		pGroundShader = &m_shGroundFromAtmosphere;

	pGroundShader->Enable();
	pGroundShader->SetUniformParameter3f("v3CameraPos", vCamera.x, vCamera.y, vCamera.z);
	pGroundShader->SetUniformParameter3f("v3LightPos", m_vLightDirection.x, m_vLightDirection.y, m_vLightDirection.z);
	pGroundShader->SetUniformParameter3f("v3InvWavelength", 1/m_fWavelength4[0], 1/m_fWavelength4[1], 1/m_fWavelength4[2]);
	pGroundShader->SetUniformParameter1f("fCameraHeight", vCamera.Magnitude());
	pGroundShader->SetUniformParameter1f("fCameraHeight2", vCamera.MagnitudeSquared());
	pGroundShader->SetUniformParameter1f("fInnerRadius", m_fInnerRadius);
	pGroundShader->SetUniformParameter1f("fInnerRadius2", m_fInnerRadius*m_fInnerRadius);
	pGroundShader->SetUniformParameter1f("fOuterRadius", m_fOuterRadius);
	pGroundShader->SetUniformParameter1f("fOuterRadius2", m_fOuterRadius*m_fOuterRadius);
	pGroundShader->SetUniformParameter1f("fKrESun", m_Kr*m_ESun);
	pGroundShader->SetUniformParameter1f("fKmESun", m_Km*m_ESun);
	pGroundShader->SetUniformParameter1f("fKr4PI", m_Kr4PI);
	pGroundShader->SetUniformParameter1f("fKm4PI", m_Km4PI);
	pGroundShader->SetUniformParameter1f("fScale", 1.0f / (m_fOuterRadius - m_fInnerRadius));
	pGroundShader->SetUniformParameter1f("fScaleDepth", m_fRayleighScaleDepth);
	pGroundShader->SetUniformParameter1f("fScaleOverScaleDepth", (1.0f / (m_fOuterRadius - m_fInnerRadius)) / m_fRayleighScaleDepth);
	pGroundShader->SetUniformParameter1f("g", m_g);
	pGroundShader->SetUniformParameter1f("g2", m_g*m_g);
	pGroundShader->SetUniformParameter1i("s2Test", 0);

	/*
	if(vCamera.z < 0 && pGroundShader == &m_shGroundFromAtmosphere)
	{
		// Try setting the moon as a light source
		CVector vLightDir = CVector(0.0f, 0.0f, -50.0f) - vCamera;
		vLightDir.Normalize();
		pGroundShader->SetUniformParameter3f("v3LightPos", vLightDir.x, vLightDir.y, vLightDir.z);
		pGroundShader->SetUniformParameter1f("fKrESun", m_Kr*m_ESun*0.1f);
		pGroundShader->SetUniformParameter1f("fKmESun", 10.0f*m_Km*m_ESun*0.1f);
		pGroundShader->SetUniformParameter1f("g", -0.75f);
		pGroundShader->SetUniformParameter1f("g2", -0.75f * -0.75f);
	}
	*/
	GLUquadricObj *pSphere = gluNewQuadric();
	m_tEarth.Enable();
	gluSphere(pSphere, m_fInnerRadius, 100, 50);
	m_tEarth.Disable();
	gluDeleteQuadric(pSphere);
	pGroundShader->Disable();

	CShaderObject *pSkyShader;
	if(vCamera.Magnitude() >= m_fOuterRadius)
		pSkyShader = &m_shSkyFromSpace;
	else
		pSkyShader = &m_shSkyFromAtmosphere;

	pSkyShader->Enable();
	pSkyShader->SetUniformParameter3f("v3CameraPos", vCamera.x, vCamera.y, vCamera.z);
	pSkyShader->SetUniformParameter3f("v3LightPos", m_vLightDirection.x, m_vLightDirection.y, m_vLightDirection.z);
	pSkyShader->SetUniformParameter3f("v3InvWavelength", 1/m_fWavelength4[0], 1/m_fWavelength4[1], 1/m_fWavelength4[2]);
	pSkyShader->SetUniformParameter1f("fCameraHeight", vCamera.Magnitude());
	pSkyShader->SetUniformParameter1f("fCameraHeight2", vCamera.MagnitudeSquared());
	pSkyShader->SetUniformParameter1f("fInnerRadius", m_fInnerRadius);
	pSkyShader->SetUniformParameter1f("fInnerRadius2", m_fInnerRadius*m_fInnerRadius);
	pSkyShader->SetUniformParameter1f("fOuterRadius", m_fOuterRadius);
	pSkyShader->SetUniformParameter1f("fOuterRadius2", m_fOuterRadius*m_fOuterRadius);
	pSkyShader->SetUniformParameter1f("fKrESun", m_Kr*m_ESun);
	pSkyShader->SetUniformParameter1f("fKmESun", m_Km*m_ESun);
	pSkyShader->SetUniformParameter1f("fKr4PI", m_Kr4PI);
	pSkyShader->SetUniformParameter1f("fKm4PI", m_Km4PI);
	pSkyShader->SetUniformParameter1f("fScale", 1.0f / (m_fOuterRadius - m_fInnerRadius));
	pSkyShader->SetUniformParameter1f("fScaleDepth", m_fRayleighScaleDepth);
	pSkyShader->SetUniformParameter1f("fScaleOverScaleDepth", (1.0f / (m_fOuterRadius - m_fInnerRadius)) / m_fRayleighScaleDepth);
	pSkyShader->SetUniformParameter1f("g", m_g);
	pSkyShader->SetUniformParameter1f("g2", m_g*m_g);

	/*
	if(vCamera.z < 0 && pSkyShader == &m_shSkyFromAtmosphere)
	{
		// Try setting the moon as a light source
		CVector vLightDir = CVector(0.0f, 0.0f, -50.0f) - vCamera;
		vLightDir.Normalize();
		pSkyShader->SetUniformParameter3f("v3LightPos", vLightDir.x, vLightDir.y, vLightDir.z);
		pSkyShader->SetUniformParameter1f("fKrESun", m_Kr*m_ESun*0.1f);
		pSkyShader->SetUniformParameter1f("fKmESun", 10.0f*m_Km*m_ESun*0.1f);
		pSkyShader->SetUniformParameter1f("g", -0.75f);
		pSkyShader->SetUniformParameter1f("g2", -0.75f * -0.75f);
	}
	*/
	glFrontFace(GL_CW);
	//glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	pSphere = gluNewQuadric();
	gluSphere(pSphere, m_fOuterRadius, 100, 50);
	gluDeleteQuadric(pSphere);

	//glDisable(GL_BLEND);
	glFrontFace(GL_CCW);
	pSkyShader->Disable();

	glPopMatrix();
	glFlush();

	//CTexture tTest;
	//tTest.InitCopy(0, 0, 1024, 1024);

	GLUtil()->MakeCurrent();
	glViewport(0, 0, GetGameApp()->GetWidth(), GetGameApp()->GetHeight());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_LIGHTING);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	//tTest.Enable();
	m_pBuffer.BindTexture(m_fExposure, m_bUseHDR);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(0, 0);	// For rect texture, can't use 1 as the max texture coord
	glTexCoord2f(1, 0); glVertex2f(1, 0);
	glTexCoord2f(1, 1); glVertex2f(1, 1);
	glTexCoord2f(0, 1); glVertex2f(0, 1);
	glEnd();
	m_pBuffer.ReleaseTexture();
	//tTest.Disable();

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_LIGHTING);

	// Draw info in the top-left corner
	char szBuffer[256];
	m_fFont.Begin();
	glColor3d(1.0, 1.0, 1.0);
	m_fFont.SetPosition(0, 0);
	m_fFont.Print(szFrameCount);
	m_fFont.SetPosition(0, 15);
	sprintf(szBuffer, "Samples (+/-): %d", m_nSamples);
	m_fFont.Print(szBuffer);
	m_fFont.SetPosition(0, 30);
	sprintf(szBuffer, "Kr (1/Sh+1): %-4.4f", m_Kr);
	m_fFont.Print(szBuffer);
	m_fFont.SetPosition(0, 45);
	sprintf(szBuffer, "Km (2/Sh+2): %-4.4f", m_Km);
	m_fFont.Print(szBuffer);
	m_fFont.SetPosition(0, 60);
	sprintf(szBuffer, "g (3/Sh+3): %-3.3f", m_g);
	m_fFont.Print(szBuffer);
	m_fFont.SetPosition(0, 75);
	sprintf(szBuffer, "ESun (4/Sh+4): %-1.1f", m_ESun);
	m_fFont.Print(szBuffer);
	m_fFont.SetPosition(0, 90);
	sprintf(szBuffer, "Red (5/Sh+5): %-3.3f", m_fWavelength[0]);
	m_fFont.Print(szBuffer);
	m_fFont.SetPosition(0, 105);
	sprintf(szBuffer, "Green (6/Sh+6): %-3.3f", m_fWavelength[1]);
	m_fFont.Print(szBuffer);
	m_fFont.SetPosition(0, 120);
	sprintf(szBuffer, "Blue (7/Sh+7): %-3.3f", m_fWavelength[2]);
	m_fFont.Print(szBuffer);
	m_fFont.SetPosition(0, 135);
	sprintf(szBuffer, "Exposure (8/Sh+8): %-2.2f", m_fExposure);
	m_fFont.Print(szBuffer);
	m_fFont.End();
	glFlush();
}

void CGameEngine::OnChar(WPARAM c)
{
	switch(c)
	{
		case 'p':
			m_nPolygonMode = (m_nPolygonMode == GL_FILL) ? GL_LINE : GL_FILL;
			glPolygonMode(GL_FRONT, m_nPolygonMode);
			break;
		case 'h':
			m_bUseHDR = !m_bUseHDR;
			break;
		case '+':
			m_nSamples++;
			break;
		case '-':
			m_nSamples--;
			break;
	}
}

void CGameEngine::HandleInput(float fSeconds)
{
	if((GetKeyState('1') & 0x8000))
	{
		if((GetKeyState(VK_SHIFT) & 0x8000))
			m_Kr = Max(0.0f, m_Kr - 0.0001f);
		else
			m_Kr += 0.0001f;
		m_Kr4PI = m_Kr*4.0f*PI;
	}
	else if((GetKeyState('2') & 0x8000))
	{
		if((GetKeyState(VK_SHIFT) & 0x8000))
			m_Km = Max(0.0f, m_Km - 0.0001f);
		else
			m_Km += 0.0001f;
		m_Km4PI = m_Km*4.0f*PI;
	}
	else if((GetKeyState('3') & 0x8000))
	{
		if((GetKeyState(VK_SHIFT) & 0x8000))
			m_g = Max(-1.0f, m_g-0.001f);
		else
			m_g = Min(1.0f, m_g+0.001f);
	}
	else if((GetKeyState('4') & 0x8000))
	{
		if((GetKeyState(VK_SHIFT) & 0x8000))
			m_ESun = Max(0.0f, m_ESun - 0.1f);
		else
			m_ESun += 0.1f;
	}
	else if((GetKeyState('5') & 0x8000))
	{
		if((GetKeyState(VK_SHIFT) & 0x8000))
			m_fWavelength[0] = Max(0.001f, m_fWavelength[0] -= 0.001f);
		else
			m_fWavelength[0] += 0.001f;
		m_fWavelength4[0] = powf(m_fWavelength[0], 4.0f);
	}
	else if((GetKeyState('6') & 0x8000))
	{
		if((GetKeyState(VK_SHIFT) & 0x8000))
			m_fWavelength[1] = Max(0.001f, m_fWavelength[1] -= 0.001f);
		else
			m_fWavelength[1] += 0.001f;
		m_fWavelength4[1] = powf(m_fWavelength[1], 4.0f);
	}
	else if((GetKeyState('7') & 0x8000))
	{
		if((GetKeyState(VK_SHIFT) & 0x8000))
			m_fWavelength[2] = Max(0.001f, m_fWavelength[2] -= 0.001f);
		else
			m_fWavelength[2] += 0.001f;
		m_fWavelength4[2] = powf(m_fWavelength[2], 4.0f);
	}
	else if((GetKeyState('8') & 0x8000))
	{
		if((GetKeyState(VK_SHIFT) & 0x8000))
			m_fExposure = Max(0.1f, m_fExposure-0.1f);
		else
			m_fExposure += 0.1f;
	}


	const float ROTATE_SPEED = 1.0f;

	// Turn left/right means rotate around the up axis
	if((GetKeyState(VK_NUMPAD6) & 0x8000) || (GetKeyState(VK_RIGHT) & 0x8000))
		m_3DCamera.Rotate(m_3DCamera.GetUpAxis(), fSeconds * -ROTATE_SPEED);
	if((GetKeyState(VK_NUMPAD4) & 0x8000) || (GetKeyState(VK_LEFT) & 0x8000))
		m_3DCamera.Rotate(m_3DCamera.GetUpAxis(), fSeconds * ROTATE_SPEED);

	// Turn up/down means rotate around the right axis
	if((GetKeyState(VK_NUMPAD8) & 0x8000) || (GetKeyState(VK_UP) & 0x8000))
		m_3DCamera.Rotate(m_3DCamera.GetRightAxis(), fSeconds * -ROTATE_SPEED);
	if((GetKeyState(VK_NUMPAD2) & 0x8000) || (GetKeyState(VK_DOWN) & 0x8000))
		m_3DCamera.Rotate(m_3DCamera.GetRightAxis(), fSeconds * ROTATE_SPEED);

	// Roll means rotate around the view axis
	if(GetKeyState(VK_NUMPAD7) & 0x8000)
		m_3DCamera.Rotate(m_3DCamera.GetViewAxis(), fSeconds * -ROTATE_SPEED);
	if(GetKeyState(VK_NUMPAD9) & 0x8000)
		m_3DCamera.Rotate(m_3DCamera.GetViewAxis(), fSeconds * ROTATE_SPEED);

#define THRUST		1.0f	// Acceleration rate due to thrusters (units/s*s)
#define RESISTANCE	0.1f	// Damping effect on velocity

	// Handle acceleration keys
	CVector vAccel(0.0f);
	if(GetKeyState(VK_SPACE) & 0x8000)
		m_3DCamera.SetVelocity(CVector(0.0f));	// Full stop
	else
	{
		// Add camera's acceleration due to thrusters
		float fThrust = THRUST;
		if(GetKeyState(VK_CONTROL) & 0x8000)
			fThrust *= 10.0f;

		// Thrust forward/reverse affects velocity along the view axis
		if(GetKeyState('W') & 0x8000)
			vAccel += m_3DCamera.GetViewAxis() * fThrust;
		if(GetKeyState('S') & 0x8000)
			vAccel += m_3DCamera.GetViewAxis() * -fThrust;

		// Thrust left/right affects velocity along the right axis
		if(GetKeyState('D') & 0x8000)
			vAccel += m_3DCamera.GetRightAxis() * fThrust;
		if(GetKeyState('A') & 0x8000)
			vAccel += m_3DCamera.GetRightAxis() * -fThrust;

		// Thrust up/down affects velocity along the up axis
//#define WORLD_UPDOWN
#ifdef WORLD_UPDOWN
		CVector v = m_3DCamera.GetPosition();
		v.Normalize();
		if(GetKeyState('M') & 0x8000)
			vAccel += v * fThrust;
		if(GetKeyState('N') & 0x8000)
			vAccel += v * -fThrust;
#else
		if(GetKeyState('M') & 0x8000)
			vAccel += m_3DCamera.GetUpAxis() * fThrust;
		if(GetKeyState('N') & 0x8000)
			vAccel += m_3DCamera.GetUpAxis() * -fThrust;
#endif

		m_3DCamera.Accelerate(vAccel, fSeconds, RESISTANCE);
		CVector vPos = m_3DCamera.GetPosition();
		float fMagnitude = vPos.Magnitude();
		if(fMagnitude < m_fInnerRadius)
		{
			vPos *= (m_fInnerRadius * (1 + DELTA)) / fMagnitude;
			m_3DCamera.SetPosition(CDoubleVector(vPos.x, vPos.y, vPos.z));
			m_3DCamera.SetVelocity(-m_3DCamera.GetVelocity());
		}
	}
}

