#include "DXUT.h"
#include "Platform.h"
#include "Wind.h"

namespace {
	float powerToAngle(float p)
	{
		return p * D3DX_PI * 0.5f;
	}
}

float windWithPseudoTurbulence(float t)
{
	// y = cos(PI*x) * cos(PI*3*x) * cos(PI*5*x) * cos(PI*7*x) - 0.1*sin(PI*25*x)

	const size_t coefs = 4;

	float f = 1.0f;
	float i = 1.0f;
	for(size_t q = 0; q < coefs; ++q)
	{
		f *= cosf(D3DX_PI * t * i);
		i += 2.0f;
	}

	const float a = 25.0f;
	const float b = 0.1f;
	f -= b * sinf(D3DX_PI * t * a);
	return f;
}

float windWithPseudoTurbulence2(float t)
{
	// y = (sin(PI*x) + sin(PI*3*x) + sin(PI*5*x) + sin(PI*7*x)) / 4

	const size_t coefs = 4;
	const float invCoefs = 1.0f / (float)coefs;

	float f = 0.0f;
	float i = 1.0f;
	for(size_t q = 0; q < coefs; ++q)
	{
		f += sinf(D3DX_PI * t * i);
		i += 2.0f;
	}
	return f * invCoefs;
}

float windSmoothWithSlightNoise(float t)
{
	// y = (cos(PI*x)^2 * cos(PI*3*x) * cos(PI*5*x) - 0.02*sin(PI*25*x)

	float f = 1.0f;
	f = pow(cosf(D3DX_PI * t), 2.0f) * cos(D3DX_PI * t * 3.0f) * cos(D3DX_PI * t * 5.0f);

	const float a = 25.0f;
	const float b = 0.02f;
	f -= b * sinf(D3DX_PI * t * a);

	return f;
}

float windPeriodicWithNoise(float t)
{
	// y = (cos(PI*x)^2 * cos(PI*3*x) * cos(PI*5*x) - 0.1*sin(PI*25*x)

	float f = 1.0f;
	f = cosf(D3DX_PI * t * 10.0f);

	const float a = 25.0f;
	const float b = 0.1f;
	f -= b * sinf(D3DX_PI * t * a);

	return f;
}

D3DXQUATERNION calcWindRotation(D3DXVECTOR2 const& windDirection, D3DXVECTOR2 const& windPower)
{
	float angleX = powerToAngle(windPower.x);
	D3DXVECTOR3 rotWindDir = D3DXVECTOR3(windDirection.x, windDirection.y, 0.0f);
	D3DXVECTOR3 windTangent = D3DXVECTOR3(-rotWindDir.y, rotWindDir.x, 0.0f);
	
	D3DXQUATERNION qr;
	D3DXQuaternionRotationAxis(&qr, &windTangent, angleX);

	D3DXMATRIX rMatrix;
	D3DXMatrixRotationQuaternion(&rMatrix, &qr);
	D3DXVec3TransformNormal(&rotWindDir, D3DXVec3Normalize(&rotWindDir, &rotWindDir), &rMatrix);

	float angleY = 0;
	angleY = powerToAngle(windPower.y);
	D3DXQUATERNION qr2;
	D3DXQuaternionRotationAxis(&qr2, &rotWindDir, angleY);
	return qr * qr2;
}