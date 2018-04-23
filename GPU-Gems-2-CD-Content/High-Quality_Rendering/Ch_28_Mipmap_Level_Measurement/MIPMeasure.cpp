// Copyright 2004, Climax Online.
#define WIN32_LEAN_AND_MEAN

#define D3D_DEBUG_INFO
#include <d3dx9.h>
#include <mmsystem.h>
#include <assert.h>
#include <strstream>

const DWORD nVtxOnEdge = 100;
const DWORD nVtx = nVtxOnEdge * nVtxOnEdge;
UINT nIndices = 0, nPrims = 0;

// We test four levels, plus there's a query for the reference.  If you change
// the number of levels tested, you need to change the calibration texture and
// the pixel shader.
const int N_QUERIES = 5;
const int REF_QUERY = N_QUERIES-1;

HWND					hWnd       = NULL;
LPDIRECT3D9				pD3D       = NULL;
LPDIRECT3DDEVICE9		pD3DDevice = NULL;
LPDIRECT3DVERTEXBUFFER9 pVB      = NULL;
LPDIRECT3DINDEXBUFFER9  pIB      = NULL;
LPDIRECT3DTEXTURE9		pTexture = NULL;
LPD3DXFONT				pFont    = NULL;
LPDIRECT3DPIXELSHADER9	pPixelShader = NULL;

LPDIRECT3DQUERY9		pQueries[N_QUERIES];
bool					haveResult[N_QUERIES];
DWORD					queryResults[N_QUERIES];
bool					isMeasured = false;
bool					isWaiting  = false;
int						visualisedDIP = REF_QUERY;
int						frameCount = 0;
float					threshold = 10;
int						measuredLevel = 0;
bool					showInstructions = true;

const char* copyright = "Copyright Climax Online 2004";
const char* author    = "Author: Iain Cantlay";

const char* instructions = 
	"Move the object and/or eye to see how the query results vary.\n"
	"Key presses are:\n"
	"    F1: toggle these instructions\n"
	"    up/down: move the eyepoint in/out\n"
	"    left/right: rotate the object\n"
	"    page up/down: alter the threshold percentage in 5% increments\n"
	"    space: step through visualisations of each draw call\n"
	"\n";

#define D3DFVF_MY_VERTEX (D3DFVF_XYZ | D3DFVF_DIFFUSE | D3DFVF_TEX1)

struct Vertex
{
	float x, y, z;
    DWORD color;
	float u, v;
};

float eyeZ = 8, eyeYaw = D3DX_PI / 2.0f;
bool isPaused = false;

typedef unsigned int	uint;
typedef unsigned short	Index;

// Windows message processing including key presses.
LRESULT CALLBACK WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
	{
        case WM_KEYDOWN:
		{
			switch (wParam)
			{
				case VK_ESCAPE:
					PostQuitMessage(0);
				break;

				case VK_F1:
					showInstructions = !showInstructions;
					break;

				case VK_SPACE:
					visualisedDIP = (visualisedDIP+1) % N_QUERIES;
					break;

				case VK_LEFT:
					eyeYaw += 0.04f;
					break;
				case VK_RIGHT:
					eyeYaw -= 0.04f;
					break;

				case VK_UP:
					eyeZ -= 0.2f;
					break;
				case VK_DOWN:
					eyeZ += 0.2f;
					break;

				case VK_PRIOR:
					threshold += 5;
					break;
				case VK_NEXT:
					threshold -= 5;
					break;

				case VK_PAUSE:
					isPaused = !isPaused;
					break;
			}
			break;
		}

		case WM_CLOSE:
			PostQuitMessage(0);	
			break;
		
        case WM_DESTROY:
            PostQuitMessage(0);
	        break;

		default:
			return DefWindowProc(hWnd, msg, wParam, lParam);
	}

	// Keep the threshold within the 0 to 100 range.
	if (threshold > 100)
		threshold = 100;
	else if (threshold < 0)
		threshold = 0;

	return 0;
}

// We use a sigmoid function because 1) it's easy to code and 2) it looks 
// approximately like a sloping piece of terrain.
float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(x));
}

void createVB()
{
	const float width  = 4;
	const float height = 3;
	const DWORD nBytes = nVtx * sizeof(Vertex);
	pD3DDevice->CreateVertexBuffer(nBytes, D3DUSAGE_WRITEONLY, D3DFVF_MY_VERTEX, D3DPOOL_DEFAULT, &pVB, NULL);

	Vertex* pVertices = NULL;
	pVB->Lock(0, nBytes, (void**)&pVertices, 0);

	const D3DCOLOR white = D3DCOLOR_COLORVALUE(1.0, 1.0, 1.0, 1.0);

	DWORD nSet = 0;
	for (int i=0; i!=nVtxOnEdge; ++i)
	{
		const float u = static_cast<float>(i) / static_cast<float>(nVtxOnEdge-1);
		const float x = width * (u - 0.5f);
		const float z = sigmoid(3 * x);
		const D3DCOLOR col = D3DCOLOR_COLORVALUE(1.0, z, 0.0, 1.0);

		for (int j=0; j!=nVtxOnEdge; ++j)
		{
			const float v = static_cast<float>(j) / static_cast<float>(nVtxOnEdge-1);
			const float y = width * (v - 0.5f);
			pVertices->x = x;
			pVertices->y = height * (z - 0.5f);
			pVertices->z = y;
			pVertices->u = v;
			pVertices->v = u;
			pVertices->color = white;
			++nSet;
			++pVertices;
		}
	}

	assert(nSet == nVtx);
	pVB->Unlock();
}

// Computes the index of some vertex within a VB.
uint indexOf(uint x, uint y, uint vertsPerRow)
{
	return x + y * vertsPerRow;
}

DWORD populateIB(Index* pIx)
{
	// This creates one long tri-strip for all the polygons in the mesh.
	const uint nPolyRows    = nVtxOnEdge - 1;
	const uint nVertsPerRow = nVtxOnEdge;

	DWORD result = 0;
	if (pIx)
	{
		for(uint row = nPolyRows; row > 0; --row)
		{
			// If this isn't the first row, add a degenerate stitch to the start of
			// the next...
			if (nPolyRows != row)
			{
				pIx[result++] = indexOf(0,                row, nVertsPerRow);
				pIx[result++] = indexOf(nVertsPerRow - 1, row, nVertsPerRow);
			}

			// Then populate the rest of the row as a tri-strip
			for(uint col = nVertsPerRow; col > 0; --col)
			{
				pIx[result++] = indexOf(col - 1, row,     nVertsPerRow);
				pIx[result++] = indexOf(col - 1, row - 1, nVertsPerRow);
			}
		}

		assert(result == populateIB(NULL));
	}
	else
	{
		const uint nStitches = 2 * (nPolyRows - 1);
		result = nPolyRows * 2 * nVertsPerRow + nStitches;
	}

	return result;
}

void createIB()
{
	nIndices = populateIB(NULL);
	nPrims = nIndices - 2;
	const DWORD nBytes = nIndices * 2;
	pD3DDevice->CreateIndexBuffer(nBytes, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &pIB, NULL);

	Index* pIndices = NULL;
	pIB->Lock(0, nBytes, (void**)&pIndices, 0);
	populateIB(pIndices);
	pIB->Unlock();
}

void createFontStuff()
{
	// For displaying the results and instructions.
	D3DXCreateFont(pD3DDevice, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, 
		OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, 
		"Arial", &pFont);
}

void createDevice()
{
	pD3D = Direct3DCreate9(D3D_SDK_VERSION);

	const D3DDEVTYPE devType = D3DDEVTYPE_HAL;
	const UINT adapterNo = D3DADAPTER_DEFAULT;

	D3DDISPLAYMODE displayMode;
	pD3D->GetAdapterDisplayMode(adapterNo, &displayMode);

	D3DPRESENT_PARAMETERS presentParams;
	ZeroMemory(&presentParams, sizeof(presentParams));

	presentParams.Windowed               = TRUE;
	presentParams.SwapEffect             = D3DSWAPEFFECT_DISCARD;
	presentParams.BackBufferFormat       = displayMode.Format;
	presentParams.EnableAutoDepthStencil = TRUE;
	presentParams.AutoDepthStencilFormat = D3DFMT_D16;
	presentParams.PresentationInterval   = D3DPRESENT_INTERVAL_IMMEDIATE;

	pD3D->CreateDevice(adapterNo, devType, hWnd,
		D3DCREATE_HARDWARE_VERTEXPROCESSING,
		&presentParams, &pD3DDevice);
}

void createPixelShader()
{
	LPD3DXBUFFER pCode;
	const char* path = "MIPMeasure.psh";
	const DWORD flags = 0;
	D3DXAssembleShaderFromFile(path, NULL, NULL, flags, &pCode, NULL);

	pD3DDevice->CreatePixelShader((DWORD*)pCode->GetBufferPointer(), &pPixelShader);
	pCode->Release();
}

void createTexture()
{
	const char* filename = "MIPMeasure512.dds";
	D3DXCreateTextureFromFile(pD3DDevice, filename, &pTexture);
}

void createQueries()
{
	for (int i=0; i!=N_QUERIES; ++i)
	{
		pQueries[i] = NULL;
		pD3DDevice->CreateQuery(D3DQUERYTYPE_OCCLUSION, &(pQueries[i]));
		haveResult[i] = false;
		queryResults[i] = 0;
	}

	// If someone tries to run it on a GeForce3 or older...
	if (!pQueries[0])
	{
		const char* msg = 
			"Your card does not appear to support occlusion queries.\n"
			"This program will now exit.";
		MessageBox(hWnd, msg, "Error", MB_OK | MB_ICONERROR);
	}
}

void setUpRenderState()
{
	pD3DDevice->SetRenderState(D3DRS_LIGHTING, FALSE);
	pD3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);

	D3DXMATRIX proj;
	D3DXMatrixPerspectiveFovLH(&proj, D3DXToRadian(50), 1, 1, 300);
	pD3DDevice->SetTransform(D3DTS_PROJECTION, &proj);

	// MIP filtering is our whole raison d'etre.
	pD3DDevice->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
	pD3DDevice->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
	pD3DDevice->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);

	// We use alpha testing to vary the number of pixels that make it through
	// to the occlusion query result.
	const DWORD refValue = 0x7f;
	pD3DDevice->SetRenderState(D3DRS_ALPHAREF, refValue);
	pD3DDevice->SetRenderState(D3DRS_ALPHAFUNC, D3DCMP_GREATEREQUAL);
	pD3DDevice->SetRenderState(D3DRS_ALPHATESTENABLE, false);
}

void init()
{
	createDevice();
	createQueries();
	createVB();
	createIB();
	createTexture();
	createPixelShader();
	createFontStuff();
}

void shutDown()
{
	if (pVB)
		pVB->Release(); 

	if (pIB)
		pIB->Release(); 

	if (pTexture)
		pTexture->Release();

	if (pFont)
		pFont->Release();

	if (pPixelShader)
		pPixelShader->Release();

	for (int i=0; i!=N_QUERIES; ++i)
	{
		if (pQueries[i])
			pQueries[i]->Release();
	}

    if (pD3DDevice)
        pD3DDevice->Release();

    if (pD3D)
        pD3D->Release();
}

void composeWorldTransform(D3DXMATRIX* pWorld)
{
	// Make the viewer look at our object.
	D3DXQUATERNION quat;
	D3DXQuaternionRotationYawPitchRoll(&quat, eyeYaw, 0, 0);
	const D3DXVECTOR3 pos(0,-1,eyeZ);
	
	D3DXMatrixTransformation(pWorld, NULL, NULL, NULL, NULL, &quat, &pos);
}

void checkQueryResults()
{
	isMeasured = true;

	for (int i=0; i!=N_QUERIES; ++i)
	{
		// The queries come back asyncronously, so we have to wait until they're all done.
		if (S_OK == pQueries[i]->GetData(&queryResults[i], sizeof(queryResults[i]), 0))
			haveResult[i] = true;

		isMeasured = (isMeasured && haveResult[i]);
	}

	// isWaiting is always equal to !isMeasured, except when we're waiting for the
	// very first set of results.
	isWaiting = !isMeasured;

	// Once we have all the results, pick a predominantly visible level.
	if (isMeasured)
	{
		const float ref = static_cast<float>(queryResults[REF_QUERY]);

		// If nothing passes the threshold test, the result is the first non-measured level.
		measuredLevel = REF_QUERY;

		for (int i=0; i!=REF_QUERY; ++i)
		{
			const float percent = 100.0f * static_cast<float>(queryResults[i]) / ref;

			if (percent > threshold)
			{
				measuredLevel = i;
				break;
			}
		}
	}
}

void reportResults()
{
	std::ostrstream ostr;

	ostr << copyright << ", " << author << "\n";

	if (showInstructions)
		ostr << instructions;

	const float ref = static_cast<float>(queryResults[REF_QUERY]);

	for (int i=0; i!=N_QUERIES; ++i)
	{
		const float percent = 100.0f * static_cast<float>(queryResults[i]) / ref;
		ostr << "[" << i << "] " << queryResults[i] << " pixels  (" << percent << "%)";

		if (visualisedDIP == i)
			ostr << "  (shown)\n";
		else
			ostr << "\n";
	}

	ostr << "\n";
	ostr << "threshold = " << threshold << "%\n";
	ostr << measuredLevel << " is the highest visible MIP level\n";

	ostr << std::ends;

	RECT rc;
	rc.left = rc.top = 0;
	rc.bottom = 500;
	rc.right = 500;
	const D3DCOLOR black = D3DCOLOR_COLORVALUE(0,0,0,1);
	pFont->DrawText(NULL, ostr.str(), -1, &rc, DT_NOCLIP, black);
}

void setColourWrite(int queryNo)
{
	// In a real application, you would disable colour writes for the draw calls
	// that are used to test the MIP level.  For illustration purposes here, we
	// enable colour writes for one of the five calls.
	const bool show = (visualisedDIP == queryNo);
	const DWORD rgb = D3DCOLORWRITEENABLE_BLUE | D3DCOLORWRITEENABLE_GREEN | D3DCOLORWRITEENABLE_RED;

	if (show)
		pD3DDevice->SetRenderState(D3DRS_COLORWRITEENABLE, rgb);
	else
		pD3DDevice->SetRenderState(D3DRS_COLORWRITEENABLE, 0);
}

void issueMeasurementQueries()
{
	pD3DDevice->SetRenderState(D3DRS_ALPHATESTENABLE, true);
	pD3DDevice->SetPixelShader(pPixelShader);

	// Issue a draw call for each level that we wish to measure.
	for (int i=0; i!=REF_QUERY; ++i)
	{
		if (!isWaiting)
			pQueries[i]->Issue(D3DISSUE_BEGIN);

		setColourWrite(i);

		// This value gets subtracted from the texture's alpha value.
		float c[4];
		c[3] = 0.75f - static_cast<float>(i) / static_cast<float>(REF_QUERY);
		pD3DDevice->SetPixelShaderConstantF(0, c, 1);

		pD3DDevice->DrawIndexedPrimitive(D3DPT_TRIANGLESTRIP, 0, 0, nVtx, 0, nPrims);

		if (!isWaiting)
		{
			pQueries[i]->Issue(D3DISSUE_END);
			haveResult[i] = false;
		}
	}

	pD3DDevice->SetRenderState(D3DRS_ALPHATESTENABLE, false);

	isMeasured = false;
	isWaiting  = true;
}

void update()
{
	const DWORD what = D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER;
	const D3DCOLOR background = D3DCOLOR_COLORVALUE(0.5,0.9,1.0,1.0);
    pD3DDevice->Clear(0, NULL, what, background, 1.0f, 0);

	pD3DDevice->BeginScene();
	setUpRenderState();

	D3DXMATRIX world;
	composeWorldTransform(&world);
	pD3DDevice->SetTransform(D3DTS_WORLD, &world);

	pD3DDevice->SetStreamSource(0, pVB, 0, sizeof(Vertex));
	pD3DDevice->SetFVF(D3DFVF_MY_VERTEX);
	pD3DDevice->SetIndices(pIB);
	pD3DDevice->SetTexture(0, pTexture);
	pD3DDevice->SetPixelShader(NULL);

	setColourWrite(REF_QUERY);

	// Issue a reference query when we actually render the object.  This tells
	// us the total number of pixels that would normally get rendered.
	if (!isWaiting)
		pQueries[REF_QUERY]->Issue(D3DISSUE_BEGIN);

	pD3DDevice->DrawIndexedPrimitive(D3DPT_TRIANGLESTRIP, 0, 0, nVtx, 0, nPrims);

	if (!isWaiting)
	{
		pQueries[REF_QUERY]->Issue(D3DISSUE_END);
		haveResult[REF_QUERY] = false;
	}

	issueMeasurementQueries();

	checkQueryResults();
	reportResults();

	pD3DDevice->EndScene();
    pD3DDevice->Present(NULL, NULL, NULL, NULL);
}

int WINAPI WinMain(	HINSTANCE hInstance,
				   HINSTANCE hPrevInstance,
				   LPSTR     lpCmdLine,
				   int       nCmdShow)
{
	WNDCLASSEX winClass; 
	MSG        uMsg;

	memset(&uMsg,0,sizeof(uMsg));

	winClass.lpszClassName = "GPU_GEMS_II_SAMPLE_CLASS";
	winClass.cbSize        = sizeof(WNDCLASSEX);
	winClass.style         = CS_HREDRAW | CS_VREDRAW;
	winClass.lpfnWndProc   = WindowProc;
	winClass.hInstance     = hInstance;
	winClass.hIcon	       = NULL;
	winClass.hIconSm	   = NULL;
	winClass.hCursor       = LoadCursor(NULL, IDC_ARROW);
	winClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	winClass.lpszMenuName  = NULL;
	winClass.cbClsExtra    = 0;
	winClass.cbWndExtra    = 0;

	if (!RegisterClassEx(&winClass))
		return E_FAIL;

	hWnd = CreateWindowEx(NULL, "GPU_GEMS_II_SAMPLE_CLASS", "GPU Gems II: Mipmap Level Measurement",
		WS_OVERLAPPEDWINDOW | WS_VISIBLE, 0, 0, 640, 480, NULL, NULL, hInstance, NULL);

	if (hWnd == NULL)
		return E_FAIL;

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	init();

	while (uMsg.message != WM_QUIT)
	{
		if (PeekMessage(&uMsg, NULL, 0, 0, PM_REMOVE))
		{ 
			TranslateMessage(&uMsg);
			DispatchMessage(&uMsg);
		}
		else
			update();
	}

	shutDown();

	UnregisterClass("GPU_GEMS_II_SAMPLE_CLASS", winClass.hInstance);

	return uMsg.wParam;
}
