///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Shader.cpp
//  Desc : Shader object class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Shader.h"
#include "D3dApp.h"

// Create and assemble a vertex shader from a file
int CreateVertexShaderFromFile(const LPDIRECT3DDEVICE9 plD3DDevice,const char *pFileName, IDirect3DVertexShader9 *&pVshHandle)
{
  if(!pFileName)
  {
    return APP_ERR_INVALIDPARAM;
  }

  char pbFilename[255];
  sprintf(pbFilename,"%s%s",APP_DATAPATH_SHADERS, pFileName);

  // Assemble vertex shader
  LPD3DXBUFFER pCompiledShader=0, pCompilationError=0;
  if(FAILED(D3DXAssembleShaderFromFile(pbFilename, 0, 0, D3DXSHADER_DEBUG, &pCompiledShader, &pCompilationError)))
  {

#ifdef _DEBUG
    if(pCompilationError) 
    {
      OutputMsg("Error assemblying vertex shader", (char *)pCompilationError->GetBufferPointer());
    }
#endif

    SAFE_RELEASE(pCompilationError)
    return APP_ERR_READFAIL;
  }

  if(pCompilationError)
  {
    SAFE_RELEASE(pCompilationError)
    return APP_ERR_READFAIL;
  }

  // create vertex shader..
  if(FAILED(plD3DDevice->CreateVertexShader((ulong*)pCompiledShader->GetBufferPointer(), &pVshHandle)))
  {
    SAFE_RELEASE(pCompiledShader)
    return APP_ERR_INITFAIL;
  }

  SAFE_RELEASE(pCompiledShader)
  return APP_OK;
}

// Create and assemble a pixel shader from a file
int CreatePixelShaderFromFile(const LPDIRECT3DDEVICE9 plD3DDevice, const char *pFileName, IDirect3DPixelShader9 *&pPshHandle)
{
  if(!pFileName)
  {
    return APP_ERR_INVALIDPARAM;
  }

  char pbFilename[255];
  sprintf(pbFilename,"%s%s",APP_DATAPATH_SHADERS, pFileName);

  // assemble pixel shader
  LPD3DXBUFFER pCompiledShader=0, pCompilationError=0;
  if(FAILED(D3DXAssembleShaderFromFile(pbFilename, 0, 0, D3DXSHADER_DEBUG, &pCompiledShader, &pCompilationError)))
  {

#ifdef _DEBUG
    if(pCompilationError)
    {
      OutputMsg("Error assemblying pixel shader", (char *)pCompilationError->GetBufferPointer());
    }
#endif

    SAFE_RELEASE(pCompilationError)
    return APP_ERR_READFAIL;
  }

  if(pCompilationError)
  {
    SAFE_RELEASE(pCompilationError)
    return APP_ERR_READFAIL;
  }

  // create pixel shader
  if(FAILED(plD3DDevice->CreatePixelShader((ulong*)pCompiledShader->GetBufferPointer(), &pPshHandle)))
  {
    SAFE_RELEASE(pCompiledShader)
    return APP_ERR_INITFAIL;
  }

  SAFE_RELEASE(pCompiledShader)
  return APP_OK;
}

// Create and assemble a vertex shader 
int CreateVertexShader(const LPDIRECT3DDEVICE9 plD3DDevice, const char *pSource, IDirect3DVertexShader9 *&pVshHandle)
{  
  LPD3DXBUFFER pCompiledShader=0, pCompilationError=0;
  if(FAILED(D3DXAssembleShader(pSource, (int)strlen(pSource), 0, 0, D3DXSHADER_DEBUG,  &pCompiledShader, &pCompilationError)))
  {
#ifdef _DEBUG
    if(pCompilationError)
    {
      OutputMsg("Error assemblying vertex shader", (char *)pCompilationError->GetBufferPointer());
    }
#endif

    SAFE_RELEASE(pCompilationError)
    return APP_ERR_INITFAIL;
  }

  if(FAILED(plD3DDevice->CreateVertexShader((ulong*)pCompiledShader->GetBufferPointer(), &pVshHandle)))
  {
    SAFE_RELEASE(pCompiledShader)
    return APP_ERR_INITFAIL;
  }

  SAFE_RELEASE(pCompiledShader)
  return APP_OK;
}

// Create and assemble a pixel shader 
int CreatePixelShader(const LPDIRECT3DDEVICE9 plD3DDevice, const char *pSource, IDirect3DPixelShader9 *&pPshHandle)
{
  LPD3DXBUFFER pCompiledShader=0, pCompilationError=0;
  if(FAILED(D3DXAssembleShader(pSource, (int)strlen(pSource), 0, 0, D3DXSHADER_DEBUG,  &pCompiledShader, &pCompilationError)))
  {
#ifdef _DEBUG
    if(pCompilationError)
    {
      OutputMsg("Error assemblying pixel shader", (char *)pCompilationError->GetBufferPointer());
    }
#endif

    SAFE_RELEASE(pCompilationError)
    return APP_ERR_INITFAIL;
  }

  if(FAILED(plD3DDevice->CreatePixelShader((ulong*)pCompiledShader->GetBufferPointer(), &pPshHandle)))
  {
    SAFE_RELEASE(pCompiledShader)
    return APP_ERR_INITFAIL;
  }

  SAFE_RELEASE(pCompiledShader)
  return APP_OK;  
}

// Create and assemble cg program from file
int CreateCgProgram(const CGcontext pCgContext, const char *pFileName, CGprogram &pCgProgram, CGprofile pCgProfile)
{
  if(!pFileName)
  {
    return APP_ERR_INVALIDPARAM;
  }

  char pbFilename[256];
  sprintf(pbFilename,"%s%s",APP_DATAPATH_SHADERS, pFileName);

  // must declare this for dx9 vs1.1 compability
  const char* profileOpts[] = 
  {
    "-profileopts", "dcls", NULL,
  };

  pCgProgram=cgCreateProgramFromFile(pCgContext, CG_SOURCE, pbFilename, pCgProfile, 0, profileOpts);

  if(!cgIsProgram(pCgProgram)) 
  {     
    // listing in case of an error
    const char * pbListing = cgGetLastListing(pCgContext);
    if(!pbListing) 
    {
      pbListing = "Could not find CG compiler";
    }

    OutputMsg("Error loading cg program", pbListing);

    return APP_ERR_READFAIL;
  }

  return APP_OK;
}

int CShader:: CreateVertexShader(const LPDIRECT3DDEVICE9 plD3DDevice, CGcontext pCgContext, const char *pVP, D3DVERTEXELEMENT9 *pDeclBase, CGprofile pCgProfile)
{
  assert(plD3DDevice && "CShader:: Create - Invalid D3D device");
  assert(pVP && "CShader:: Create - Invalid shader path");
  assert(pDeclBase && "CShader:: Create - Invalid D3DVERTEXELEMENT9 pointer");

  // create vertex declaration
  if(FAILED(plD3DDevice->CreateVertexDeclaration(pDeclBase, &m_plD3DVertexDcl)))
  {
    return APP_ERR_INITFAIL;
  }

  // load vertex program
  if(APP_FAILED(CreateCgProgram(pCgContext, pVP, m_pCGVertexProgram, pCgProfile)))
  {
    return APP_ERR_READFAIL;
  }

  assert(cgD3D9ValidateVertexDeclaration(m_pCGVertexProgram, pDeclBase) && "CShader:: Create - Incompatible vertex declaration for base vs");

  // create vertex shader
  if(APP_FAILED(::CreateVertexShader(plD3DDevice, cgGetProgramString(m_pCGVertexProgram, CG_COMPILED_PROGRAM), m_plD3DVertexShader)))
  {
    OutputMsg("CShader:: Create ", "Loading vertex shader: %s", pVP);
    return APP_ERR_READFAIL;
  }

  return APP_OK;
}

int CShader:: CreatePixelShader(const LPDIRECT3DDEVICE9 plD3DDevice, CGcontext pCgContext, const char *pFP, CGprofile pCgProfile)
{
  assert(plD3DDevice && "CShader:: Create - Invalid D3D device");
  assert(pFP && "CShader:: Create - Invalid shader path");

  // load fragment program
  if(APP_FAILED(CreateCgProgram(pCgContext, pFP, m_pCGFragmentProgram, pCgProfile)))
  {
    return APP_ERR_READFAIL;
  }

  // create pixel shader
  if(APP_FAILED(::CreatePixelShader(plD3DDevice, cgGetProgramString(m_pCGFragmentProgram, CG_COMPILED_PROGRAM), m_plD3DPixelShader)))
  {
    OutputMsg("CShader:: Create", "Loading pixel shader: %s", pFP);
    return APP_ERR_READFAIL;
  }

  return APP_OK;
}

int CShader:: CreateVertexShader(const LPDIRECT3DDEVICE9 plD3DDevice, const char *pVP, D3DVERTEXELEMENT9 *pDeclBase)
{
  assert(plD3DDevice && "CShader:: Create - Invalid D3D device");
  assert(pVP && "CShader:: Create - Invalid shader path");
  assert(pDeclBase && "CShader:: Create - Invalid D3DVERTEXELEMENT9 pointer");

  // create vertex declaration
  if(FAILED(plD3DDevice->CreateVertexDeclaration(pDeclBase, &m_plD3DVertexDcl)))
  {
    return APP_ERR_INITFAIL;
  }

  return CreateVertexShaderFromFile(plD3DDevice, pVP, m_plD3DVertexShader);    
}

int CShader::CreatePixelShader(const LPDIRECT3DDEVICE9 plD3DDevice, char *pFP)
{
  assert(plD3DDevice && "CShader:: Create - Invalid D3D device");
  assert(pFP && "CShader:: Create - Invalid shader path");

  return CreatePixelShaderFromFile(plD3DDevice, pFP, m_plD3DPixelShader);
}

void CShader::SetShader() const
{
  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();

  plD3DDevice->SetVertexDeclaration(m_plD3DVertexDcl);
  plD3DDevice->SetVertexShader(m_plD3DVertexShader);
  plD3DDevice->SetPixelShader(m_plD3DPixelShader);
}

int CShader::SetVertexParam(const char *pParam, const float *pParamVal, int iSize) const
{
  if(!pParam || !pParamVal)
  {
    return APP_ERR_INVALIDPARAM;
  }

  CGparameter pVertexParam = cgGetNamedParameter(m_pCGVertexProgram, pParam);
  if(!pVertexParam)
  {
     return APP_ERR_INVALIDPARAM;
  }

  DWORD pVertexParamIndex=cgGetParameterResourceIndex(pVertexParam);    

  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();
  plD3DDevice->SetVertexShaderConstantF(pVertexParamIndex, pParamVal, iSize);

  return APP_OK;
}

int CShader::SetFragmentParam(const char *pParam, const float *pParamVal, int iSize) const
{
  if(!pParam || !pParamVal)
  {
    return APP_ERR_INVALIDPARAM;
  }

  CGparameter pFragmentParam = cgGetNamedParameter(m_pCGFragmentProgram, pParam);
  if(!pFragmentParam)
  {
    return APP_ERR_INVALIDPARAM;
  }

  DWORD pFragmentParamIndex=cgGetParameterResourceIndex(pFragmentParam);    

  const CD3DApp *pD3DApp=CD3DApp::GetD3DApp();
  LPDIRECT3DDEVICE9 plD3DDevice=pD3DApp->GetD3DDevice();
  plD3DDevice->SetPixelShaderConstantF(pFragmentParamIndex, pParamVal, iSize);
  return APP_OK;
}

int CShader::GetVertexParamIndex(const char *pParam)
{
  CGparameter pVParam= cgGetNamedParameter(m_pCGVertexProgram, pParam);
  return (int) cgGetParameterResourceIndex(pVParam);        
}

int CShader::GetFragmentParamIndex(const char *pParam)
{
  CGparameter pFParam= cgGetNamedParameter(m_pCGFragmentProgram, pParam);
  return (int) cgGetParameterResourceIndex(pFParam);        
}

void CShader:: Release()
{
  SAFE_RELEASE(m_plD3DVertexDcl)
  SAFE_RELEASE(m_plD3DVertexShader)
  SAFE_RELEASE(m_plD3DPixelShader)
  m_pCGFragmentProgram=0;
  m_pCGVertexProgram=0;
}