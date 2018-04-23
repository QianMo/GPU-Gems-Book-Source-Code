///////////////////////////////////////////////////////////////////////////////////////////////////
//  Proj : GPU GEMS 2 DEMOS
//  File : Shader.h
//  Desc : Quick&Dirty shader object class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

class CShader
{
public:  
  CShader():m_plD3DVertexDcl(0), m_plD3DVertexShader(0), m_plD3DPixelShader(0)
  {
  };

  ~CShader()
  {
    Release();
  };
  
  // Create vertex shader from file
  int CreateVertexShader(const LPDIRECT3DDEVICE9 plD3DDevice, CGcontext pCgContext, const char *pVP, D3DVERTEXELEMENT9 *pDeclBase, CGprofile pCgProfile);
  // Create vertex shader from file
  int CreateVertexShader(const LPDIRECT3DDEVICE9 plD3DDevice, const char *pVP, D3DVERTEXELEMENT9 *pDeclBase);
  // Create pixel shader from file
  int CreatePixelShader(const LPDIRECT3DDEVICE9 plD3DDevice, CGcontext pCgContext, const char *pFP, CGprofile pCgProfile);  
  // Create pixel shader from file
  int CreatePixelShader(const LPDIRECT3DDEVICE9 plD3DDevice, char *pFP);
  // Set shader object
  void SetShader() const;
  // Set shader parameter
  int SetVertexParam(const char *pParam, const float *pParamVal, int iSize) const;
  // Set shader parameter
  int SetFragmentParam(const char *pParam, const float *pParamVal, int iSize) const;
  // Return vertex parameter index
  int GetVertexParamIndex(const char *pParam);
  // Return fragment parameter index
  int GetFragmentParamIndex(const char *pParam);

  // Free resources
  void Release();

  // Get methods
  IDirect3DVertexDeclaration9 *GetVertexDeclaration()
  {
    return m_plD3DVertexDcl;
  };

  IDirect3DVertexShader9      *GetVertexShader()
  {
    return m_plD3DVertexShader;
  };

  CGprogram                   *GetCGVertexProgram()
  {
    return &m_pCGVertexProgram;
  };

  IDirect3DPixelShader9       *GetPixelShader()
  {
    return m_plD3DPixelShader;
  };

  CGprogram                   *GetCGFragmentProgram()
  {
    return &m_pCGFragmentProgram;
  };

private:
  IDirect3DVertexDeclaration9 *m_plD3DVertexDcl;     // vertex declaration
  IDirect3DVertexShader9      *m_plD3DVertexShader;  // vertex shader
  CGprogram                    m_pCGVertexProgram;   // vertex program

  IDirect3DPixelShader9       *m_plD3DPixelShader;   // pixel shader
  CGprogram                    m_pCGFragmentProgram; // fragment program
};