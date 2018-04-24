#include "DXUT.h"
#include "RenderableTexture2D.hpp"

//--------------------------------------------------------------------------------------
RenderableTexture2D::RenderableTexture2D(ID3D10Device* d3dDevice,
                                         unsigned int Width, unsigned int Height,
                                         unsigned int MipLevels, DXGI_FORMAT Format,
                                         const DXGI_SAMPLE_DESC* SampleDesc)
  : m_Width(Width), m_Height(Height), m_MipLevels(MipLevels), m_Format(Format)
  , m_Array(false), m_ArrayIndex(0)
  , m_Texture(0), m_RenderTarget(0), m_ShaderResource(0)
{
  HRESULT hr;

  // Setup multisampling
  DXGI_SAMPLE_DESC RealSampleDesc;
  if (SampleDesc) {
    RealSampleDesc = *SampleDesc;
  } else {
    RealSampleDesc.Count = 1;
    RealSampleDesc.Quality = 0;
  }
  bool Multisampling = (RealSampleDesc.Count > 1 || RealSampleDesc.Quality > 0);  

  // Create texture
  D3D10_TEXTURE2D_DESC TexDesc;
  TexDesc.Width              = m_Width;
  TexDesc.Height             = m_Height;
  TexDesc.MipLevels          = m_MipLevels;
  TexDesc.ArraySize          = 1;
  TexDesc.Format             = m_Format;
  TexDesc.SampleDesc         = RealSampleDesc;
  TexDesc.Usage              = D3D10_USAGE_DEFAULT;
  TexDesc.BindFlags          = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
  TexDesc.CPUAccessFlags     = 0;
  // If they request mipmap levels, it's nice to be able to autogenerate them.
  TexDesc.MiscFlags          = (m_MipLevels == 0 ? D3D10_RESOURCE_MISC_GENERATE_MIPS : 0);

  V(d3dDevice->CreateTexture2D(&TexDesc, NULL, &m_Texture));
  // Update the description with the read number of mipmaps, etc.
  m_Texture->GetDesc(&TexDesc);

  // Create the render target view
  D3D10_RENDER_TARGET_VIEW_DESC RTDesc;
  RTDesc.Format              = TexDesc.Format;
  RTDesc.ViewDimension       = Multisampling ? D3D10_RTV_DIMENSION_TEXTURE2DMS :
                                               D3D10_RTV_DIMENSION_TEXTURE2D;
  RTDesc.Texture2D.MipSlice  = 0;

  V(d3dDevice->CreateRenderTargetView(m_Texture, &RTDesc, &m_RenderTarget));

  // Create the shader-resource view
  D3D10_SHADER_RESOURCE_VIEW_DESC SRDesc;
  SRDesc.Format                    = TexDesc.Format;
  SRDesc.ViewDimension             = Multisampling ? D3D10_SRV_DIMENSION_TEXTURE2DMS :
                                                     D3D10_SRV_DIMENSION_TEXTURE2D;
  SRDesc.Texture2D.MostDetailedMip = 0;
  SRDesc.Texture2D.MipLevels       = TexDesc.MipLevels;

  V(d3dDevice->CreateShaderResourceView(m_Texture, &SRDesc, &m_ShaderResource));
}

//--------------------------------------------------------------------------------------
RenderableTexture2D::RenderableTexture2D(ID3D10Device* d3dDevice,
                                         ID3D10Texture2D* TextureArray,
                                         unsigned int Index)
  : m_Array(true), m_ArrayIndex(Index)
  , m_Texture(TextureArray), m_RenderTarget(0), m_ShaderResource(0)
{
  m_Texture->AddRef();

  // Fill in information about the source texture
  D3D10_TEXTURE2D_DESC TexDesc;
  m_Texture->GetDesc(&TexDesc);
  m_Width     = TexDesc.Width;
  m_Height    = TexDesc.Height;
  m_MipLevels = TexDesc.MipLevels;
  m_Format    = TexDesc.Format;

  HRESULT hr;

  // Create the render target view
  D3D10_RENDER_TARGET_VIEW_DESC RTDesc;
  RTDesc.Format                          = TexDesc.Format;
  RTDesc.ViewDimension                   = D3D10_RTV_DIMENSION_TEXTURE2DARRAY;
  RTDesc.Texture2DArray.MipSlice         = 0;
  RTDesc.Texture2DArray.FirstArraySlice  = m_ArrayIndex;
  RTDesc.Texture2DArray.ArraySize        = 1;

  V(d3dDevice->CreateRenderTargetView(m_Texture, &RTDesc, &m_RenderTarget));

  // Create the shader-resource view
  D3D10_SHADER_RESOURCE_VIEW_DESC SRDesc;
  SRDesc.Format                          = TexDesc.Format;
  SRDesc.ViewDimension                   = D3D10_SRV_DIMENSION_TEXTURE2DARRAY;
  SRDesc.Texture2DArray.MostDetailedMip  = 0;
  SRDesc.Texture2DArray.MipLevels        = TexDesc.MipLevels;
  SRDesc.Texture2DArray.FirstArraySlice  = m_ArrayIndex;
  SRDesc.Texture2DArray.ArraySize        = 1;

  V(d3dDevice->CreateShaderResourceView(m_Texture, &SRDesc, &m_ShaderResource));
}

//--------------------------------------------------------------------------------------
RenderableTexture2D::~RenderableTexture2D()
{
  SAFE_RELEASE(m_ShaderResource);
  SAFE_RELEASE(m_RenderTarget);
  SAFE_RELEASE(m_Texture);
}
