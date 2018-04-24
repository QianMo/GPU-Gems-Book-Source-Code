#pragma once

//--------------------------------------------------------------------------------------
// Simple class to encapsulate a Texture2D, a render target view and a shader resource
// view. Used to simplify the creation and management of the fairly common single
// render target/shader resource view per texture case.
class RenderableTexture2D
{
public:
  // Create a new 2D texture using the given information
  RenderableTexture2D(ID3D10Device* d3dDevice, unsigned int Width, unsigned int Height,
                      unsigned int MipLevels, DXGI_FORMAT Format,
                      const DXGI_SAMPLE_DESC* SampleDesc = 0);

  // Create a render target and shader resource for the associated slice of the given
  // 2D texture array.
  RenderableTexture2D(ID3D10Device* d3dDevice, ID3D10Texture2D* TextureArray,
                      unsigned int Index);

  ~RenderableTexture2D();

  unsigned int GetWidth() const { return m_Width; }
  unsigned int GetHeight() const { return m_Height; }
  unsigned int GetMipLevels() const { return m_MipLevels; }
  DXGI_FORMAT GetFormat() const { return m_Format; }
  bool IsArray() const { return m_Array; }
  unsigned int GetArrayIndex() const { return m_ArrayIndex; }

  ID3D10Texture2D * GetTexture() { return m_Texture; }
  ID3D10RenderTargetView * GetRenderTarget() { return m_RenderTarget; }
  ID3D10ShaderResourceView * GetShaderResource() { return m_ShaderResource; }

private:
  // Not implemented
  RenderableTexture2D(const RenderableTexture2D &);

  unsigned int                        m_Width;
  unsigned int                        m_Height;
  unsigned int                        m_MipLevels;
  DXGI_FORMAT                         m_Format;
  bool                                m_Array;
  unsigned int                        m_ArrayIndex;

  ID3D10Texture2D*                    m_Texture;
  ID3D10RenderTargetView*             m_RenderTarget;
  ID3D10ShaderResourceView*           m_ShaderResource;
};