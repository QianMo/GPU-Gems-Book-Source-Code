#pragma once

#include "VSM.hpp"

//--------------------------------------------------------------------------------------
// Impelements variance shadow maps using hardware filtering
class Hardware : public VSM
{
public:
  Hardware(ID3D10Device* d3dDevice,
           ID3D10Effect* Effect,
           int Width, int Height,
           PostProcess* PostProcess,
           const DXGI_SAMPLE_DESC* SampleDesc);

  virtual bool EndShadowMap(ID3D10EffectTechnique* Technique);
  
  virtual ID3D10EffectTechnique* BeginShading();
  virtual void EndShading(ID3D10EffectTechnique* Technique);

protected:
  ID3D10EffectShaderResourceVariable* m_EffectShadowMap;     // Effect interface

private:
};