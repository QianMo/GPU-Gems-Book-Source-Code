#pragma once

#include "VSM.hpp"
//--------------------------------------------------------------------------------------
class StandardSAT : public VSM
{
public:
  // The last parameters are really meant for derived classes.
  // Only change them if you know what you're doing.
  StandardSAT(ID3D10Device* d3dDevice,
              ID3D10Effect* Effect,
              int Width, int Height,
              PostProcess* PostProcess,
              const DXGI_SAMPLE_DESC* SampleDesc,
              bool IntFormat,
              bool DistributePrecision);

  ~StandardSAT();

  // Generate SATs here
  virtual bool EndShadowMap(ID3D10EffectTechnique* Technique);
  
  virtual ID3D10EffectTechnique* BeginShading();
  virtual void EndShading(ID3D10EffectTechnique* Technique);

  // Release SATs
  virtual void EndFrame();

protected:
  // Setup for SAT generation
  void InitGenerateSAT(int RDSamplesPerPass = 4);

  // Generate a single standard summed area table via recursive doubling
  RenderableTexture2D* GenerateSATRecursiveDouble(RenderableTexture2D *Src,
                                                  bool MaintainSrc = false);

  ID3D10EffectTechnique*              m_FPToINTTechnique;        // Technique for converting fp32 to int32
  ID3D10EffectShaderResourceVariable* m_EffectSATTexture;        // Effect interface to SAT texture
  RenderableTexture2D*                m_SAT;                     // Summed area table
  RenderableTexture2D*                m_MSAAResolvedShadowMap;   // Potentially need a texture to resolve into

private:
  bool                                m_IntFormat;               // Int of Float texture
  bool                                m_DistributePrecision;     // Distribute float precision

  // Recursive doubling
  ID3D10EffectTechnique*              m_GenerateSATRDTechnique;  // Technique to use for RD SAT generation
  ID3D10EffectVectorVariable*         m_EffectSATPassOffset;     // Pass offset in the effect file
  int                                 m_GenerateSATRDSamples;    // Recursive doubling samples/pass
};