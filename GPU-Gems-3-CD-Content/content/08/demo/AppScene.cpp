#include "DXUT.h"
#include "App.hpp"
#include "CarScene.hpp"
#include "SpheresScene.hpp"
#include "CommandoScene.hpp"
#include "LargeScene.hpp"

//--------------------------------------------------------------------------------------
void App::InitScene(ID3D10Device* d3dDevice)
{
  SAFE_DELETE(m_Scene);

  // Create the new scene
  switch (m_SceneIndex) {
    case S_CAR:
      {
        m_Scene = new CarScene(d3dDevice);
        break;
      }
    case S_SPHERES:
      {
        m_Scene = new SpheresScene(d3dDevice);
        break;
      }
    case S_COMMANDO:
      {
        m_Scene = new CommandoScene(d3dDevice);
        break;
      }
    case S_LARGE:
      {
        m_Scene = new LargeScene(d3dDevice);
        break;
      }
    default:
      throw std::runtime_error("Unknown scene!");
  };

  // Use current effect if created
  if (m_Effect) {
    m_Scene->SetEffect(d3dDevice, m_Effect);
  }

  // Set light and camera position to scene default
  D3DXVECTOR3 Pos, Target;
  m_Scene->GetCameraDefaults(Pos, Target);
  m_ViewCamera.SetViewParams(&Pos, &Target);
  m_ViewCamera.FrameMove(0.0f);                 // Update internals

  m_Scene->GetLightDefaults(Pos, Target);
  m_LightCamera.SetViewParams(&Pos, &Target);
  m_LightCamera.FrameMove(0.0f);                // Update internals

  // Update our light parameters since they may have been scene-related
  UpdateLightParameters();
}

//--------------------------------------------------------------------------------------
void App::RenderScene(ID3D10Device* d3dDevice,
                      ID3D10EffectTechnique* RenderTechnique,
                      const D3DXMATRIXA16& View,
                      const D3DXMATRIXA16& Proj)
{
  // No global world transform (scene centered at 0,0,0)
  D3DXMATRIXA16 World;
  D3DXMatrixIdentity(&World);

  // Render the scene
  assert(RenderTechnique && RenderTechnique->IsValid());
  m_Scene->Render(d3dDevice, RenderTechnique, World, View, Proj);
}
