/**
 @file shadowVolume.h
 
 @maintainer Morgan McGuire, morgan@graphics3d.com
 
 @created 2001-12-16
 @edited  2004-01-13
 */

#ifndef G3D_SHADOWVOLUME_H
#define G3D_SHADOWVOLUME_H

#include "graphics3D.h"
#include "GLG3D/PosedModel.h"

namespace G3D {

/** See G3D::markShadows */
void beginMarkShadows(RenderDevice* renderDevice);

/** See G3D::markShadows */
void endMarkShadows(RenderDevice* renderDevice);

/**
 Given a posed model, renders a shadow volume using the algorithm
 described in:
 
   C. Everitt, M. Kilgard, "Practical and Robust Shadow Volumes", 
   NVIDIA Technical Report, 2002
   http://developer.nvidia.com/object/robust_shadow_volumes.html

 This algorithm <B>requires</B> models with no broken edges.  Shadows
 will not be rendered for models with broken edges.

 Note that faster methods of rendering shadow volumes have since been
 proposed, such as: 

   M. McGuire, J. F. Hughes, K. T. Egan, M. Kilgard, C. Everitt,
   "Fast, Practical, and Robust Shadow Volumes", NVIDIA Technical
   Report 2003
   http://developer.nvidia.com/object/fast_shadow_volumes.html

 Set up your rendering code as follows:

  <PRE>
    renderDevice->clear(true, true, true);
    renderDevice->pushState();
        renderDevice->enableLighting()
        // Configure ambient light
        renderDevice->setAmbientLightLevel(Color3::WHITE * .25);

        for (int m = 0; m < model.size(); ++m) {
            model[m]->render(renderDevice);
        }

        renderDevice->disableDepthWrite();
        renderDevice->setBlendFunc(RenderDevice::BLEND_ONE, RenderDevice::BLEND_ONE);
        renderDevice->setStencilTest(RenderDevice::STENCIL_EQUAL);
        renderDevice->setAmbientLightLevel(Color3::BLACK);
    
          for (int L = 0; L < light.size(); ++L) {
            beginMarkShadows(renderDevice);
                for (int m = 0; m < model.size(); ++m) {
                    markShadows(renderDevice, model[m], light[L]);
                }
            endMarkShadows(renderDevice);

            renderDevice->setLight(0, light[L]);
            for (int m = 0; m < model.size(); ++m) {
                model[m]->render(renderDevice);
            }
        }
    renderDevice->popState();
  </PRE>

  <B>Limitations</B>

  This function currently works only for directional lights (light.w == 0).
  This function requires the far clipping plane at z = -inf (the
  default for GCamera).
 */
void markShadows(
    RenderDevice*           renderDevice, 
    const PosedModelRef&    model,
    const Vector4&          light);

}

#endif
