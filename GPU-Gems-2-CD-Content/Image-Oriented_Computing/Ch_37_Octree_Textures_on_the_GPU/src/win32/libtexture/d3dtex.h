#ifndef __D3DTEX__
#define __D3DTEX__

#include <windows.h>
#include <D3D9.h>
#include <D3DX9.h>
#include "DXUtil.h"

#include "CTexture.h" 

#define CHECK_ERROR(x) if ((x) != D3D_OK) MessageBox(NULL,#x,"Error",MB_OK | MB_ICONSTOP);

namespace d3dtex
{
  
  inline LPDIRECT3DTEXTURE9 d3dtexFromTexture(LPDIRECT3DDEVICE9 d3dDevice,CTexture *tex,bool mipmap=false)
  {
    D3DLOCKED_RECT      rect0;
    LPDIRECT3DTEXTURE9  dtex;
    
    if( FAILED( d3dDevice->CreateTexture( tex->getWidth(), tex->getHeight(), mipmap?0:1, 0,
      D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &dtex, NULL ) ) )
    {
      throw CLibTextureException("Unable to create D3D texture !");
    }
    
    dtex->LockRect( 0, &rect0, 0, 0 );
    CHAR *pDst = (CHAR*)rect0.pBits;
    CHAR *pSrc = (CHAR *)tex->getData();
    int t=tex->isAlpha()?4:3;
    for (UINT i=0;i<tex->getWidth();i++)
    {
      for (UINT j=0;j<tex->getHeight();j++)
      {
        pDst[i*4+j*rect0.Pitch+2]=pSrc[(i+j*tex->getWidth())*t  ];
        pDst[i*4+j*rect0.Pitch+1]=pSrc[(i+j*tex->getWidth())*t+1];
        pDst[i*4+j*rect0.Pitch  ]=pSrc[(i+j*tex->getWidth())*t+2];
        if (t == 4)
          pDst[i*4+j*rect0.Pitch+3]=pSrc[(i+j*tex->getWidth())*t+3];
        else
          pDst[i*4+j*rect0.Pitch+3]=(char)255;
      }
    }
    if (mipmap)
    {
      D3DLOCKED_RECT      rectn_1=rect0;
      D3DLOCKED_RECT      rectn;
      int nb=dtex->GetLevelCount();
      for (int l=1;l<nb;l++)
      {
        dtex->LockRect( l, &rectn, 0, 0 );
        D3DSURFACE_DESC desc;
        dtex->GetLevelDesc(l-1,&desc);
        int sizei=desc.Width;
        int sizej=desc.Height;
        dtex->GetLevelDesc(l,&desc);
        unsigned char *pDst = (unsigned char*)rectn.pBits;
        unsigned char *pSrc = (unsigned char*)rectn_1.pBits;
        for (UINT i=0;i<sizei/2;i++)
        {
          for (UINT j=0;j<sizej/2;j++)
          {
            int r00=pSrc[(i*2)*4+(j*2)*rectn_1.Pitch  ];
            int g00=pSrc[(i*2)*4+(j*2)*rectn_1.Pitch+1];
            int b00=pSrc[(i*2)*4+(j*2)*rectn_1.Pitch+2];
            int a00=pSrc[(i*2)*4+(j*2)*rectn_1.Pitch+3];
            int r01=pSrc[(i*2)*4+(j*2+1)*rectn_1.Pitch  ];
            int g01=pSrc[(i*2)*4+(j*2+1)*rectn_1.Pitch+1];
            int b01=pSrc[(i*2)*4+(j*2+1)*rectn_1.Pitch+2];
            int a01=pSrc[(i*2)*4+(j*2+1)*rectn_1.Pitch+3];
            int r10=pSrc[(i*2+1)*4+(j*2)*rectn_1.Pitch  ];
            int g10=pSrc[(i*2+1)*4+(j*2)*rectn_1.Pitch+1];
            int b10=pSrc[(i*2+1)*4+(j*2)*rectn_1.Pitch+2];
            int a10=pSrc[(i*2+1)*4+(j*2)*rectn_1.Pitch+3];
            int r11=pSrc[(i*2+1)*4+(j*2+1)*rectn_1.Pitch  ];
            int g11=pSrc[(i*2+1)*4+(j*2+1)*rectn_1.Pitch+1];
            int b11=pSrc[(i*2+1)*4+(j*2+1)*rectn_1.Pitch+2];
            int a11=pSrc[(i*2+1)*4+(j*2+1)*rectn_1.Pitch+3];
            
            int r=(r00+r01+r10+r11)/4;
            int g=(g00+g01+g10+g11)/4;
            int b=(b00+b01+b10+b11)/4;
            int a=(a00+a01+a10+a11)/4;
            
            pDst[i*4+j*rectn.Pitch  ]=r;
            pDst[i*4+j*rectn.Pitch+1]=g;
            pDst[i*4+j*rectn.Pitch+2]=b;
            pDst[i*4+j*rectn.Pitch+3]=a;
          }
        }
        dtex->UnlockRect(l-1);
        rectn_1=rectn;
      }
    }
    else
      dtex->UnlockRect(0);
    
    return (dtex);
  }
  
  inline LPDIRECT3DTEXTURE9 d3dtexLoadTexture(LPDIRECT3DDEVICE9 d3dDevice,const char *n,bool mipmap=false)
  {
    CTexture *tex=CTexture::loadTexture(n);
    LPDIRECT3DTEXTURE9 dtex=d3dtexFromTexture(d3dDevice,tex,mipmap);
    delete (tex);
    return (dtex);
  }
  
  inline CTexture *d3dtexTexFromScreen(LPDIRECT3DDEVICE9 d3dDevice)
  {
    LPDIRECT3DSURFACE9  surf;
    LPDIRECT3DTEXTURE9  tmptex;
    LPDIRECT3DSURFACE9  tmpsurf;
    D3DLOCKED_RECT      rect;
    CTexture           *tex;
    D3DSURFACE_DESC     desc;
    
    // read screen
    d3dDevice->GetRenderTarget(0,&surf);
    surf->GetDesc(&desc);
    CHECK_ERROR( d3dDevice->CreateTexture(desc.Width, desc.Height , 1, 0,
      D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &tmptex, NULL ));
    tmptex->GetSurfaceLevel(0,&tmpsurf);
    CHECK_ERROR(d3dDevice->GetRenderTargetData(surf,tmpsurf));
    CHECK_ERROR(tmpsurf->LockRect(&rect,0,0));
    // copy data
    unsigned char *data=new unsigned char[desc.Width*desc.Height*3];
    unsigned char *ptr=(unsigned char *)rect.pBits;
    for (int i=0;i<desc.Width;i++)
    {
      for (int j=0;j<desc.Height;j++)
      {
        data[(i+(desc.Height-j-1)*desc.Width)*3  ]=ptr[i*4+j*rect.Pitch+2];
        data[(i+(desc.Height-j-1)*desc.Width)*3+1]=ptr[i*4+j*rect.Pitch+1];
        data[(i+(desc.Height-j-1)*desc.Width)*3+2]=ptr[i*4+j*rect.Pitch  ];
      }
    }
    // release all
    CHECK_ERROR(tmpsurf->UnlockRect());	
    SAFE_RELEASE(tmpsurf);
    SAFE_RELEASE(tmptex);
    SAFE_RELEASE(surf);
    // create and return texture
    tex=new CTexture("[screen]",desc.Width,desc.Height,false,data);
    tex->setDataOwner(true);
    return (tex);
  }
  
}

#endif
