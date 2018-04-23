//  builds a whole lot of texture data.
//    calculation of diffuse cubemap is done in two phases, given an input cubemap:
//    1)  compute N^2 spherical harmonic coefficients Llm
//        - for each coefficient, read 6K2 texels (or 2K'^2, dual paraboloid) from irradiance sample
//          cubemap.  read same number of texels from weight texture (precomputed to be 
//          (4PI*area(texel)/(totalarea(texels)*numtexels)) x SHlm(texel)
//          
//    2)  compute an M^2 cubemap using those coefficients
//        - for each texel, read N^2 coefficients and multiply by N^2 precomputed weights
//          weights are stored in N^2 textures, and calculated as:
//          BRDFcoefficient(l) * Ylm(texel)
//
//    total cost:
//       phase 1:  6K^2 * 2 lookups / pixel.  N^2 pixels
//       phase 2:  2N^2 lookups/pixel.  6M^2 pixels.
//
//  so, for K=32 (dual-paraboloid), N=3, M=32:
//       phase 1:  4K lookups/pixel * 9 pixels = 36K lookups / radiance sample
//       phase 2:  18 lookups/pixel * 6144 pixels = 110K lookups / radiance sample
//
//
//  references:
//    View-independent Environment Maps (Heidrich & Seidel, Eurographics 1998)
//    A Signal-Processing Framework for Inverse Rendering (Hanrahan & Ramamoorthi, SIGGRAPH 2001)
//    An Efficient Representation for Irradiance Environment Maps (Hanrahan & Ramamoorthi, SIGGRAPH 2001)
//    Precomputed Radiance Transfer for Real-Time Lighting in Dynamic, Low-Frequency Lighting Environments (Sloan, Katz, & Snyder, SIGGRAPH 2002)
//    Frequency-Space Environment Map Rendering (Hanrahan & Ramamoorthi, SIGGRAPH 2002)
//    Spherical Harmonic Lighting: The Gritty Details (Green, GDC 2003, http://www.research.scea.com/gdc2003/spherical-harmonic-lighting.pdf)
//    Interactive Global Illumination on the GPU (Nijasure, 2003, http://www.cs.ucf.edu/graphics/GPUassistedGI/)
//
#include "nvafx.h"

static BOOL IsPow2(DWORD i) { return ((i&(i-1))==0); }
static DWORD NextPow2(DWORD i) { DWORD d=0x1; while (d<i) d<<=1; return d; }

//  builds the textures required for phase 2, using an arbitrary BRDF.
//  individual bands are saved as subrects in a larger texture, so that only 6 textures are
//  generated.  6 textures are generated, 1 for each cube face.
//  resulting environment map is parameterized based on the normal
HRESULT BuildIrradianceTextures(LPDIRECT3DTEXTURE9 *weightTextures, LPDIRECT3DDEVICE9 lpDevice, DWORD dwOrder, DWORD dwSize, const SH_Reflectance_Al_Evaluator& AlEvaluator)
{
    if (!weightTextures || !IsPow2(dwSize))
        return E_FAIL;

    DWORD nSize = NextPow2( dwOrder );
    float* basisProj = new float[dwOrder*dwOrder];

    for (int face=0; face<6; face++)
    {
        //  allocate the texture, lock the surface
        //  use an R32F texture to store the coefficients
        if (FAILED(lpDevice->CreateTexture(dwSize*nSize, dwSize*nSize, 1, 0, D3DFMT_R32F, D3DPOOL_MANAGED, &weightTextures[face], NULL)))
            return E_FAIL;

        float* coefficients;
        coefficients = new float[dwSize*nSize * dwSize*nSize];
        ZeroMemory(coefficients, dwSize*dwSize*nSize*nSize*sizeof(float));

        for (unsigned int t=0; t<dwSize; t++)
        {
            for (unsigned int s=0; s<dwSize; s++)
            {
                D3DXVECTOR3 cubeVec;
                double sd=((s+0.5)/double(dwSize))*2. - 1.;
                double td=((t+0.5)/double(dwSize))*2. - 1.;
                D3DXVECTOR2 stVec ( (float)sd, (float)td );

                CubeCoord(&cubeVec, face, &stVec);

                //  compute the N^2 spherical harmonic basis functions
                float* basisProj = new float[dwOrder*dwOrder];
                D3DXSHEvalDirection( basisProj, dwOrder, &cubeVec );

                int basis=0;
                for (int l=0; l<(int)dwOrder; l++)
                {
                    double Al = AlEvaluator(l);
                    for (int m=-l; m<=l; m++, basis++)
                    {
                        int tiley = basis / dwOrder;
                        int tilex = basis % dwOrder;
                        double Ylm = basisProj[l*(l+1) + m];
                        
                        int offset = ((tiley*dwSize+t)*dwSize*nSize) + tilex*dwSize+s;
                        coefficients[offset] = (float)(Al * Ylm);
                    }
                }
            }
        }
        
        D3DLOCKED_RECT lockRect;
        if (FAILED(weightTextures[face]->LockRect(0, &lockRect, NULL, 0)))
            return E_FAIL;
        unsigned char* dst = (unsigned char*)lockRect.pBits;
        unsigned char* src = (unsigned char*)coefficients;
        unsigned int  srcPitch = dwSize * nSize * sizeof(float);

        for ( UINT i=0; i<dwSize*nSize; i++, dst+=lockRect.Pitch, src+=srcPitch )
            memcpy(dst, src, srcPitch);
        
        weightTextures[face]->UnlockRect(0);
        delete [] coefficients;
    }    
    
    delete [] basisProj;
    return S_OK;
}

HRESULT BuildLambertIrradianceTextures(LPDIRECT3DTEXTURE9 *weightTextures, LPDIRECT3DDEVICE9 lpDevice, DWORD dwOrder, DWORD dwSize)
{
    return BuildIrradianceTextures( weightTextures, lpDevice, dwOrder, dwSize, Lambert_Al_Evaluator() );
}

HRESULT BuildPhongIrradianceTextures(LPDIRECT3DTEXTURE9 *weightTextures, LPDIRECT3DDEVICE9 lpDevice, DWORD dwOrder, DWORD dwSize, FLOAT specular)
{
    return BuildIrradianceTextures( weightTextures, lpDevice, dwOrder, dwSize, Phong_Al_Evaluator(specular) );
}


//  These two functions compute weight coefficients for each texel in a rendered image.
//  iterate over the texels, summing weight*color, and the end result will be a spherical harmonic
//  coefficient.  to compute 16 coefficients, render to a 4x4 image (there will be subrects in the
//  weight texture corresponding to each coefficient)
HRESULT BuildDualParaboloidWeightTextures(LPDIRECT3DTEXTURE9 *weightTextures, LPDIRECT3DDEVICE9 lpDevice, DWORD dwOrder, DWORD dwSize)
{
    if (!weightTextures || !IsPow2(dwSize))
        return E_FAIL;

    DWORD nSize = NextPow2( dwOrder );

    //  texels need to be weighted by solid angle
    //  compute differential solid angle at texel center (cos(theta)/r^2), and then normalize and scale by 4*PI
    double *d_omega;
    double sum_d_omega = 0.f;
    d_omega = new double[dwSize*dwSize];

    unsigned int s, t;
    //  paraboloids are symmetrical, so compute total diff. solid angle for one half, and double it.
    for (t=0; t<dwSize; t++)
    {
        for (s=0; s<dwSize; s++)
        {
            double x=((s+0.5)/double(dwSize))*2. - 1.;
            double y=((t+0.5)/double(dwSize))*2. - 1.;
            double r_sqr = x*x + y*y;

            int index = t*dwSize + s;
            if ( r_sqr > 1. )  // only count points inside the circle
            {
                d_omega[index] = 0.;
                continue;
            }

            double z = 0.5*(1. - r_sqr);
            double mag = sqrt(r_sqr + z*z);  // =0.5[1+(x*x + y*y)]

            double cosTheta = 1.;  // cos(theta) terms fall out, since dA is first projected 
                                   // orthographically onto the paraboloid ( dA' = dA / dot(zAxis, Np) ), then reflected
                                   // and projected onto the unit sphere (dA'' = dA' dot(R,Np) / len^2)
                                   // dot(zAxis, Np) == dot(R, Np), so dA'' = dA / len^2
            d_omega[index] = cosTheta / (mag*mag);  //  = 1 / (mag^2)
            sum_d_omega += d_omega[index];
        }
    }

    double d_omega_scale = 4.*M_PI / (2.f*sum_d_omega);

    float* basisProj = new float[dwOrder*dwOrder];

    for (int face=0; face<2; face++)
    {
        //  allocate the texture, lock the surface
        //  use an R32F texture to store the coefficients
        if (FAILED(lpDevice->CreateTexture(dwSize*nSize, dwSize*nSize, 1, 0, D3DFMT_R32F, D3DPOOL_MANAGED, &weightTextures[face], NULL)))
            return E_FAIL;

        float *coefficients;
        coefficients = new float[dwSize*nSize * dwSize*nSize];
        ZeroMemory(coefficients, dwSize*dwSize*nSize*nSize*sizeof(float));

        for (t=0; t<dwSize; t++)
        {
            for (s=0; s<dwSize; s++)
            {
                D3DXVECTOR3 parabVec;
                double sd=((s+0.5)/double(dwSize))*2. - 1.;
                double td=((t+0.5)/double(dwSize))*2. - 1.;
                D3DXVECTOR2 stVec ( (float)sd, (float)td );

                if (!ParaboloidCoord(&parabVec, face, &stVec))
                    continue;   //  skip if this texel is outside the paraboloid


                //  compute the N^2 spherical harmonic basis functions
                D3DXSHEvalDirection(basisProj, dwOrder, &parabVec);

                int basis=0;
                int index = t*dwSize + s;
                for (int l=0; l<(int)dwOrder; l++)
                {
                    for (int m=-l; m<=l; m++, basis++)
                    {
                        int tiley = basis / dwOrder;
                        int tilex = basis % dwOrder;
                        double Ylm = basisProj[l*(l+1) + m];
                        
                        int offset = ((tiley*dwSize+t)*dwSize*nSize) + tilex*dwSize+s;
                        float weight = (float)(Ylm * d_omega[index] * d_omega_scale);
                        coefficients[offset] = weight;
                    }
                }
            }
        }

        D3DLOCKED_RECT lockRect;
        if (FAILED(weightTextures[face]->LockRect(0, &lockRect, NULL, 0)))
            return E_FAIL;
        unsigned char* dst = (unsigned char*)lockRect.pBits;
        unsigned char* src = (unsigned char*)coefficients;
        unsigned int  srcPitch = dwSize * nSize * sizeof(float);

        for ( UINT i=0; i<dwSize*nSize; i++, dst+=lockRect.Pitch, src+=srcPitch )
            memcpy(dst, src, srcPitch);

        weightTextures[face]->UnlockRect(0);
        delete [] coefficients;
    }

    delete [] basisProj;
    delete [] d_omega;
    return S_OK;
}

HRESULT BuildCubemapWeightTextures(LPDIRECT3DTEXTURE9 *weightTextures, LPDIRECT3DDEVICE9 lpDevice, DWORD dwOrder, DWORD dwSize)
{
    if (!weightTextures || !IsPow2(dwSize))
        return E_FAIL;

    DWORD nSize = NextPow2( dwOrder );

    // compute the total differential solid angle for samples over the dwSize*dwSize cube.  this is
    // used as a normalization scale factor for the individual solid angles...
    double *d_omega;
    double sum_d_omega = 0.f; 
    d_omega = new double[dwSize*dwSize];
    float* basisProj = new float[dwOrder*dwOrder];

    unsigned int s, t;

    //  faces are symmetrical, so just compute d_omega for one face, and replicate 6 times.
    for (t=0; t<dwSize; t++)
    {
        for (s=0; s<dwSize; s++)
        {
            D3DXVECTOR3 cubeVec;
            double sd=((s+0.5)/double(dwSize))*2. - 1.;
            double td=((t+0.5)/double(dwSize))*2. - 1.;
            D3DXVECTOR2 stVec ( (float)sd, (float)td );
            int index = t*dwSize + s;
            double r_sqr = sd*sd+td*td+1;        // distance origin to texel
            double cos_theta = 1./sqrt(r_sqr);   // dot product between cube vector (sphere direction)
                                                 // and surface normal (axis direction)
            d_omega[index] = cos_theta / r_sqr;  // =(r^-3/2)
            sum_d_omega += d_omega[index];
        }
    }

    double d_omega_scale = 4.*M_PI / (6.f*sum_d_omega);

    for (int face=0; face<6; face++)
    {
        //  allocate the texture, lock the surface
        //  use an R32F texture to store the coefficients
        if (FAILED(lpDevice->CreateTexture(dwSize*nSize, dwSize*nSize, 1, 0, D3DFMT_R32F, D3DPOOL_MANAGED, &weightTextures[face], NULL)))
            return E_FAIL;

        float* coefficients;
        coefficients = new float[dwSize*nSize * dwSize*nSize];
        ZeroMemory(coefficients, dwSize*dwSize*nSize*nSize*sizeof(float));

        for (t=0; t<dwSize; t++)
        {
            for (s=0; s<dwSize; s++)
            {
                D3DXVECTOR3 cubeVec;
                double sd=((s+0.5)/double(dwSize))*2. - 1.;
                double td=((t+0.5)/double(dwSize))*2. - 1.;
                D3DXVECTOR2 stVec ( (float)sd, (float)td );

                CubeCoord(&cubeVec, face, &stVec);

                int index = t*dwSize + s;

                //  compute the N^2 spherical harmonic basis functions
                D3DXSHEvalDirection( basisProj, dwOrder, &cubeVec );

                int basis=0;
                for (int l=0; l<(int)dwOrder; l++)
                {
                    for (int m=-l; m<=l; m++, basis++)
                    {
                        int tiley = basis / dwOrder;
                        int tilex = basis % dwOrder;
                        double Ylm = basisProj[l*(l+1)+m];
                        
                        int offset = ((tiley*dwSize+t)*dwSize*nSize) + tilex*dwSize+s;
                        coefficients[offset] = (float)(Ylm * d_omega[index] * d_omega_scale);
                    }
                }
            }
        }

        D3DLOCKED_RECT lockRect;
        if (FAILED(weightTextures[face]->LockRect(0, &lockRect, NULL, 0)))
            return E_FAIL;
        unsigned char* dst = (unsigned char*)lockRect.pBits;
        unsigned char* src = (unsigned char*)coefficients;
        unsigned int  srcPitch = dwSize * nSize * sizeof(float);

        for ( UINT i=0; i<dwSize*nSize; i++, dst+=lockRect.Pitch, src+=srcPitch )
            memcpy(dst, src, srcPitch);

        weightTextures[face]->UnlockRect(0);
        delete [] coefficients;
    }
    
    delete [] basisProj;
    delete [] d_omega;

    return S_OK;
}