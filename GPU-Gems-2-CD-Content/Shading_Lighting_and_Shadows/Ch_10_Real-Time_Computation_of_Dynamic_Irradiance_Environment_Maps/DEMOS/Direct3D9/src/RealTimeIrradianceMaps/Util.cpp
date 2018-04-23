#include "nvafx.h"

double factorial(unsigned int y)
{
    const double table[16] = { 1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880., 
        3628800., 39916800., 479001600., 6227020800., 87178291200., 1307674368000. };
   
    double result = table[(y>15)?15:y];

    while (y>=16)
    {
        result = result * double(y);
        y--;
    }

    return result;
}

void CubeCoord( D3DXVECTOR3* vec, int face, const D3DXVECTOR2* uv )
{
    D3DXVECTOR3 tmp;
    switch (face)
    {
    case 0: tmp = D3DXVECTOR3(   1.f, -uv->y, -uv->x); break;
    case 1: tmp = D3DXVECTOR3(  -1.f, -uv->y,  uv->x); break;
    case 2: tmp = D3DXVECTOR3( uv->x,    1.f,  uv->y); break;
    case 3: tmp = D3DXVECTOR3( uv->x,   -1.f, -uv->y); break;
    case 4: tmp = D3DXVECTOR3( uv->x, -uv->y,    1.f); break;
    case 5: tmp = D3DXVECTOR3(-uv->x, -uv->y,   -1.f); break;
    }
    D3DXVec3Normalize(&tmp, &tmp);
    vec->x =  tmp.z;
    vec->y = -tmp.x;
    vec->z =  tmp.y;
    //*vec = tmp;
}

bool ParaboloidCoord( D3DXVECTOR3* vec, int face, const D3DXVECTOR2* uv )
{
    //  sphere direction is the reflection of the orthographic view direction (determined by
    //  face), reflected about the normal to the paraboloid at uv
    double r_sqr = uv->x*uv->x + uv->y*uv->y;
    
    if (r_sqr > 1.)
        return false;

    D3DXVECTOR3 axis;
    if (face==0)
        axis = D3DXVECTOR3(0.f, 0.f, -1.f);
    else
        axis = D3DXVECTOR3(0.f, 0.f, 1.f);

    // compute normal on the parabaloid at uv
    D3DXVECTOR3 N( uv->x, uv->y, 1.f );
    D3DXVec3Normalize(&N, &N);

    // reflect axis around N, to compute sphere direction
    float v_dot_n = D3DXVec3Dot(&axis, &N);
    D3DXVECTOR3 R = axis - 2*v_dot_n*N;
    
    *vec = R;
    return true;
}