#ifndef __UTIL_H_included_
#define __UTIL_H_included_

#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif

#define ALERT_RETURN( X, MSG ) \
    if ( FAILED((X)) ) \
    { \
      MessageBox( NULL, _T(MSG), _T("Error"), MB_OK|MB_SETFOREGROUND|MB_TOPMOST ); \
      return; \
    }

#define ALERT(MSG) \
    MessageBox( NULL, _T(MSG), _T("Error"), MB_OK|MB_SETFOREGROUND|MB_TOPMOST ); 

#ifdef _DEBUG
#define DBG_ASSERT(x) assert(x)
#else
#define DBG_ASSERT(x)
#endif

#define DO_ONCE(X) \
    { \
        static bool first = true; \
        if (first) \
        { \
            first = false; \
            X \
        } \
    }

double factorial(unsigned int y);
bool ParaboloidCoord( D3DXVECTOR3* vec, int face, const D3DXVECTOR2* uv );
void CubeCoord( D3DXVECTOR3* vec, int face, const D3DXVECTOR2* uv );


#endif