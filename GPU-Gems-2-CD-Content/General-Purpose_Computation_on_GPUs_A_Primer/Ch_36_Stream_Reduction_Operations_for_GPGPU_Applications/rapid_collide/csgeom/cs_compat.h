#ifndef CS_COMPAT_H_
#define CS_COMPAT_H_
#define SCF_DECLARE_IBASE

#define SMALL_EPSILON .000001
#define EPSILON .00001
#define ABS(x) (x>=0?x:-x)
#define __CS_CSSYSDEFS_H__
#include "csgeom/vector3.h"
#include "csgeom/matrix3.h"
typedef unsigned char uint8;
#include "csgeom/box.h"


#include <stdlib.h>
#include <string.h>
#include <assert.h>

namespace csMath3 {
  inline void SetMinMax (const csVector3& v,
			 csVector3& min, csVector3& max)
    {
      if (v.x > max.x) max.x = v.x; else if (v.x < min.x ) min.x = v.x;
      if (v.y > max.y) max.y = v.y; else if (v.y < min.y ) min.y = v.y;
      if (v.z > max.z) max.z = v.z; else if (v.z < min.z ) min.z = v.z;
    }
}
namespace csSquaredDist {
  inline float PointPoint (const csVector3 &a, const csVector3 &b) {
    return (a-b).SquaredNorm();
  }

}
/*blah
class csReversibleTransform {
  Matrix mat;
 public:
  csReversibleTransform () {
  }
  csReversibleTransform (Matrix m) {
    CopyMatrix (mat,m);
  }
  csReversibleTransform (const Transformation &t) {
    t.to_matrix (mat);
  }
  
  csReversibleTransform GetInverse () const{
    csReversibleTransform mytrans;
    InvertMatrix (mytrans.mat,mat);
    return mytrans;
  }
  
  csReversibleTransform operator * (const csReversibleTransform &b) const {
    Matrix a;
    MultMatrix (a,b.mat,mat);
    return csReversibleTransform (a);
  }
  csReversibleTransform operator *= (const csReversibleTransform &b) {
    return (*this = ((*this) * b));
  }
  csVector3 GetO2TTranslation () const {return csVector3(mat[12],mat[13],mat[14]);}
  csVector3 GetOrigin () const {return GetO2TTranslation();}
  void SetOrigin (const csVector3 &v) {mat[12]=v.x; mat[13]=v.y; mat[14]=v.z;}
  void SetO2TTranslation (const csVector3 &v) {SetOrigin (v);}
  csMatrix3 GetO2T () const {
    return csMatrix3 (mat[0],mat[4],mat[8],
		      mat[1],mat[5],mat[9],
		      mat[2],mat[6],mat[10]);
  }

};
*/
class csObject {
};
struct iBase {
};
#define SCF_VERSION(a,b,c,d)

#ifdef CS_DECLARE_GROWING_ARRAY_REF
#undef CS_DECLARE_GROWING_ARRAY_REF
#endif
#define CS_DECLARE_GROWING_ARRAY_REF(a,b) std::vector<b> a

#define SCF_IMPLEMENT_IBASE(csblah)
#define SCF_IMPLEMENTS_INTERFACE(collideblah)
#define SCF_IMPLEMENT_IBASE_END
#define SCF_CONSTRUCT_IBASE(blah);

#define SCF_DECLARE_IBASE_EXT(csObject)

#define CS_ASSERT assert
#endif
