#include <brook/brook.hpp>
#include <stdio.h>
#include "rapcol.h"
#include "prapid.h"
extern void LoadPly (const char * file, std::vector<Tri> &ret);
extern unsigned int doCollide(unsigned int wida, unsigned int heia, Tri * a,
                              unsigned int bboxwida, unsigned int bboxheia, BBox * bboxesa,
                              unsigned int widb, unsigned int heib, Tri * b,
                              unsigned int bboxwidb, unsigned int bboxheib, BBox * bboxesb,
                              float3 rX, float3 rY, float3 rZ,
                              float3 trans,
                              float3 csrX, float3 csrY, float3 csrZ,
                              float3 cstrans,
                              float4 **intersections);
float3 convertVec(csVector3 v){
  return float3 (v.x,v.y,v.z);
}

int main (int argc, char ** argv) {
   float angle=90;
   int i;
   for (i=0;i<argc;++i) {
     char match=0;
     int j;
     if (strncmp(argv[i],"-angle",6)==0) {
       match=1;
       angle=(float)atof(argv[i]+6);
     }
     if (match) {
       for (j=i+1;j<argc;++j) argv[j-1]=argv[j];
       argc--;
       i--;
     }
   }
  
  float4 * intersections =0;
  angle*=(float)(3.1415926536/180);
  csMatrix3 R1(cos(angle),sin(angle),0,
                -sin(angle),cos(angle),0,
                0,0,1);
  csVector3 T1(0,0,0);
 
  srand(1);
  std::vector <bsp_polygon> model;
  std::vector<Tri> triangles;
  std::vector<BBox> bboxes;
  LoadPly (argc>1?argv[1]:"dragon.ply",model);
  csRapidCollider collide(model);
  collide.createBrookGeometry(bboxes,triangles);
  const csCdBBox *b1 = collide.GetBbox();
  const csCdBBox *b2 = collide.GetBbox();

  csMatrix3 tR1 = R1 * b1->m_Rotation;
  csVector3 tT1 = (R1 * b1->m_Translation) + T1;
  csMatrix3 tR2 = b2->m_Rotation;
  csVector3 tT2 = b2->m_Translation;

  csMatrix3 rot = tR1.GetTranspose () * tR2;
  csVector3 trans = tR1.GetTranspose () * (tT2 - tT1);
  csRapidCollider::mR = R1;
  csRapidCollider::mT = T1;




  collide.CollideRecursive(const_cast<csCdBBox*>( collide.GetBbox()),
                           const_cast<csCdBBox*>(collide.GetBbox()),
                           rot,
                           trans);

  fprintf (stderr,"Num Collisions %d Num BBoxes %d Num Triangles %d\n",
           csRapidCollider::numHits,
           bboxes.size(),
           triangles.size());
  unsigned int num = doCollide(triangles.size()/2048,2048,&triangles[0],
                               bboxes.size()/2048,2048,&bboxes[0],
                               triangles.size()/2048,2048,&triangles[0],
                               bboxes.size()/2048,2048,&bboxes[0],
                               convertVec(rot.Row1()),
                               convertVec(rot.Row2()),
                               convertVec(rot.Row3()),
                               convertVec(trans),
                               convertVec(csRapidCollider::mR.Row1()),
                               convertVec(csRapidCollider::mR.Row2()),
                               convertVec(csRapidCollider::mR.Row3()),
                               convertVec(csRapidCollider::mT),
                               &intersections);
  fprintf (stderr,"\nNum Collisions brook %d\n",num);
  if(0)for (unsigned int i=0;i<num;++i) {
     printf ("{%f %f %f %f}\n",intersections[i].x,
             intersections[i].y,
             intersections[i].z,
             intersections[i].w);
  }
  return 0;
}
