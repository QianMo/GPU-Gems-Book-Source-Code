#include <brook/brook.hpp>
#include <stdio.h>
float ABS (float x) {
  return x>=0?x:-x;
}
#include "csgeom/vector3.h"
#include "csgeom/matrix3.h"
#include "rapcol.h"
/*
typedef struct Tri_t {
  float3 A;
  float3 B;
  float3 C;
  } Tri;*/
extern int tri_contact (const csVector3 &,const csVector3 &,const csVector3 &,
                        const csVector3 &,const csVector3 &,const csVector3 &);

#define USE_EPSILON_TEST	1

//const   float EPSILON=.00001f;


float min3(float a, float b, float c) {
  return a>b?(b>c?c:b):a;
}
float max3(float a, float b, float c) {
  return a<b?(b<c?c:b):a;
}

extern int project6 (csVector3 ax, csVector3 p1, csVector3 p2, csVector3 p3,
                     csVector3 q1, csVector3 q2, csVector3 q3);

bool tri_contact_nodiv (csVector3 P1, csVector3 P2, csVector3 P3,
                        csVector3 Q1, csVector3 Q2, csVector3 Q3)
{
  //
  // One triangle is (p1,p2,p3).  Other is (q1,q2,q3).
  // Edges are (e1,e2,e3) and (f1,f2,f3).
  // Normals are n1 and m1
  // Outwards are (g1,g2,g3) and (h1,h2,h3).
  //
  // We assume that the triangle vertices are in the same coordinate system.
  //
  // First thing we do is establish a new c.s. so that p1 is at (0,0,0).
  //

  csVector3 p1 = P1 - P2;
  csVector3 p2 = P2 - P1;
  csVector3 p3 = P3 - P1;

  csVector3 q1 = Q1 - P1;
  csVector3 q2 = Q2 - P1;
  csVector3 q3 = Q3 - P1;

  csVector3 e1 = p2 - p1;
  csVector3 e2 = p3 - p2;
  csVector3 e3 = p1 - p3;

  csVector3 f1 = q2 - q1;
  csVector3 f2 = q3 - q2;
  csVector3 f3 = q1 - q3;

  csVector3 n1 = e1 % e2;
  csVector3 m1 = f1 % f2;

  csVector3 g1 = e1 % n1;
  csVector3 g2 = e2 % n1;
  csVector3 g3 = e3 % n1;
  csVector3 h1 = f1 % m1;
  csVector3 h2 = f2 % m1;
  csVector3 h3 = f3 % m1;

  csVector3 ef11 = e1 % f1;
  csVector3 ef12 = e1 % f2;
  csVector3 ef13 = e1 % f3;

  csVector3 ef21 = e2 % f1;
  csVector3 ef22 = e2 % f2;
  csVector3 ef23 = e2 % f3;

  csVector3 ef31 = e3 % f1;
  csVector3 ef32 = e3 % f2;
  csVector3 ef33 = e3 % f3;

  // now begin the series of tests

  if (!project6(n1, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(m1, p1, p2, p3, q1, q2, q3)) return false;

  if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;

  if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

  return true;
}



extern void doTest(float * matrix,
                   int sizex,int sizey,
                   float3 * hitsa,
                   float3* hitsb,
                   float4 * ohits);

extern void doTestNoDiv(float * matrix,
                        int sizex,int sizey,
                        float3 * hitsa,
                        float3* hitsb,
                        float4 * ohits);
int obb_disjoint (const csMatrix3& B, const csVector3& T,
                  const csVector3& a, const csVector3& b);
extern float TestObbDisjoint(float3 X, float3 Y, float3 Z, float3 T,
                             float3 a, float3 b);
#define SIZEX 1
#define SIZEY 1
void SetTriangles(int size, float3* a, float3 *b) {
  float rm = RAND_MAX*.25;
  for (int i=0;i<size;++i) {
  a[i*3]=float3(rand()/rm,rand()/rm,rand()/rm);
  a[i*3+1]=float3(-1-rand()/rm,-1-rand()/rm,-1-rand()/rm);
  a[i*3+2]=float3(-1-rand()/rm,-1-rand()/rm,-1-rand()/rm);
  b[i*3]=float3(-1-rand()/rm,-1-rand()/rm,-1-rand()/rm);
  b[i*3+1]=float3(-1-rand()/rm,-1-rand()/rm,-1-rand()/rm);
  b[i*3+2]=float3(-1-rand()/rm,-1-rand()/rm,-1-rand()/rm);
  }
}
float3 convertVec(csVector3 v){
  return float3 (v.x,v.y,v.z);
}

extern float TestObbDisjoint (float3 X, float3 Y, float3 Z, float3 T,
                              float3  a, float3 b);
void testOBB() {
  for (unsigned int i=0;i<1024*32;++i) {

  float rm = RAND_MAX;
  csMatrix3 mat (rand()/rm,rand()/rm,rand()/rm,
                 rand()/rm,rand()/rm,rand()/rm,
                 rand()/rm,rand()/rm,rand()/rm);
  float3 T (rand()/rm*3-1.5f,rand()/rm*3-1.5f,rand()/rm*3-1.5f);
  
  float3 bX = convertVec(mat.Row1());
  float3 bY = convertVec(mat.Row2());
  float3 bZ = convertVec(mat.Row3());
  float3 r1 (rand()/rm,rand()/rm,rand()/rm);
  float3 r2 (rand()/rm,rand()/rm,rand()/rm);
  bool cputest = obb_disjoint(mat,T,r1,r2)?1:0;
  bool brooktest = TestObbDisjoint(bX,bY,bZ,T,r1,r2)?1:0;
  assert(cputest==brooktest);
  } 
}
extern void LoadPly (const char * file, std::vector<Tri> &ret);
extern unsigned int doCollide(unsigned int wida, unsigned int heia, Tri * a,
                              unsigned int bboxwida, unsigned int bboxheia, BBox * bboxesa,
                              unsigned int widb, unsigned int heib, Tri * b,
                              unsigned int bboxwidb, unsigned int bboxheib, BBox * bboxesb,
                              float3 rX, float3 rY, float3 rZ,
                              float3 trans,
                              float4 **intersections);

int main (int argc, char ** argv) {
  float3 * a= (float3*)malloc(sizeof(float3)*3*SIZEX*SIZEY);
  float3 * b= (float3*)malloc(sizeof(float3)*3*SIZEX*SIZEY);
  float4 * rez = (float4*) malloc (sizeof(float4)*SIZEX*SIZEY);
  float4 * rez_nodiv = (float4*) malloc (sizeof(float4)*SIZEX*SIZEY);
  float matrix[16]={1,0,0,0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1};
  srand(1);
  std::vector <bsp_polygon> model;
  std::vector<Tri> triangles;
  std::vector<BBox> bboxes;
  LoadPly ("bunny.ply",model);
  csRapidCollider collide(model);
  collide.createBrookGeometry(bboxes,triangles);
  printf ("Num BBoxes %d Num Triangles %d",bboxes.size(),triangles.size());
  testOBB();
  for (unsigned int k=0;k<12;++k) {
  SetTriangles(SIZEX*SIZEY,a,b);
  }
  int counter=0;
  int counter2=0;
  for (unsigned int j=0;j<1024*16;++j) {
  SetTriangles(SIZEX*SIZEY,a,b);
  doTest(matrix,SIZEX,SIZEY,a,b,rez);
  doTestNoDiv(matrix,SIZEX,SIZEY,a,b,rez_nodiv);
  for (unsigned int i=0;i<SIZEX*SIZEY;++i) {
    bool hit = (rez[i].x!=-1);
    bool khitnodiv = (rez_nodiv[i].x!=-1);
    bool hitnodiv = tri_contact_nodiv(a[i*3],a[i*3+1],a[i*3+2],
                                      b[i*3],b[i*3+1],b[i*3+2])?1:0;
    bool hitdiv = tri_contact(a[i*3],a[i*3+1],a[i*3+2],
                              b[i*3],b[i*3+1],b[i*3+2])?1:0;
    assert(hit==hitdiv);
    //    assert(khitnodiv==hitnodiv);
    if (hit!=hitnodiv)
      counter2++;
    if (hit&&!hitnodiv) {
      counter--;
    }
    if (hitnodiv&&!hit) {
      counter++;
    }
  }
  }
  printf ("Num differences btw dif and nodiv %d %d\n",counter2, counter);
  free (a);
  free (b);
  free (rez);
  return 0;
}
