#include <vector>
#include "subdiv.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
using std::vector;
float neighboreps=0.0001f;
bool eq(float a, float b) {
  return fabs(a-b)<neighboreps;
}
bool ne(float4 a, float4 b){
	return !(eq(a.x, b.x) && eq(a.y, b.y) && eq(a.z, b.z));
}
void LoadPly (const char * file,vector<STri> &ret);
bool operator ==(const float4 &a, const float4 &b) {
   return eq(a.x,b.x)&&eq(a.y,b.y)&&eq(a.z,b.z);
}
void checkEdgeNeighbor(const float4 &a, const float4 &b, float4 &c,const STri &t){ 
   if ((t.A==a&&t.B==b)||(t.A==b&&t.B==a))
      c = t.C;
   if ((t.A==a&&t.C==b)||(t.A==b&&t.C==a))
      c = t.B;
   if ((t.B==a&&t.C==b)||(t.B==b&&t.C==a))
      c = t.A;
}

void checkEdgeNeighbor2(const float4 &a, const float4 &alreadyhave, const float4 &b, float4 &c, const STri &t){ 
	if (((t.A==a&&t.B==b)||(t.A==b&&t.B==a))&&(ne(t.C,alreadyhave)))
      c = t.C;
	if (((t.A==a&&t.C==b)||(t.A==b&&t.C==a))&&(ne(t.B,alreadyhave)))
      c = t.B;
	if (((t.B==a&&t.C==b)||(t.B==b&&t.C==a))&&(ne(t.A,alreadyhave)))
      c = t.A;
}
void checkNeighbors (STri * t, Neighbor * a, Neighbor* b, unsigned int tListsize) {
  for (unsigned int i=0;i<tListsize;++i) {
    if (isinf_float(b[i].AB.x)||isinf_float(a[i].AB.x))
      continue;

    float4 * af = (float4*)(a+i);
    float4 * bf = (float4*)(b+i);
    fprintf (stderr,"(%.3f %.3f %.3f) (%.3f %.3f %.3f) (%.3f %.3f %.3f)\n",t[i].A.x,t[i].A.y,t[i].A.z,t[i].B.x,t[i].B.y,t[i].B.z,t[i].C.x,t[i].C.y,t[i].C.z);
    
    for (unsigned int j=0;j<9;++j) {
      if (ne(af[j],bf[j])) {
        fprintf (stderr,"neighbor %d mismatch %.3f %.3f %.3f and %.3f %.3f %.3f\n",
                j,af[j].x,af[j].y,af[j].z,bf[j].x,bf[j].y,bf[j].z);
      }else fprintf(stderr,"%d ok %.3f %.3f %.3f\n",j,af[j].x,af[j].y,af[j].z);
    }
  }
}

void recomputeNeighbors (STri * tList, Neighbor* neigh, unsigned int tListsize) {
   for (unsigned int i=0;i<tListsize;++i) {
      unsigned int j;
      float4 zero4(0,0,0,0);
      STri t = tList[i];
      t.A.w=0;
      t.B.w=0;
      t.C.w=0;
      Neighbor n;
      memset(&n,0,sizeof(Neighbor));
      if (1) {
        for (j=0;j<tListsize;++j) {
          //check for AB, BC, AC
          STri nei=tList[j];
					if(j==i)
						continue;
          checkEdgeNeighbor(t.A,t.B,n.AB,nei);
          checkEdgeNeighbor(t.A,t.C,n.AC,nei);
          checkEdgeNeighbor(t.B,t.C,n.BC,nei);         
        }
        for (j=0;j<tListsize;++j) {
          STri nei=tList[j];
					if(j==i)
						continue;
          checkEdgeNeighbor2(t.A,t.C,n.AC,n.AAC,nei);
          checkEdgeNeighbor2(t.A,t.B,n.AB,n.AAB,nei);
          checkEdgeNeighbor2(t.B,t.C,n.BC,n.BBC,nei);
          checkEdgeNeighbor2(t.B,t.A,n.AB,n.ABB,nei);
          checkEdgeNeighbor2(t.C,t.B,n.BC,n.BCC,nei);
          checkEdgeNeighbor2(t.C,t.A,n.AC,n.ACC,nei);
        }
      }
      if (n.ABB==n.BBC)
        n.BBC=zero4;
      if (n.BCC==n.ACC)
        n.ACC=zero4;
      if (n.AAB==n.AAC)
        n.AAB=zero4;
      if (n.AB==n.ABB)
        n.ABB=zero4;
      if (n.AAB==n.AB)
        n.AAB=zero4;
      if (n.AAB==n.AC)
        n.AAB=zero4;
      if (n.AAC==n.AB)
        n.AAC=zero4;
      if (n.AAC==n.AC)
        n.AAC=zero4;
      if (n.ACC==n.AC)
        n.ACC=zero4;
      if (n.ACC==n.BC)
        n.ACC=zero4;
      if (n.BCC==n.AC)
        n.BCC=zero4;
      if (n.BCC==n.BC)
        n.BCC=zero4;
      if (n.BBC==n.AB)
        n.BBC=zero4;
      if (n.BBC==n.BC)
        n.BBC=zero4;
      if (n.ABB==n.AB)
        n.ABB=zero4;
      if (n.ABB==n.BC)
        n.ABB=zero4;

      neigh[i]=n;
   }
}
unsigned int loadModelData(const char * filename,
                           STri ** tri,
                           Neighbor ** neigh) {
   std::vector<STri>tList;
   if (strcmp(filename,"dabunny")==0) 
     LoadPly("bunny.ply",tList);
   else
     LoadPly(filename,tList);
   *tri = (STri*)malloc(sizeof(STri)*tList.size());
   *neigh =(Neighbor*)malloc(sizeof(Neighbor)*tList.size());
   for (unsigned int i=0;i<tList.size();++i) {
     float eps=.015625;
     if (1||tList[i].A.z==0) 
       tList[i].A.z+=eps;
     if (1||tList[i].B.z==0)       
       tList[i].B.z+=eps;
     if(1||tList[i].C.z==0) {
       tList[i].C.z+=eps;
     }
     (*tri)[i]=tList[i];
   }
   if (strcmp(filename,"dabunny")) {
     recomputeNeighbors(*tri,*neigh,tList.size());
   }
   return tList.size();
}

int myLog (int l) {
  if (l==0)
    return 0;
  int i;
  for (i=0;i<32;++i) {
    if ((1<<i)>=l)
      break;
  }
  return i;
}
float dawt (float3 a, float3 b) {
  return a.x*b.x+a.y*b.y+a.z*b.z;
}
extern void computeFunctionCallPattern(float epsilon,
                                       int argc, 
                                       char ** argv, 
                                       int numTri,
                                       STri*triangles,
                                       Neighbor *neigh){
  printf ("Compute Function Call Pattern with eps = %f\n",epsilon);
  std::string filename = "sum-subdiv";
  int i;
  for (i=1;i<argc;++i) {
    if (argv[i][0]>='0'&&argv[i][0]<='9') {
      filename+="-";
    }
    filename+=argv[i];
  }
  char bleh[128];
  sprintf(bleh,"-eps%f",epsilon);
  filename+=bleh;
  int EstablishGuess=0;
  int UpdateGuess = 0;
  int RelativeGather=0;
  int produceTriP=0;
  int splitTriangles=0;
  int writeFinalTriangles=0;
  vector <STri> trivec;
  for (i=0;i<numTri;++i) {
    trivec.push_back(triangles[i]);
  }
  do {
    produceTriP+=trivec.size();
    EstablishGuess+=2*trivec.size();
    UpdateGuess+=2*(myLog(trivec.size())-1)*trivec.size();
    vector <STri> split;
    vector <STri> nosplit;
    //do vout stage on CPU;
    for (unsigned int j=0;j<trivec.size();++j) {
      STri t = trivec[j];
      float3 ab(t.A.x-t.B.x,t.A.y-t.B.y,t.A.z-t.B.z);
      float3 ac(t.A.x-t.C.x,t.A.y-t.C.y,t.A.z-t.C.z);
      float3 bc(t.B.x-t.C.x,t.B.y-t.C.y,t.B.z-t.C.z);
      if (dawt(ab,ab)<epsilon&&
          dawt(ac,ac)<epsilon&&
          dawt(bc,bc)<epsilon) {
        nosplit.push_back(t);
      }else {
        float4 a2b(.5f*(t.A.x+t.B.x),
                   .5f*(t.A.y+t.B.y),
                   .5f*(t.A.z+t.B.z),0);
                   
        float4 a2c(.5f*(t.A.x+t.C.x),
                   .5f*(t.A.y+t.C.y),
                   .5f*(t.A.z+t.C.z),0);
        float4 b2c(.5f*(t.B.x+t.C.x),
                   .5f*(t.B.y+t.C.y),
                   .5f*(t.B.z+t.C.z),0);
        STri u;
        u.A=t.A;
        u.B=a2b;
        u.C=a2c;
        split.push_back(u);
        u.A=a2b;
        u.B=t.B;
        u.C=b2c;
        split.push_back(u);
        u.A=a2c;
        u.B=b2c;
        u.C=t.C;
        split.push_back(u);
        u.A=a2c;
        u.B=a2b;
        u.C=b2c;
        split.push_back(u);
      }
    }
    RelativeGather+=split.size()/4;
    splitTriangles+=split.size()/4;
    trivec.swap(split);
    RelativeGather+=nosplit.size();
    writeFinalTriangles+=nosplit.size();
  } while (trivec.size());
  FILE * fp = fopen (filename.c_str(),"w");
  fprintf (fp,"computeNeighbors %d\n",splitTriangles);  
  fprintf (fp,"EstablishGuess %d\n",EstablishGuess);
  fprintf (fp,"linearReorgSplitTriangles %d\n",4*splitTriangles);
  fprintf (fp,"NanToBoolRight %d\n",EstablishGuess);
  fprintf (fp,"NanToRight %d\n",UpdateGuess);
  fprintf (fp,"produceTriP %d\n",produceTriP);
  fprintf (fp,"splitTriangles %d\n",splitTriangles);
  fprintf (fp,"RelativeGather %d\n",RelativeGather);
  fprintf (fp,"UpdateGuess %d\n",UpdateGuess);
  fprintf (fp,"writeFinalTriangles %d\n",writeFinalTriangles*3);
  fclose(fp);
}
