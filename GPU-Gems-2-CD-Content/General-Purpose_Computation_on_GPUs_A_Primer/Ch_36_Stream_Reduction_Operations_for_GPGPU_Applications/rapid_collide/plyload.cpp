#include <stdio.h>
#include "rapcol.h"
#include <vector>
#include <brook/brook.hpp>
using std::vector;
void LoadPly (const char * file,vector<Tri> &ret) {
  float ver;char mchar;
  int numvertex,numface,propertycount=0;
  vector<tri_vertex_t> vertices;
  
  FILE * fp = fopen (file,"r");
  if (!fp) return;
  fscanf(fp,"ply\nformat ascii %f\n",&ver);
  int comment;
  while (1==(comment=fscanf(fp,"commen%c %*[^\n]\n",&mchar))) {
     int j=1;
  }
  fscanf(fp,"element vertex %d\n",&numvertex);
  while (1==fscanf(fp,"propert%c %*s %*s\n",&mchar)) {
    propertycount++;
  }
  fscanf(fp,"element face %d\n",&numface);
  while (1==fscanf(fp,"propert%c %*[^\n]\n",&mchar));
  fscanf(fp,"end_header\n");
  for (int i=0;i<numvertex;++i) {
    tri_vertex_t in;
    fscanf(fp,"%f %f %f",&in.x,&in.y,&in.z);
    for (int i=3;i<propertycount;++i) {
      fscanf(fp,"%*f");
    }
    vertices.push_back(in);
  }
  for (int i=0;i<numface;++i) {
    int num=0;
    int a,b,c,count=0;
    fscanf(fp,"%d",&num);
    if (num>=2) {
      count=2;
      fscanf(fp,"%d %d",&a,&b);
    }
    for (int i=count;i<num;++i) {
      fscanf(fp,"%d",&c);
      ret.push_back(Tri());
      ret.back().A = vertices[a];
      ret.back().B = vertices[b];
      ret.back().C = vertices[c];
      a = b;
      b = c;
    }
    fscanf(fp,"\n");
  }
  fclose(fp);
}
