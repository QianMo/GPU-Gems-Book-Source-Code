#include <stdio.h>
#include <list>
#include <vector>
#include <assert.h>
using namespace std;
struct Vec {
  double x,y,z;
  Vec(float xx, float yy, float zz) {
    x=xx;y=yy;z=zz;
  }
  Vec(){}
  bool operator == (const Vec &e) const{
    return x==e.x&&y==e.y&&z==e.z;
  }  
  void print ()const{
    printf ("%lf, %lf, %lf\n",x,y,z);
  }
  bool operator <(const Vec &v)const{
    if (x<v.x)
      return true;
    else if (x==v.x) {
      if (y<v.y)
        return true;
      else if (y==v.y) 
        return z<v.z;
      else
        return false;
    }else{
      return false;
    }
  }
};
struct Edge {
  Vec a,b;
  Edge (Vec aa, Vec bb) {
    a=aa;
    b=bb;
  }
  void print()const {
    //printf ("{");
    a.print();
    //printf(", ");
    b.print();
    //printf ("}\n");
  }
  bool operator <(const Edge &e)const {
    if (a<e.a) return true;
    if (a==e.a)
      return b<e.b;
    return false;
  }
  bool operator == (const Edge &e) const{
    return (a==e.a&&b==e.b)||(a==e.b&&b==e.a);
  }
};
vector<Vec> model;
list<Edge> edges;
int main (int argc, char** argv) {
  char * modelfn="data";
  if (argc>1){
    modelfn = argv[1];
  }
  FILE * fp = fopen (modelfn,"r");
  unsigned int nvertices=0;
  fscanf(fp,"%d\n",&nvertices);

  unsigned int i;
  for (i=0;i<nvertices;++i) {
    Vec next(-666,-666,-666);
    int hits=
    fscanf (fp,"%lf, %lf, %lf\n",&next.x,&next.y,&next.z);
    assert (hits==3);
    model.push_back(next);
  }
  fclose(fp);
  for (i=0;i<model.size();i+=3) {
    edges.push_back(Edge(model[i],model[i+1]));
    edges.push_back(Edge(model[i],model[i+2]));
    edges.push_back(Edge(model[i+1],model[i+2]));        
  }
  edges.sort();
  for (list<Edge>::iterator i=edges.begin();i!=edges.end();++i) {
    //(*i).print();
  }
  for (list<Edge>::iterator i=edges.begin();i!=edges.end();) {
    list<Edge>::iterator j=i;
    j++;
    bool found=false;
    for (;j!=edges.end();++j) {
      if (*i==*j) {
        assert (i!=j);
        edges.erase(j);
        i=edges.erase(i);
        found=true;
        break;
      }
    }      
    if (!found){
      //printf ("Edge missing:");
      //(*i).print();
      i++;

    }
  }
  {
    printf ("%d\n",edges.size()*3);
    for (list<Edge>::iterator i=edges.begin();i!=edges.end();++i) {
      (*i).print();
      (*i).a.print();
    }
  }

  fprintf (stderr,"%d Edges Left\n",edges.size());
  return 0;
}
