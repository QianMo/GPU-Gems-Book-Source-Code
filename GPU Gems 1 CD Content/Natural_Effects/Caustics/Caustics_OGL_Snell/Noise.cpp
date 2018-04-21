#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include "noise.h"


#define DOT(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

double vec[3];

#define B 256

static p[B+B+2];
static double g[B+B+2][3];
static start=1;

void setup(int i,int &b0,int &b1,double &r0,double &r1)
{
float t;
t=(float)vec[i]+10000.f;
b0=((int)t) & (B-1);
b1=(b0+1) & (B-1);
r0=t-(int)t;
r1=r0 - 1.;
}


static void init()
{
int i,j,k;
double v[3],s;

srand(1);
for (i=0;i<B;i++)
        {
        do
                {
                for (j=0;j<3;j++)
                        v[j]=(double)((rand() % (B+B)) -B)/B;
                s=DOT(v,v);
                } while (s>1.0);
        s=sqrt(s);
        for (j=0;j<3;j++)
                g[i][j]=v[j]/s;
        }
for (i=0;i<B;i++)
        p[i]=i;
for (i=B;i>0;i-=2)
        {
        k=p[i];
        p[i]=p[j=rand()%B];
        p[j]=k;
        }
for (i=0;i<B+2;i++)
        {
        p[B+i]=p[i];
        for (j=0;j<3;j++)
                g[B+i][j]=g[i][j];
        }
}



double at(double rx,double ry,double rz,double *q)
{
return (rx*q[0] + ry*q[1] + rz*q[2]);
}


double s_curve(double t)
{
return (t*t*(3.-2.*t));
}

double lerp(double t,double a,double b)
{
return (a+t*(b-a));
}


double noise(double x, double y, double z)
{

int bx0,bx1,by0,by1,bz0,bz1,b00,b10,b01,b11;
double rx0,rx1,ry0,ry1,rz0,rz1, *q, sx, sy,sz,a,b,c,d,u,v;
register i,j;
double res;

vec[0]=x;
vec[1]=y;
vec[2]=z;
if (start)
        {
        start=0;
        init();
        }
setup(0,bx0,bx1,rx0,rx1);
setup(1,by0,by1,ry0,ry1);
setup(2,bz0,bz1,rz0,rz1);
i=p[bx0];
j=p[bx1];
b00=p[i+by0];
b10=p[j+by0];
b01=p[i+by1];
b11=p[j+by1];

sx=s_curve(rx0);
sy=s_curve(ry0);
sz=s_curve(rz0);

q=g[b00+bz0]; u=at(rx0,ry0,rz0,q);
q=g[b10+bz0]; v=at(rx1,ry0,rz0,q);
a=lerp(sx,u,v);

q=g[b01+bz0]; u=at(rx0,ry1,rz0,q);
q=g[b11+bz0]; v=at(rx1,ry1,rz0,q);
b=lerp(sx,u,v);

c=lerp(sy,a,b);

q=g[b00+bz1]; u=at(rx0,ry0,rz1,q);
q=g[b10+bz1]; v=at(rx1,ry0,rz1,q);
a=lerp(sx,u,v);

q=g[b01+bz1]; u=at(rx0,ry1,rz1,q);
q=g[b11+bz1]; v=at(rx1,ry1,rz1,q);
b=lerp(sx,u,v);

d=lerp(sy,a,b);

res = 1.5*lerp(sz,c,d);
if (res>1.0) res=1.0;
if (res<-1.0) res=-1.0;
return res;
}
