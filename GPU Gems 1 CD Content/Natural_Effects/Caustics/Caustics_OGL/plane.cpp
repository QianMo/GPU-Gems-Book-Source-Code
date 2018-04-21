// Disable warning for loss of data
#pragma warning( disable : 4244 )  
#pragma warning( disable : 4305 )  


#include "point.h"
#include "plane.h"


plane::plane()
{
a=0;
b=0;
c=0;
d=0;
n.create(a,b,c);
}


plane::plane(float pa,float pb,float pc,float pd)
{
a=pa;
b=pb;
c=pc;
d=pd;
n.create(a,b,c);
n.normalize();
}


void plane::create(float pa,float pb,float pc,float pd)
{
a=pa;
b=pb;
c=pc;
d=pd;
n.create(a,b,c);
n.normalize();
}


plane::plane(point p1,point p2,point p3)
{
point p,q,r;

p = p2 - p1;
q = p3 - p1;
r = p^q;
r.normalize();
a = r.x;
b = r.y;
c = r.z;
n = r;
d = - (a*p1.x+b*p1.y+c*p1.z);
}


void plane::negate()
{
n.negate();
a=n.x;
b=n.y;
c=n.z;
d=-d;
}


void plane::create(point p1,point p2,point p3)
{
point p,q,r;

p = p2 - p1;
q = p3 - p1;
r = p^q;
r.normalize();
a = r.x;
b = r.y;
c = r.z;
n = r;
d = - (a*p1.x+b*p1.y+c*p1.z);
}



double plane::testline(point r,point vec,point &pres)
// return value is distance along the line to the collision point
{
double fres;
double t;
fres=vec*n;
pres=r;
if ((fres)!=0)
	{
	t=-(n*r + d)/fres;
	pres=r+vec*t;
	}
// hack... arreglar
else t=0;
return t;
}


double plane::testpoint(point p)
{
double res;

res=p*n+d;
return (res);
}


int plane::magicnumber(point p)
{
float val;
int valor;

val = a*p.x+b*p.y+c*p.z + d;
if (val<0) valor=-1;
else if (val>0) valor=1;
return (valor);
}


double plane::evalxy(double x,double y)
{
if (c==0) return 0;
else return -(a*x+b*y+d)/c;
}


double plane::evalxz(double x,double z)
{
if (b==0) return 0;
else return -(a*x+c*z+d)/b;
}


double plane::evalyz(double y,double z)
{
if (a==0) return 0;
else return -(b*y+c*z+d)/a;
}


int plane::testtri(point p1,point p2,point p3)
{
float valor1,valor2,valor3;
int valor;

valor1 = a*p1.x+b*p1.y+c*p1.z - d;
if (valor1>0) valor1=-1;
else if (valor1>0) valor1=+1;
valor2 = a*p2.x+b*p2.y+c*p2.z - d;
if (valor2<0) valor2=-1;
else if (valor2>0) valor2=+1;
valor3 = a*p3.x+b*p3.y+c*p3.z - d;
if (valor3<0) valor3=-1;
else if (valor3>0) valor3=+1;
valor = (int)(valor1+valor2+valor3);
if (valor==-3) valor=-1;
else if (valor==+3) valor=+1;
else valor = 0;
return(valor);
}


point plane::getpointfromplane()
// gives us a random point on the surface of the plane
{
point res;

if (a!=0)
	{
	res.create(-d/a,0,0);
	return res;
	}
if (b!=0)
	{
	res.create(0,-d/a,0);
	return res;
	}
if (c!=0)
	{
	res.create(0,0,-d/a);
	}
return res;
}


void plane::operator=(plane p)
{
a=p.a;
b=p.b;
c=p.c;
d=p.d;
n=p.n;
}


plane::~plane()
{


}