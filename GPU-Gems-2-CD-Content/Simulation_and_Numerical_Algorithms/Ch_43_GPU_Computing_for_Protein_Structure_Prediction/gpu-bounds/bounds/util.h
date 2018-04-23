#ifndef _MP_UTIL_
#define	_MP_UTIL_

#include <fstream>
#include <math.h>

#define	INF		32000

int WarshallFloyd(float *out_graph, float *in_graph, int dim);
int Copy(float *out_graph, float *in_graph, int dim);

int UnitPath(float *graph, int dim)
{
	int error=1;	// no error
	int i,j;

	for(i=0;i<dim;i++)
		for(j=i;j<dim;j++)
			if(i+1==j)
				graph[i*dim+j]=graph[j*dim+i]=1.0f;
			else
				graph[i*dim+j]=graph[j*dim+i]=INF;

	return error;
}

void Transpose(float *graph, int dim)
{
	int i,j;
	float temp;

	for(i=0;i<dim-1;i++)
		for(j=i+1;j<dim;j++)
		{
			temp=graph[i*dim+j];
			graph[i*dim+j]=graph[j*dim+i];
			graph[j*dim+i]=temp;
		}
}

int WarshallFloyd(float *out_graph, float *in_graph, int dim)
{
	int error=1;	// no error
	int i,j,k;

	Copy(out_graph, in_graph, dim);

	for(k=0;k<dim;k++)
		for(i=0;i<dim;i++)
			for(j=0;j<dim;j++)
			{
				float temp=out_graph[i*dim+k]+out_graph[k*dim+j];
				if(temp<out_graph[i*dim+j])
					out_graph[i*dim+j]=temp;
			}

	return error;
}

int Zero(float *matrix, int dim)
{
	int i,j;
	for(i=0;i<dim;i++)
		for(j=0;j<dim;j++)
			matrix[i*dim+j]=0;

	return 1;
}

void PrepareBounds(float *u_bounds, float *l_bounds, int dim)
{
	UnitPath(u_bounds, dim);
	Zero(l_bounds, dim);
	for(int i=0;i<dim-1;i++)
	{
		l_bounds[i*dim+i+1]=l_bounds[(i+1)*dim+i]=1+2*(i%2);
		u_bounds[i*dim+i+1]=u_bounds[(i+1)*dim+i]=1+2*(i%2);
	}
}

int Copy(float *out_graph, float *in_graph, int dim)
{
	int error=1;	// no error
	int i,j;

	for(i=0;i<dim;i++)
		for(j=0;j<dim;j++)
			out_graph[i*dim+j]=in_graph[i*dim+j];

	return error;
}

#endif