////////////////////////////////////////////////////////////////////
//
// programmer: Paulius Micikevicius (paulius@drake.armstrong.edu)
//  Armstrong Atlantic State University
//
////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <cg/CG.h>
#include <cg/CGgl.h>
#include <stdio.h>

#include <fstream>
#include "timer.h"
#include "extensions.h"
#include "pbuffer.h"
#include "util.h"

HPBUFFERARB hPBuffer;
HDC			hPBufferDC;
HGLRC		hPBufferRC;
HDC			hFBufferDC;
HGLRC		hFBufferRC;

GLuint	fpTexture=0;	// fp32 texture object

CGcontext context;
CGprogram upper_sp_channel, lower_sp_channel;	// single-channel packed-texture programs
CGprofile fprofile=CG_PROFILE_FP30;

float	*upper_bounds=NULL;
float	*lower_bounds=NULL;

int	N;			// number of nodes (atoms)
int	nreps;		// number of times an experimetn is repeated (for timing experiments)
int print;		// flag indicating whether to print results (1 for print, 0 for no print)

int pbuffr_width, pbuffr_height;

void CGSetup();
void OGLSetup();
void InitPBuffer();
void PrintSingleChannelBuffer();
void ReleaseResources();
void GPU_Computation_Upper_Triangle_Packed(int n, int k);
void GPU_Computation_Lower_Triangle_Packed(int n, int k);
float SmoothUpper_TrianglePacked(float* upper_bounds, int n, int numframes=1);
float SmoothLower_TrianglePacked(float *bounds, int n, int numframes=1);
void Experiment();

void CGSetup()
{
	context=cgCreateContext();

	upper_sp_channel=cgCreateProgramFromFile(context, CG_SOURCE, "fshader.cg", fprofile, "upper_packed", NULL);
	cgGLLoadProgram(upper_sp_channel);

	lower_sp_channel=cgCreateProgramFromFile(context, CG_SOURCE, "fshader.cg", fprofile, "lower_packed", NULL);
	cgGLLoadProgram(lower_sp_channel);
}

void OGLSetup()
{
	/////////////////////////////////////
	// general OpenGL init

	hFBufferDC=wglGetCurrentDC();
	hFBufferRC=wglGetCurrentContext();

	PrepareExtensionFunctions();
	glGenTextures(1, &fpTexture);
	
	////////////////////////////////////
	// pbuffer-specific init

	pbuffr_width=N;
	pbuffr_height=N;
	if(0==PreparePBuffer(hPBuffer, hPBufferDC, hPBufferRC, pbuffr_width, pbuffr_height, hFBufferDC))
		cerr<<"failed to create pBuffer "<<endl;
	
	wglMakeCurrent(hPBufferDC, hPBufferRC);
	CGSetup();
	wglShareLists(hPBufferRC, hFBufferRC);
	
	wglMakeCurrent(hFBufferDC, hFBufferRC);
}

void GPU_Computation_Upper_Triangle_Packed(int n, int k)
{
	// computation for the k-th iteration
	glBegin(GL_TRIANGLES);
		////////////////////////////////////////////////////
		// i<j, i<=k, j<=k
		//
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, 0, 0, 0, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 0, k);
		glVertex2f(0, 0);

		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, k+1, k+1, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k+1, k);
		glVertex2f(k+1, k+1);
		
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, 0, k+1, 0, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k+1, k);
		glVertex2f(0, k+1);

		////////////////////////////////////////////////////
		// i<j, i>k, j>k
		//
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, k+1, k, k+1);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, k+1);
		glVertex2f(k+1, k+1);

		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, n, n, k, n);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, n);
		glVertex2f(n, n);
		
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, n, k, k+1);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, n);
		glVertex2f(k+1, n);
	glEnd();

	glBegin(GL_QUADS);
		////////////////////////////////////////////////////
		// i<j, i<k, j>k
		//
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, 0, k+1, 0, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, k+1);
		glVertex2f(0, k+1);
		
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, k+1, k+1, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, k+1);
		glVertex2f(k+1, k+1);

		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, n, k+1, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, n);
		glVertex2f(k+1, n);

		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, 0, n, 0, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, n);
		glVertex2f(0, n);	
	glEnd();
}

void GPU_Computation_Lower_Triangle_Packed(int n, int k)
{
	// computation for the k-th iteration
	glBegin(GL_TRIANGLES);
		////////////////////////////////////////////////////
		// i>j, i<=k, j<k
		//
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, 1, 0, k, 1);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, 0);
		glVertex2f(1, 0);

		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, k, k, k+1);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, k);
		glVertex2f(k+1, k);
		
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, 0, k, k+1);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, 0);
		glVertex2f(k+1, 0);

		////////////////////////////////////////////////////
		// i>j, i>k, j>k
		//
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, k+1, k+1, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k+1, k);
		glVertex2f(k+1, k+1);

		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, n, k+1, n, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k+1, k);
		glVertex2f(n, k+1);
		
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, n, n-1, n, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, n-1, k);
		glVertex2f(n, n-1);
	glEnd();

	glBegin(GL_QUADS);
		////////////////////////////////////////////////////
		// i>j, i<k, j>=k
		//
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, 0, k+1, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, 0);
		glVertex2f(k+1, 0);
		
		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, n, 0, n, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, 0);
		glVertex2f(n, 0);

		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, n, k+1, n, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, k+1);
		glVertex2f(n, k+1);

		glMultiTexCoord4fARB(GL_TEXTURE0_ARB, k+1, k+1, k+1, k);
		glMultiTexCoord2fARB(GL_TEXTURE1_ARB, k, k+1);
		glVertex2f(k+1, k+1);
	glEnd();

}

void InitPBuffer()
{
	// select the context
	wglMakeCurrent(hPBufferDC, hPBufferRC);
		
	//glClearColor(INF, INF, INF, INF);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	// set the matrices
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, pbuffr_width, 0.0f, pbuffr_height, -1.0f, 1.0f);
	glViewport(0, 0, pbuffr_width, pbuffr_height);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
				
	// set the states
	glEnable(GL_TEXTURE_RECTANGLE_NV);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	// enable the fragment shader
	cgGLEnableProfile(fprofile);
}

void PrintSingleChannelBuffer()
{
	int i,j;
	
	float *temp=new float[N*N];
	if(temp==NULL)
	{
		cerr<<"couldn't allocate memory (PrintTexture)"<<endl;
		return;
	}

	glReadPixels(0, 0, N, N, GL_RED, GL_FLOAT, temp);

	for(i=0;i<N;i++)
	{
		cout<<"  ";
		for(j=0;j<N;j++)
			if(temp[j*N+i]!=INF)
				cout<<temp[j*N+i]<<", ";
			else
				cout<<"., ";
		cout<<endl;
	}

	delete[] temp;
}

float SmoothUpper_TrianglePacked(float* upper_bounds, int n, int numframes)
{
	CTimer	timer;
	int currPB;
	
	// set up the pbuffer for computation
	InitPBuffer();	
	cgGLBindProgram(upper_sp_channel);

	// initialize the arrays to help with ping-ponging between the buffers
	GLenum draw_buffer[2]={GL_FRONT_LEFT, GL_BACK_LEFT};
	GLenum tex_buffer[2]={WGL_BACK_LEFT_ARB, WGL_FRONT_LEFT_ARB};

	glFinish();
	timer.Start();

	for(int i=0;i<numframes;i++)
	{
		currPB=0;
		
		// write the initial upper-bound matrix to the front buffer
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, fpTexture);
		wglBindTexImageARB(hPBuffer,WGL_FRONT_LEFT_ARB);
		glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, n, n, GL_RED, GL_FLOAT, upper_bounds);
		wglReleaseTexImageARB(hPBuffer,WGL_FRONT_LEFT_ARB);

		// go through the n iterations of the algorithm
		for(int k=0;k<n;k++)
		{
			currPB=!currPB;	// update the index for ping-ponging

			wglBindTexImageARB(hPBuffer,tex_buffer[currPB]);
			glDrawBuffer(draw_buffer[currPB]);
			GPU_Computation_Upper_Triangle_Packed(n, k);
			wglReleaseTexImageARB(hPBuffer,tex_buffer[currPB]);
		}
	}

	glFinish();
	float et=1000*timer.GetET()/numframes;
	cerr<<"----------------"<<endl<<"Triangle Upper:\t"<<et<<"ms"<<endl;

	if(1==print)
	{
		cout<<"--- Upper Bounds ----"<<endl;	
		glReadBuffer(draw_buffer[currPB]);
		PrintSingleChannelBuffer();
	}
	
	// restore the framebuffer context
	wglMakeCurrent(hFBufferDC, hFBufferRC);

	return et;
}

float SmoothLower_TrianglePacked(float *bounds, int n, int numframes)
{
	CTimer	timer;
	int i,k;
	int currPB;
	
	// set up the pbuffer for computation
	InitPBuffer();	
	cgGLBindProgram(lower_sp_channel);

	// initialize the arrays to help with ping-ponging between the buffers
	GLenum draw_buffer[2]={GL_FRONT_LEFT, GL_BACK_LEFT};
	GLenum tex_buffer[2]={WGL_BACK_LEFT_ARB, WGL_FRONT_LEFT_ARB};

	glFinish();
	timer.Start();

	for(i=0;i<numframes;i++)
	{
		currPB=0;
		
		// write the initial distance-bound matrix to both buffers
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, fpTexture);
		wglBindTexImageARB(hPBuffer,WGL_FRONT_LEFT_ARB);
		glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, n, n, GL_RED, GL_FLOAT, bounds);
		wglReleaseTexImageARB(hPBuffer,WGL_FRONT_LEFT_ARB);

		wglBindTexImageARB(hPBuffer,WGL_BACK_LEFT_ARB);
		glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, n, n, GL_RED, GL_FLOAT, bounds);
		wglReleaseTexImageARB(hPBuffer,WGL_BACK_LEFT_ARB);

		// go through the n iterations of the algorithm
		for(k=0;k<n;k++)
		{
			currPB=!currPB;	// update the index for ping-ponging

			wglBindTexImageARB(hPBuffer,tex_buffer[currPB]);
			glDrawBuffer(draw_buffer[currPB]);
			GPU_Computation_Lower_Triangle_Packed(n, k);
			wglReleaseTexImageARB(hPBuffer,tex_buffer[currPB]);
		}	
	}

	glFinish();
	float et=1000*timer.GetET()/numframes;
	cerr<<"----------------"<<endl<<"Triangle Lower:\t"<<et<<"ms"<<endl;

	if(1==print)
	{
		cout<<"--- Lower Bounds ----"<<endl;	
		glReadBuffer(draw_buffer[currPB]);
		PrintSingleChannelBuffer();
	}
	
	// restore the framebuffer context
	wglMakeCurrent(hFBufferDC, hFBufferRC);

	return et;
}

void ReleaseResources()
{
	// release the pbuffers
	wglDeleteContext(hPBufferRC);
	wglReleasePbufferDCARB(hPBuffer, hPBufferDC);
	wglDestroyPbufferARB(hPBuffer);
}

void Experiment()
{
	float upper_time, lower_time;	// times 
	int i;

	////////////////////////////////////////////////////////////////////
	// allocate and initialize the distance-bound data
	//
	float *upper_bounds=new float[N*N];	// just the upper bounds
	float *lower_bounds=new float[N*N];	// just the lower bounds
	float *bounds=new float[N*N];			// both upper and lower bounds

	if(NULL==upper_bounds || NULL==lower_bounds || NULL==bounds)
	{
		cerr<<"couldn't allocate the adjacency matrices"<<endl;
		return ;
	}

	PrepareBounds(upper_bounds, lower_bounds, N);	// generate upper and lower bounds
	
	WarshallFloyd(bounds, upper_bounds, N);			//
													//
	for(i=1;i<N;i++)								// This block of code prepares the matrix
		for(int j=0;j<i;j++)						// for lower bound smoothing.
			bounds[i*N+j]=lower_bounds[i*N+j];		// The array contains the smoothed upper
													// bounds and the original lower bounds.
	for(i=0;i<N;i++)								// Transpose is simply for properly aligning
		bounds[i*N+i]=0;							// the matrix with the texture.
													//
	Transpose(bounds, N);							//


	////////////////////////////////////////////////////////////////////
	// execute the experiments
	//
	upper_time=SmoothUpper_TrianglePacked(upper_bounds, N, nreps);
	lower_time=SmoothLower_TrianglePacked(bounds, N, nreps);


	////////////////////////////////////////////////////////////////////
	// write timing results to a file
	//
	char filename[30];
	sprintf(filename, "timings-%d.txt", N);
	ofstream outfile(filename);
	outfile<<"N="<<N<<", averaged "<<nreps<<" times"<<endl;
	outfile<<"Upper Single:\t"<<upper_time<<endl;
	outfile<<"Lower Single:\t"<<lower_time<<endl;
	outfile.close();

	////////////////////////////////////////////////////////////////////
	// free the memory used for distance-bounds
	//
	if(upper_bounds)
		delete[] upper_bounds;
	if(lower_bounds)
		delete[] lower_bounds;
	if(bounds)
		delete[] bounds;
}

int main(int argc, char* argv[])
{
	N=8;
	nreps=0;

	if(argc>=2)
		N=atoi(argv[1]);
	if(argc==3)
		nreps=atoi(argv[2]);

	// determine whether to print out the updated bounds
	//	  command-line parameter 0 will execute the experiment once and print
	if(0==nreps)
		print=nreps=1;
		
	else
		print=0;

	glutInit( &argc, argv );
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA);
	glutInitWindowSize(10, 10);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Distance-Bounds");
	
	OGLSetup();

	Experiment();
		
	return 0;
}
