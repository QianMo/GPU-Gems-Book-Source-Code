#include "defines.h"
#include "GLSLShader.h"
#include "RenderTexture.h"                                                   
#include "GbTimer.h"
#include <GL/glut.h>

// the double-buffer texture
RenderTexture *buffer;
unsigned int targetBuffer = 1;
#ifdef _WIN32
GLenum wglTargets[] = { WGL_FRONT_LEFT_ARB, WGL_BACK_LEFT_ARB };
#endif
GLenum glTargets[] = { GL_FRONT_LEFT, GL_BACK_LEFT };

// the sorting shaders
GLSLShader transitionSort;
GLSLShader oddevenMergeSort;
GLSLShader bitonicMergeSort_Col, bitonicMergeSort_RowN, bitonicMergeSort_Row01;
GLuint bitonicSortLists=0, currentBitonicSortList=0;

GLSLShader renderShader;
bool sortFloats = false;

// field size to sort
int logFieldsize = 8;
int fieldsize = (1<<logFieldsize);

// number of sorting steps to execute on next redraw
int stepsToDo = 0;
// number of steps left until sort is complete
int stepsLeft = 0;
// number of steps needed for full sort
int totalSteps = 0;
// current parameters
int stage=0, pass=0;
int sortAlgorithm = 1;
int width=0, height=0;

typedef enum { RANDOM_DATA, SORTED_DATA, INVERSE_SORTED_DATA } DataStyle;

bool dumpit=false;
bool perfTest=false;

char sortAlgText[4][64] = { "empty",
			    "odd-even transition sort", 
			    "odd-even merge sort",
			    "bitonic merge sort" };

RenderTexture* 
createRenderTexture(const char *initstr)
{
    staticdebugmsg("createRenderTexture","Creating with init string: \""<<initstr<<"\"");

    RenderTexture *rt2 = new RenderTexture;
    rt2->Reset(initstr);
    if (!rt2->Initialize(fieldsize,fieldsize)) {
        staticerrormsg("createRenderTexture", "RenderTexture Initialization failed !");
	return NULL;
    }

    // the sorting needs nearest neighbor sampling
    if (rt2->IsTexture()) {
	rt2->Bind();
	glTexParameteri(rt2->GetTextureTarget(), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(rt2->GetTextureTarget(), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkGLError("createRenderTexture set sampling");
    }

    return rt2;
}

bool
loadShader()
{
    staticdebugmsg("loadShader","Initializing shaders ...");

    return transitionSort.loadFromFile("./shader/transitionSort.vs","./shader/transitionSort.fs") &&
	oddevenMergeSort.loadFromFile("./shader/oddevenMergeSort.vs","./shader/oddevenMergeSort.fs") &&
	bitonicMergeSort_Col.loadFromFile("./shader/bitonicMergeSort.vs","./shader/bitonicMergeSort_Col.fs") &&
	bitonicMergeSort_RowN.loadFromFile("./shader/bitonicMergeSort.vs","./shader/bitonicMergeSort_RowN.fs") &&
	bitonicMergeSort_Row01.loadFromFile("./shader/bitonicMergeSort.vs","./shader/bitonicMergeSort_Row01.fs") &&
	renderShader.loadFromFile("./shader/renderRGBA16f.vs","./shader/renderRGBA16f.fs");
}


void
createDisplaylists()
{
    if (!bitonicSortLists) {

	// the display lists for all stages of the sorting quads to render
	// in normalized screen coords for the packed representation
	int nstages = logFieldsize+logFieldsize;
	int nlists = (nstages*(nstages-1))/2;
	bitonicSortLists = glGenLists(nlists);
	if (bitonicSortLists) {
	    staticdebugmsg("createDisplaylists","creating "<<nlists<<" display lists for bitonic sort");
	    
	    currentBitonicSortList = bitonicSortLists;

	    // the lowest column is replaced by a special shader that combines the last two cols
	    // the first stage has already been computed by the pack (upload) routine
	    int stride = 2;
	    for (int s=2; s<=nstages; ++s) {
// 		staticdebugmsg("log","stage "<<s<<":");
		// stage s has 2^s columns to process (row 0 is internal in special shader)
		for (int c=(s-1); c>0; --c) {
		    
		    glNewList(currentBitonicSortList++,GL_COMPILE);
		    
		    glBegin(GL_QUADS);
		    
		    if (c<logFieldsize) {
			// quads for row sort
			staticdebugmsg("row","[row "<<c<<"] ");
			int rc;
			float m,rx0,rx1,rxt0,rxt1,rv0,rv1;
			for (int r=0; r<(1<<(logFieldsize-1)); r+=(1<<c)) {
// 			    staticdebugmsg("row","quad "<<r);
			    // select type of compare to do
			    if ( ((r>>(c-1))%2) != ((r>>(s-1))%2) ) {
				// do a greater-equal compare
				m = -1.0f;			
			    }
			    else {
				// do a less-than compare
				m = 1.0f;
			    }

			    // sorting parameters
			    // compare the two inputs which index differs only in bit c
			    // i.e. toggle bit c
			    rc = (1<<(c-1));
			
			    // # texture: packed data
			    // # texcoord[0]: texposx texposy search_dir comp_op
			    // # texcoord[1]: texdistx 1/stride stride fragheight
			
			    // displacement (cannot store glProgramLocalParameter4fARB
			    // in dsplist, so use texcoord instead)
			    glMultiTexCoord4fARB(GL_TEXTURE1_ARB,
						 float(rc),
						 float(stride),
						 float(fieldsize),
						 float(stride/2)-0.5f); 
			    rx0 = float(r);
			    rx1 = float(r+(1<<c));
			    rxt0 = rx0;
			    rxt1 = rx1;
			    rv0 = ((rxt0/float(fieldsize))*2.0f)-1.0f;
			    rv1 = ((rxt1/float(fieldsize))*2.0f)-1.0f;
			
			    // always need to flip top-bottom in texcoords
			    glMultiTexCoord4fARB(GL_TEXTURE0_ARB,rxt0,0.0f,1.0f,m);
			    glVertex2f(rv0,-1.0f);	
			    glMultiTexCoord4fARB(GL_TEXTURE0_ARB,rxt1,0.0f,-1.0f,-m);
			    glVertex2f(rv1,-1.0f);	
			    glMultiTexCoord4fARB(GL_TEXTURE0_ARB,rxt1,float(height),-1.0f,-m);
			    glVertex2f(rv1,1.0f);	
			    glMultiTexCoord4fARB(GL_TEXTURE0_ARB,rxt0,float(height),1.0f,m);
			    glVertex2f(rv0,1.0f);	
			    staticdebugmsg("row","stride="<<stride<<" rc="<<rc<<" m="<<m<<" ("<<rx0<<","<<rx1<<") {"<<rxt0<<","<<rxt1<<"} ["<<rv0<<","<<rv1<<"]");
			}
		    
		    }
		    else {

			// quads for column sort
			staticdebugmsg("col","[col "<<c<<"] ");
			int rc;
			float m,ry0,ry1,ryt0,ryt1,rv0,rv1;
			for (int r=0; r<(1<<(logFieldsize)); r+=(1<<(c-logFieldsize+1))) {
// 			    staticdebugmsg("col","quad "<<r);
			    // select type of compare to do
			    if ( ((r>>(c-logFieldsize))%2) != ((r>>(s-logFieldsize))%2) ) {
				// do a greater-equal compare
				m = -1.0f;			
			    }
			    else {
				// do a less-than compare
				m = 1.0f;
			    }

			    // sorting parameters
			    // compare the two inputs which index differs only in bit c
			    // i.e. toggle bit c
			    rc = (1<<(c-1));
			
			    // # texture: packed data
			    // # texcoord[0]: texposx texposy search_dir comp_op
			    // # texcoord[1]: texdisty 1/stride stride fragheight
			
			    // displacement (cannot store glProgramLocalParameter4fARB
			    // in dsplist, so use texcoord instead)
			    glMultiTexCoord4fARB(GL_TEXTURE1_ARB,
						 float(rc>>(logFieldsize-1)),
						 float(stride),
						 float(fieldsize),
						 float(stride/2)-0.5f); 
			    ry0 = float(r);
			    ry1 = float(r+(1<<(c-logFieldsize+1)));
			    ryt0 = ry0;
			    ryt1 = ry1;
			    rv0 = -((1.0f-(ryt0/float(fieldsize)))*2.0f)+1.0f;
			    rv1 = -((1.0f-(ryt1/float(fieldsize)))*2.0f)+1.0f;
			
			    // always need to flip top-bottom in texcoords
			    // quads are only half in width because of packing !
			    glMultiTexCoord4fARB(GL_TEXTURE0_ARB,0.0f,ryt1,-1.0f,-m);
			    glVertex2f(-1.0f,rv1);	
			    glMultiTexCoord4fARB(GL_TEXTURE0_ARB,float(width/2),ryt1,-1.0f,-m);
			    glVertex2f(0.0f,rv1);	
			    glMultiTexCoord4fARB(GL_TEXTURE0_ARB,float(width/2),ryt0,1.0f,m);
			    glVertex2f(0.0f,rv0);	
			    glMultiTexCoord4fARB(GL_TEXTURE0_ARB,0.0f,ryt0,1.0f,m);
			    glVertex2f(-1.0f,rv0);	
//  			    staticdebugmsg("col","stride="<<stride<<" rc="<<(rc>>(logFieldsize-1))<<" m="<<m<<" ("<<ry0<<","<<ry1<<") {"<<ryt0<<","<<ryt1<<"} ["<<rv0<<","<<rv1<<"]");
			}

		    }
		
		    glEnd();

		    glEndList();
		
		} // for all columns

		if (s>=logFieldsize) {
		    stride *= 2;
		}

	    } // for all stages
	
	}
	else
	    staticerrormsg("createDisplaylists","unable to create "<<nlists<<" display lists");

	staticdebugmsg("createDisplaylists","lists alloc "<<nlists<<" lists created "<<currentBitonicSortList-bitonicSortLists);
	staticdebugmsg("createDisplaylists","done creating display lists");
    }
}


void 
myIdle()
{
    glutPostRedisplay();
}


void 
myReshape(int w, int h)
{
    if (h == 0) h = 1;
    
    glViewport(0, 0, w, h);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    
    gluPerspective(60.0, GLfloat(w)/GLfloat(h), 1, 5.0);
}


void 
sortstep(bool odd)
{
    // perform one step of the current sorting algorithm

    // swap buffers
    int sourceBuffer = targetBuffer;
    targetBuffer = (targetBuffer+1)%2;   
    int pstage = (1<<stage);
    int ppass  = (1<<pass);
    int activeBitonicShader = 0;

#ifdef _WIN32
    buffer->BindBuffer(wglTargets[sourceBuffer]);
#else
    buffer->BindBuffer(glTargets[sourceBuffer]);
#endif
    if (buffer->IsDoubleBuffered()) glDrawBuffer(glTargets[targetBuffer]);

    checkGLError("after db");

    // switch on correct sorting shader
    switch (sortAlgorithm) {

        case 1:
            transitionSort.bind();
            glUniform3fARB(transitionSort.getUniformLocation("Param"), float(width),float(height),odd?1.0f:-1.0f);
            glUniform1iARB(transitionSort.getUniformLocation("Data"), 0);
            staticdebugmsg("sort","sorting pass "<<totalSteps-stepsLeft);
            break;

        case 2:
            oddevenMergeSort.bind();
            glUniform3fARB(oddevenMergeSort.getUniformLocation("Param1"), float(pstage+pstage), 
			   float(ppass%pstage), float((pstage+pstage)-(ppass%pstage)-1));
            glUniform3fARB(oddevenMergeSort.getUniformLocation("Param2"), float(width), float(height), float(ppass));
            glUniform1iARB(oddevenMergeSort.getUniformLocation("Data"), 0);
            staticdebugmsg("sort","stage "<<pstage<<" pass "<<ppass);
            break;

        case 3:
            staticdebugmsg("sort","stage "<<stage<<" pass "<<pass);
            checkGLError("before sort");

            if (pass<logFieldsize) {

                // row sort
                if (pass==1) {
                    bitonicMergeSort_Row01.bind();
                    glUniform1iARB(bitonicMergeSort_Row01.getUniformLocation("PackedData"), 0);
                    activeBitonicShader=1;
                    staticdebugmsg("out","[row "<<pass<<",0] ");
                }
                else {
                    bitonicMergeSort_RowN.bind();
                    glUniform1iARB(bitonicMergeSort_RowN.getUniformLocation("PackedData"), 0);
                    activeBitonicShader=2;
                    staticdebugmsg("out","[row "<<pass<<"] ");
                }
                checkGLError("bind bitonic row sort fragment program");


            }
            else {

                // col sort
                bitonicMergeSort_Col.bind();
                glUniform1iARB(bitonicMergeSort_Col.getUniformLocation("PackedData"), 0);
                activeBitonicShader=3;
                checkGLError("bind bitonic col sort fragment program");

                staticdebugmsg("out","[col "<<pass<<"] ");
            }
            break;

        default:
            staticerrormsg("sort","unknown sorting algorithm");
            return;
    }

    // This clear is not necessary for sort to function. But if we are in interactive mode 
    // unused parts of the texture that are visible will look bad.
    if (!perfTest) glClear(GL_COLOR_BUFFER_BIT);

    buffer->Bind();
    buffer->EnableTextureTarget();

    // initiate the sorting step on the GPU
    switch (sortAlgorithm) {

        case 1:
        case 2:
            // a full-screen quad
            glBegin(GL_QUADS);
            glMultiTexCoord4fARB(GL_TEXTURE0_ARB,0.0f,0.0f,0.0f,1.0f);
            glVertex2f(-1.0f,-1.0f);	
            glMultiTexCoord4fARB(GL_TEXTURE0_ARB,float(width),0.0f,1.0f,1.0f);
            glVertex2f(1.0f,-1.0f);	
            glMultiTexCoord4fARB(GL_TEXTURE0_ARB,float(width),float(height),1.0f,0.0f);
            glVertex2f(1.0f,1.0f);	
            glMultiTexCoord4fARB(GL_TEXTURE0_ARB,0.0f,float(height),0.0f,0.0f);
            glVertex2f(-1.0f,1.0f);	
            glEnd();
            break;

        case 3:
            // execute current display list
            glCallList(currentBitonicSortList++);
            checkGLError("call dsplist");
            break;

        default:
            staticerrormsg("sort","unknown sorting algorithm");
            return;
    }

    // switch off sorting shader
    switch (sortAlgorithm) {

        case 1:
            transitionSort.release();
            break;

        case 2:
            oddevenMergeSort.release();
            break;

        case 3:
            if (activeBitonicShader==1) {
                bitonicMergeSort_Row01.release();
            }
            else if (activeBitonicShader==2) {
                bitonicMergeSort_RowN.release();
            }
            else if (activeBitonicShader==3) {
                bitonicMergeSort_Col.release();
            }
            break;

        default:
            staticerrormsg("sort","unknown sorting algorithm");
            return;
    }

    buffer->DisableTextureTarget();

    checkGLError("RT Update");
}


void
checksort()
{
    // Check if we have to do one sorting step in user interactive mode
    // and perform it if so.
    static bool odd = true;

    if (stepsToDo && stepsLeft) {

	// determine next stage,pass,step
	switch (sortAlgorithm) {
    	    
	    case 1:
		odd = !odd;
		break;

	    case 2:
		pass--;
		if (pass<0) {
		    // next stage
		    stage++;
		    pass=stage;
		}
		break;

	    case 3:
		pass--;
		if (pass<1) {
		    // next stage
		    pass=stage;
		    stage++;
		}
		break;
    		
	    default:
		staticerrormsg("sort","unknown sorting algorithm");
		return;
	}

	checkGLError("before sort begin capture");
        if (buffer->IsInitialized() && buffer->BeginCapture(false)) {
            
            sortstep(odd);
            buffer->EndCapture();

	}
        
	stepsToDo--;
	stepsLeft--;
    }
}


void 
myDisplay()
{
    // sorting to do ?
    checksort();

    // draw current buffer on screen so we can see the sort progress
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1, 1, 1);
        
    if (buffer->IsTexture()) {
        if (buffer->IsDoubleBuffered()) 
#ifdef _WIN32
	    buffer->BindBuffer(wglTargets[targetBuffer]);
#else
	buffer->BindBuffer(glTargets[targetBuffer]);
#endif
        buffer->Bind();
    }

    buffer->EnableTextureTarget();

    int maxS = buffer->GetMaxS();
    int maxT = buffer->GetMaxT();

    if (dumpit) {
	// Write current buffer that we see to the console for examination.
	// Note that we only dump the keys.
	std::vector<float> ftexels;
	ftexels.resize(fieldsize*fieldsize*4);
	std::vector<unsigned char> ctexels;
	ctexels.resize(fieldsize*fieldsize*4);
	if (sortFloats)
	    glGetTexImage(buffer->GetTextureTarget(),
			  0,
			  GL_RGBA,
			  GL_FLOAT,
			  &(ftexels[0]));
	else
	    glGetTexImage(buffer->GetTextureTarget(),
			  0,
			  GL_RGBA,
			  GL_UNSIGNED_BYTE,
			  &(ctexels[0]));
    	
	std::cerr<<"target "<<targetBuffer<<": "<<std::endl;
	for (int i=0; i<fieldsize*fieldsize; ++i) {
	    if ((i%fieldsize)==0) std::cerr<<std::endl;
	    if (sortAlgorithm==3) {
		if ((i%fieldsize)<(fieldsize/2))
		    if (sortFloats)
			std::cerr<<"("<<ftexels[i*4]<<" "<<ftexels[i*4+1]<<") ";
		    else
			std::cerr<<"("<<int(ctexels[i*4])<<" "<<int(ctexels[i*4+1])<<") ";
	    }
	    else
		if (sortFloats)
		    std::cerr<<ftexels[i*4]<<" ";
		else
		    std::cerr<<int(ctexels[i*4])<<" ";
	}
	std::cerr<<std::endl;
	dumpit=false;
    }

    if (sortFloats) {
	// cannot draw float tex directly
	renderShader.bind();
	glUniform1fARB(renderShader.getUniformLocation("MaxValue"), 256.0f);
	glUniform1iARB(renderShader.getUniformLocation("Data"), 0);
    }

    glBegin(GL_QUADS);
    glTexCoord2f(0,    maxT); glVertex2f(-1, -1);
    glTexCoord2f(maxS, maxT); glVertex2f( 1, -1);
    glTexCoord2f(maxS,    0); glVertex2f( 1,  1);
    glTexCoord2f(0,       0); glVertex2f(-1,  1);
    glEnd();
    
    if (sortFloats) renderShader.release();

    buffer->DisableTextureTarget();

    glColor4f( 0, 0, 0, 1);
    glRasterPos2f( -1,1.05 );

    char text[256];
    std::sprintf(text,"%s: stage %d pass %d - step %d/%d",sortAlgText[sortAlgorithm],stage,pass,totalSteps-stepsLeft,totalSteps);
    for (unsigned int i=0; i<std::strlen(text); ++i)
	glutBitmapCharacter(GLUT_BITMAP_9_BY_15, text[i]);

    checkGLError("end of myDisplay");
    glutSwapBuffers();
}

void
resetSortAlgorithm()
{
    // reset state variables so the sort can restart anew
    switch (sortAlgorithm) {
	case 1:
	    stepsLeft = totalSteps = width*height;
	    stage = pass = 0;
	    break;
	case 2:
	    stepsLeft = totalSteps = ((logFieldsize+logFieldsize)*(logFieldsize+logFieldsize+1))/2;
	    stage = pass = -1;
	    break;
	case 3:
	    stepsLeft = totalSteps = ((logFieldsize+logFieldsize)*(logFieldsize+logFieldsize-1))/2;
	    stage = 1; pass = 0;
	    currentBitonicSortList = bitonicSortLists;
	    break;
	default:
	    staticerrormsg("resetSortAlgorithm","unknown sort algorithm");
	    break;
    }
    stepsToDo = 0;
}

float 
unitRandom(float seed=0.0f) 
{
    // get a random number in [0;1]
    if ( seed > 0.0f )
	srand((unsigned int)seed);

    return float(rand())/float(RAND_MAX);
}

void
makeData(DataStyle style)
{
    // Create and upload key/index data that is to be sorted.
    // Use already sorted data to see if the sort can run continuously.
    // Use inverse sorted data to see complete reordering.
    // Use random data to see general behaviour.
    
    // Note that we generate key/index pairs that would be used for indexed lookup
    // in the data reorder pass that normally follows the sorting algorithm.
    // In this demo however the indices are not used.

    std::vector<float> fupload;
    fupload.resize(fieldsize*fieldsize*4);
    std::vector<unsigned char> cupload;
    cupload.resize(fieldsize*fieldsize*4);
    staticdebugmsg("makeData","initializing new texture");
   
    int i=0, j=0; float key0, key1;
    float ft;
    int counter, countdir;
    if (style==SORTED_DATA) {
	counter=0; countdir=1;
	staticdebugmsg("makeData","initializing new sorted texture");
    }
    else {
	counter=(fieldsize*fieldsize)-1; countdir=-1;
	staticdebugmsg("makeData","initializing new inverse sorted texture");
    }

    switch (sortAlgorithm) {
	case 1:
	case 2:
	    // odd-even merge sort: one pair per fragment
	    while (i<fieldsize*fieldsize) {
		if (style==RANDOM_DATA)
		    key0 = unitRandom()*255.0f;
		else
		    key0 = float(counter)*256.0f/float(fieldsize*fieldsize);
		counter+=countdir;
		cupload[j  ] = int(key0); // the sorting key
		fupload[j++] = key0; // the sorting key
		cupload[j  ] = int(key0); // store index here - not used by the shader
		fupload[j++] = float(i); // store index here - not used by the shader
		cupload[j  ] = 255;  // not used
		fupload[j++] = 0.0f; // not used
		cupload[j  ] = 255;  // not used
		fupload[j++] = 0.0f; // not used
		i++;
	    }
	    break;
	case 3:
	    // bitonic merge sort: packed two pairs per fragment
	    while (i<fieldsize*fieldsize) {
		if ((i%fieldsize)<(fieldsize/2)) {
		    if (style==RANDOM_DATA) {
			key0 = unitRandom()*255.0f;
			key1 = unitRandom()*255.0f;
		    }
		    else {
			key0 = float(counter)*256.0f/float(fieldsize*fieldsize);
			counter+=countdir;
			key1 = float(counter)*256.0f/float(fieldsize*fieldsize);
			counter+=countdir;
		    }
		    // first sort pass !!!
		    ft=(float(i%2)-0.5f);
		    if ((key0*ft)<(key1*ft)) { float tmp=key0;key0=key1;key1=tmp; }
		    cupload[j  ] = int(key0); // the first sorting key
		    fupload[j++] = key0; // the first sorting key
		    cupload[j  ] = int(key1); // the second sorting key
		    fupload[j++] = key1; // the second sorting key
		    cupload[j  ] = 255; // store index for first key here - not used by shader
		    fupload[j++] = 255.0f; // store index for first key here - not used by shader
		    cupload[j  ] = 255; // store index for second key here - not used by shader
		    fupload[j++] = 255.0f; // store index for second key here - not used by shader
		}
                else {
                    fupload[j  ] = 0.0f;
                    cupload[j++] = 0; // only need half of texture size
                    fupload[j  ] = 0.0f; 
                    cupload[j++] = 0; // this is here only so I don't have to resize
                    fupload[j  ] = 0.0f;
                    cupload[j++] = 0; // you would drop this half in your actual
                    fupload[j  ] = 0.0f;
                    cupload[j++] = 0; // implementation
                }
                i++;
	    }
	    break;
	default:
	    staticerrormsg("makeData","unknown sorting algorithm");
	    return;
    }
	    
    targetBuffer = 1;
#ifdef _WIN32
    buffer->BindBuffer(wglTargets[targetBuffer]);
#else
    buffer->BindBuffer(glTargets[targetBuffer]);
#endif
    buffer->Bind();
    buffer->EnableTextureTarget();
    checkGLError("enable target buffer");

    if (sortFloats) {

	if (GLEW_NV_float_buffer)
	    glTexImage2D(buffer->GetTextureTarget(),
			 0,
			 GL_FLOAT_RGBA16_NV,
			 fieldsize, fieldsize,
			 0,
			 GL_RGBA,
			 GL_FLOAT,
			 &(fupload[0]) );
	else if (GLEW_ATI_texture_float)
	    glTexImage2D(buffer->GetTextureTarget(),
			 0,
			 GL_RGBA_FLOAT16_ATI,
			 fieldsize, fieldsize,
			 0,
			 GL_RGBA,
			 GL_FLOAT,
			 &(fupload[0]) );
	else
	    staticerrormsg("makeData","unknown floating point format");
    }
    else {
	glTexImage2D(buffer->GetTextureTarget(),
		     0,
		     GL_RGBA8,
		     fieldsize, fieldsize,
		     0,
		     GL_RGBA,
		     GL_UNSIGNED_BYTE,
		     &(cupload[0]) );
    }

    checkGLError("upload random texture");

    buffer->DisableTextureTarget();
    checkGLError("disable target buffer");

    resetSortAlgorithm();
}


void
disturb()
{
    // Put some extreme values at 8 random positions in the current
    // texture. If you have already sorted data, this would simulate
    // for example randomly set positions of particles if they are
    // reincarnated. A good sorting algorithm should move them as
    // quickly as possible to the correct positions without messing
    // up the rest of the field while doing so.

    std::vector<float> ftexels;
    ftexels.resize(fieldsize*fieldsize*4);
    std::vector<unsigned char> ctexels;
    ctexels.resize(fieldsize*fieldsize*4);

    if (buffer->IsTexture()) {
        if (buffer->IsDoubleBuffered()) 
#ifdef _WIN32
	    buffer->BindBuffer(wglTargets[targetBuffer]);
#else
	buffer->BindBuffer(glTargets[targetBuffer]);
#endif
        buffer->Bind();
    }

    buffer->EnableTextureTarget();

    if (sortFloats)
	glGetTexImage(buffer->GetTextureTarget(),
		      0,
		      GL_RGBA,
		      GL_FLOAT,
		      &(ftexels[0]));
    else
	glGetTexImage(buffer->GetTextureTarget(),
		      0,
		      GL_RGBA,
		      GL_UNSIGNED_BYTE,
		      &(ctexels[0]));
    checkGLError("download random texture");

    for (int i=0; i<8; ++i) {
	int id = int(unitRandom()*float((fieldsize*fieldsize)-1));
	if (sortAlgorithm==3) if (id%fieldsize >= fieldsize/2) id = (id%fieldsize-fieldsize/2) + (id/fieldsize)*fieldsize;
	staticdebugmsg("disturb","modify "<<id<<" to 0");
	if (sortFloats) { ftexels[id*4] = 0.0f; ftexels[id*4+1] = 64.0f; }
	else { ctexels[id*4] = ctexels[id*4+1] = 0; }
	id = int(unitRandom()*float((fieldsize*fieldsize)-1));
	if (sortAlgorithm==3) if (id%fieldsize >= fieldsize/2) id = (id%fieldsize-fieldsize/2) + (id/fieldsize)*fieldsize;
	staticdebugmsg("disturb","modify "<<id<<" to 255");
	if (sortFloats) { ftexels[id*4] = 255.0f; ftexels[id*4+1] = 64.0f; }
	else { ctexels[id*4] = ctexels[id*4+1] = 255; }
    }

    if (sortFloats) {
	if (GLEW_NV_float_buffer)
	    glTexImage2D(buffer->GetTextureTarget(),
			 0,
			 GL_FLOAT_RGBA16_NV,
			 fieldsize, fieldsize,
			 0,
			 GL_RGBA,
			 GL_FLOAT,
			 &(ftexels[0]) );
	else if (GLEW_ATI_texture_float)
	    glTexImage2D(buffer->GetTextureTarget(),
			 0,
			 GL_RGBA_FLOAT16_ATI,
			 fieldsize, fieldsize,
			 0,
			 GL_RGBA,
			 GL_FLOAT,
			 &(ftexels[0]) );
	else
	    staticerrormsg("makeData","unknown floating point format");
    }
    else {
	glTexImage2D(buffer->GetTextureTarget(),
		     0,
		     GL_RGBA8,
		     fieldsize, fieldsize,
		     0,
		     GL_RGBA,
		     GL_UNSIGNED_BYTE,
		     &(ctexels[0]) );
    }
    checkGLError("upload random texture");

    buffer->DisableTextureTarget();

    resetSortAlgorithm();
}

void
performance()
{
    // do 100 full sorts without user interaction or drawing on screen
    static bool odd = true;

    GbTimer t;
    staticinfomsg("performance","sorting ... please wait");
    perfTest=true;
    t.start();
    for (int i=0; i<100; ++i) {
	resetSortAlgorithm();

	checkGLError("before sort begin capture");
	if (buffer->IsInitialized() && buffer->BeginCapture(false)) {
    
	    stepsLeft = totalSteps;
	    while (stepsLeft) {
		// determine next stage,pass,step
		switch (sortAlgorithm) {
		    case 1:
			odd = !odd;
			break;
		    case 2:
			pass--;
			if (pass<0) {
			    // next stage
			    stage++;
			    pass=stage;
			}
			break;
		    case 3:
			pass--;
			if (pass<1) {
			    // next stage
			    pass=stage;
			    stage++;
			}
			break;
		    default:
			staticerrormsg("sort","unknown sorting algorithm");
			return;
		}
		
		sortstep(odd);            
		stepsLeft--;
		buffer->_MaybeCopyBuffer();
	    }
	    buffer->EndCapture();
	}
    }
    float elapsed = t.stop();
    staticinfomsg("performance",sortAlgText[sortAlgorithm]<<": "<<elapsed<<" sec = "<<float(fieldsize*fieldsize*100)/elapsed<<" key/sec");
    perfTest=false;
}


void 
myKeyboard(unsigned char key, int x, int y)
{
    switch (key) {
	case 27: 
	case 'q':
	    exit(0);
	    break;
	case 's':
	    makeData(SORTED_DATA);
	    break;
	case 'i':
	    makeData(INVERSE_SORTED_DATA);
	    break;
	case 'r':
	    makeData(RANDOM_DATA);
	    break;
	case 'm':
	    disturb();
	    break;
	case 'p':
	    makeData(INVERSE_SORTED_DATA);
	    performance();
	    break;
	case ' ':
	    if (stepsLeft) stepsToDo++;
	    break;
	case 13:
	    stepsToDo = stepsLeft;
	    break;
	case '1':
	    width = height = fieldsize;
	    sortAlgorithm = 1; makeData(INVERSE_SORTED_DATA);
	    break;
	case '2':
	    width = height = fieldsize;
	    sortAlgorithm = 2; makeData(INVERSE_SORTED_DATA);
	    break;
	case '3':
	    width = height = fieldsize;
	    createDisplaylists();
	    sortAlgorithm = 3; makeData(INVERSE_SORTED_DATA);
	    width = fieldsize/2; height = fieldsize;
	    break;
	case 'd':
	    dumpit=true;
	    break;
	default:
	    break;
    }
}


bool
machineCheck()
{
    bool good=true;

    if (!GLEW_ARB_vertex_shader) {
	staticerrormsg("machineCheck","GL_ARB_vertex_shader not supported");
	good=false;
    }
    if (!GLEW_ARB_fragment_shader) {
	staticerrormsg("machineCheck","GL_ARB_fragment_shader not supported");
	good=false;
    }
    if (!GLEW_ARB_shading_language_100) {
	staticerrormsg("machineCheck","GL_ARB_shading_language_100 not supported");
	good=false;
    }
    if (sortFloats && !GLEW_NV_float_buffer && !GLEW_ATI_texture_float) {
	staticerrormsg("machineCheck","floating point textures not supported");
	good=false;
    }
#ifdef _WIN32
    if (!WGLEW_ARB_pbuffer) {
	staticerrormsg("machineCheck","WGL_ARB_pbuffer not supported");
	good=false;
    }
    if (!WGLEW_ARB_pixel_format) {
	staticerrormsg("machineCheck","WGL_ARB_pixel_format not supported");
	good=false;
    }
#else
    if (!GLXEW_SGIX_pbuffer) {
	staticerrormsg("machineCheck","GLX_SGIX_pbuffer not supported");
	good=false;
    }
    if (!GLXEW_SGIX_fbconfig) {
	staticerrormsg("machineCheck","GLX_SGIX_fbconfig not supported");
	good=false;
    }
#endif

    return good;
}


int 
main(int argc, char* argv [])
{
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE);
    glutInitWindowPosition(50, 50);
    glutInitWindowSize(600, 600);
    glutCreateWindow("GPUGems2 Demo: Sorting on the GPU");  
    
    int err = glewInit();
    if (GLEW_OK != err) {
	// problem: glewInit failed, something is seriously wrong
        staticerrormsg("main","cannot initialize OpenGL");
	staticerrormsg("main","GLEW Error: "<<glewGetErrorString(err));
        exit(1);
    }  
    
    glutDisplayFunc(myDisplay);
    glutIdleFunc(myIdle);
    glutReshapeFunc(myReshape);
    glutKeyboardFunc(myKeyboard);
    
    myReshape(512, 512);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 2, 0, 0, 0, 0, 1, 0);
    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glDisable(GL_DEPTH_TEST); 
    glClearColor(0.4, 0.6, 0.8, 1);
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    if (argc>1) {
	logFieldsize = atoi(argv[1]);
	fieldsize = (1<<logFieldsize);
	staticinfomsg("main","sorting "<<fieldsize<<"x"<<fieldsize<<" keys");

	if (argc>2) sortFloats = (atoi(argv[2]) == 0) ? false : true;
    }
    staticinfomsg("main","using "<<(sortFloats?"RGBA16f":"RGBA8")<<" textures");

    if (!machineCheck()) {
        staticerrormsg("main","insufficient system resources");
	exit(2);
    }

    std::string texmode("doublebuffer texRECT ");
    if (sortFloats)
	texmode += "rgba=16f";
    else
	texmode += "rgba";

#ifdef _WIN32
    if (!WGLEW_ARB_render_texture) {
#endif
	texmode += " ctt";
	staticinfomsg("main","render-to-texture not supported by windowing system");
	staticinfomsg("main","--> expect poor performance");
#ifdef _WIN32
    }
#endif
    
    buffer = createRenderTexture(texmode.c_str());
    
    if (buffer == NULL) {
        staticerrormsg("main","unable to create buffers");
	exit(3);
    }

    if (!loadShader()) {
	staticerrormsg("main","unable to load sort shader");
	exit(4);
    }

    staticinfomsg("main","Usage:");
    staticinfomsg("main","<space>\tnext sorting pass");
    staticinfomsg("main","<enter>\tcomplete sort");
    staticinfomsg("main","<1>\tswitch to simple transition sort");
    staticinfomsg("main","<2>\tswitch to odd-even merge sort");
    staticinfomsg("main","<3>\tswitch to bitonic merge sort");
    staticinfomsg("main","<r>\tprovide random data");
    staticinfomsg("main","<s>\tprovide sorted data");
    staticinfomsg("main","<i>\tprovide inverse sorted data");
    staticinfomsg("main","<m>\tmodify some data");
    staticinfomsg("main","<d>\tdump current buffer");
    staticinfomsg("main","<p>\tsorting performance test");
    staticinfomsg("main","<q>\tquit");
    
    glutMainLoop();

    delete buffer;

    return 0;
}
