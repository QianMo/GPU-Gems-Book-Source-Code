// ************************************************
// rand_helpers.h
// authors: Lee Howes and David B. Thomas
//
// Contains support code for the random number
// generation necessary for initialising the 
// cuda simulations correctly.
// 
// Ziggurat code taken from Marsaglia's
// paper.
// ************************************************

#ifndef __rand_helpers_h
#define __rand_helpers_h

// RNG choices
#define USE_ZIGGURAT 1
#define USE_GSL 0


#include <math.h>
#include <assert.h>

#if USE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif


unsigned Kiss()
{
	static unsigned z=362436069, w=521288629, jsr=123456789, jcong=380116160;

	z=36969*(z&65535)+(z>>16);
	w=18000*(w&65535)+(w>>16);
	unsigned mwc=(z<<16)+w;
	jsr^=(jsr<<17);
	jsr^=(jsr>>13);
	jsr^=(jsr<<5);
	jcong=69069*jcong+1234567;
	return (mwc^jcong)+jsr;
}

double Rand()
{
	unsigned long long x=Kiss();
	x=(x<<32)|Kiss();
	return x*5.4210108624275221703311375920553e-20;
}

#if USE_GSL
const gsl_rng_type **t, **t0;
gsl_rng *rng;
bool initialised = false;

void initRand()
{
    gsl_rng_env_setup();

    t0 = gsl_rng_types_setup ();

    printf ("Available generators:\n");

    for (t = t0; *t != 0; t++)
    {
        printf ("%s\n", (*t)->name);
        if( strcmp("mt19937_1999", (*t)->name) == 0 ) break;
    }

    rng = gsl_rng_alloc (*t);
}

double RandN()
{
    return gsl_ran_gaussian_ziggurat  (rng, 1.0);

}
#else

#if USE_ZIGGURAT

/* Period parameters */  
#define CPU_MT_N 624
#define CPU_MT_M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define CPU_MT_UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define CPU_MT_LOWER_MASK 0x7fffffffUL /* least significant r bits */

// MT
unsigned long init[4]={0x123, 0x234, 0x345, 0x456}, length=4;

static unsigned long mt[CPU_MT_N]; /* the array for the state vector  */
static int mti=CPU_MT_N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
void init_genrand(unsigned long s)
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<CPU_MT_N; mti++) {
        mt[mti] = 
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned long init_key[], int key_length)
{
    int i, j, k;
    init_genrand(19650218UL);
    i=1; j=0;
    k = (CPU_MT_N>key_length ? CPU_MT_N : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=CPU_MT_N) { mt[0] = mt[CPU_MT_N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=CPU_MT_N-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=CPU_MT_N) { mt[0] = mt[CPU_MT_N-1]; i=1; }
    }

    mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */ 
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= CPU_MT_N) { /* generate N words at one time */
        int kk;

        if (mti == CPU_MT_N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL); /* a default initial seed is used */

        for (kk=0;kk<CPU_MT_N-CPU_MT_M;kk++) {
            y = (mt[kk]&CPU_MT_UPPER_MASK)|(mt[kk+1]&CPU_MT_LOWER_MASK);
            mt[kk] = mt[kk+CPU_MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<CPU_MT_N-1;kk++) {
            y = (mt[kk]&CPU_MT_UPPER_MASK)|(mt[kk+1]&CPU_MT_LOWER_MASK);
            mt[kk] = mt[kk+(CPU_MT_M-CPU_MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[CPU_MT_N-1]&CPU_MT_UPPER_MASK)|(mt[0]&CPU_MT_LOWER_MASK);
        mt[CPU_MT_N-1] = mt[CPU_MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }
  
    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

double genrand_real2(void)
{
    return genrand_int32()*(1.0/4294967296.0); 
    /* divided by 2^32 */
}

double genrand_real3(void)
{
    return (((double)genrand_int32()) + 0.5)*(1.0/4294967296.0); 
    /* divided by 2^32 */
}


// ZIGGURAT
#define abs(X) abs((int)X)
//#define SHR3 (jz=jsr, jsr^=(jsr<<13), jsr^=(jsr>>17), jsr^=(jsr<<5),jz+jsr)
#define SHR3 genrand_int32()
//#define UNI (.5 + (signed) SHR3 * .2328306e-9)
#define UNI genrand_real3()  
#define RandN() (hz=SHR3, iz=hz&127, (abs(hz)<kn[iz])? hz*wn[iz] : nfix())

static unsigned long iz,jz,jsr=123456789,kn[128],ke[256];
static long hz; static float wn[128],fn[128], we[256],fe[256];

float nfix(void) { /*provides RNOR if #define cannot */
    const float r = 3.442620f; static float x, y;
    for(;;){ x=hz*wn[iz];
        if(iz==0){ 
            do{
                x=-log(UNI)*0.2904764; 
                y=-log(UNI);
            } while(y+y<x*x);
            return (hz>0)? r+x : -r-x;
        }
        if( fn[iz]+UNI*(fn[iz-1]-fn[iz]) < exp(-.5*x*x) ) return x;
        hz=SHR3; iz=hz&127;if(abs(hz)<kn[iz]) return (hz*wn[iz]);
    } 
}


/*--------This procedure sets the seed and creates the tables------*/
void initRand() {
    unsigned long jsrseed = 123456789;
    const double m1 = 2147483648.0, m2 = 4294967296.;
    double dn=3.442619855899,tn=dn,vn=9.91256303526217e-3, q;
    double de=7.697117470131487, te=de, ve=3.949659822581572e-3;
    int i; jsr=jsrseed;

    /* Tables for RNOR: */ q=vn/exp(-.5*dn*dn);
    kn[0]=(dn/q)*m1; kn[1]=0;
    wn[0]=q/m1; wn[127]=dn/m1;
    fn[0]=1.; fn[127]=exp(-.5*dn*dn);
    for(i=126;i>=1;i--) {
        dn=sqrt(-2.*log(vn/dn+exp(-.5*dn*dn)));
        kn[i+1]=(dn/tn)*m1; tn=dn;
        fn[i]=exp(-.5*dn*dn); wn[i]=dn/m1; 
    }

    /* Tables for REXP */ q = ve/exp(-de);
    ke[0]=(de/q)*m2; ke[1]=0;
    we[0]=q/m2; we[255]=de/m2;
    fe[0]=1.; fe[255]=exp(-de);
    for(i=254;i>=1;i--) {
        de=-log(ve/de+exp(-de));
        ke[i+1]= (de/te)*m2; te=de;
        fe[i]=exp(-de); we[i]=de/m2;
    }

    init_by_array(init, length);
}







#else

void initRand()
{

}

double RandN()
{
	static bool cached=false;
	static double cn;

	if(cached){
		cached=false;
		return cn;
	}

	double a=sqrt(-2*log(Rand()));
	double b=6.283185307179586476925286766559*Rand();
	cn=sin(b)*a;
	cached=true;
	return cos(b)*a;
}
#endif // USE_ZIGGURAT
#endif // USE_GSL

double MakeChi2Scale(unsigned N)
{
	const double chic1 = sqrt ( sqrt (1.0 - 1.0 / N));
	const double chic2 = sqrt (1.0 - chic1 * chic1);
	return chic1+chic2*RandN();
}

#endif
