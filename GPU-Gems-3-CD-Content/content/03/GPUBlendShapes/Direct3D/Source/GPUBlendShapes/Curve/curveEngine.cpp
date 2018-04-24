/**
    file: curveEngine.h
    Copyright (c) NVIDIA Corporation. All rights reserved.
 **/
#pragma warning(disable: 4244)//conversion from 'double' to 'float', possible loss of data

#include "curveEngine.h"

/*
 statics for sMachineTolerance computation
*/
float Curve::sMachineTolerance;

Curve::Curve()
{
	m_inputistime = false;
	m_numKeys = 0;
	m_isWeighted = false;
	m_isStatic = false;
	m_preInfinity = kInfinityConstant;
	m_postInfinity = kInfinityConstant;
	// evaluate cache
	m_lastKey = NULL;
	m_lastIndex = 0;
	m_lastInterval = 0;
	m_isStep = false;
	m_isLinear = false;
	m_fX1 = 0;
	m_fX4 = 0;
	//m_fCoeff[4];
	//m_fPolyY[4];

}
Curve::~Curve()
{
}

/****************************************************************************/ /**
		Init the machine tolerance, sMachineTolerance, is defined to be the
		double that satisfies:

		- (1)  sMachineTolerance = 2^(k), for some integer k;
		- (2*) (1.0 + sMachineTolerance > 1.0) is TRUE;
		- (3*) (1.0 + sMachineTolerance/2.0 == 1.0) is TRUE.
		- (*)  When NO floating point optimization is used.
		.

		To foil floating point optimizers, sMachineTolerance must be
		computed using dbl_mult(), dbl_add() and dbl_gt().
*/
void
Curve::init_tolerance ()
{
    float one, half, sum;
    one  = 1.0;
    half = 0.5;
    sMachineTolerance = 1.0;
    do {
		dbl_mult (&sMachineTolerance, &half, &sMachineTolerance);
		dbl_add (&sMachineTolerance, &one, &sum);
    } while (dbl_gt (&sum, &one));
	sMachineTolerance = 2.0 * sMachineTolerance;
}

/****************************************************************************/ /**
		We want to ensure that (x1, x2) is inside the ellipse
		(x1^2 + x2^2 - 2(x1 +x2) + x1*x2 + 1) given that we know
		x1 is within the x bounds of the ellipse.
*/
void
Curve::constrainInsideBounds (float *x1, float *x2)
{
	float b, c,  discr,  root;

	if ((*x1 + sMachineTolerance) < kFourThirds) {
		b = *x1 - 2.0;
		c = *x1 - 1.0;
		discr = sqrt (b * b - 4 * c * c);
		root = (-b + discr) * 0.5;
		if ((*x2 + sMachineTolerance) > root) {
			*x2 = root - sMachineTolerance;
		}
		else {
			root = (-b - discr) * 0.5;
			if (*x2 < (root + sMachineTolerance)) {
				*x2 = root + sMachineTolerance;
			}
		}
	}
	else {
		*x1 = kFourThirds - sMachineTolerance;
		*x2 = kOneThird - sMachineTolerance;
	}
}

/****************************************************************************/ /**

		Given the bezier curve

			 B(t) = [t^3 t^2 t 1] * | -1  3 -3  1 | * | 0  |
									|  3 -6  3  0 |   | x1 |
									| -3  3  0  0 |   | x2 |
									|  1  0  0  0 |   | 1  |

		We want to ensure that the B(t) is a monotonically increasing function.
		We can do this by computing

			 B'(t) = [3t^2 2t 1 0] * | -1  3 -3  1 | * | 0  |
									 |  3 -6  3  0 |   | x1 |
									 | -3  3  0  0 |   | x2 |
									 |  1  0  0  0 |   | 1  |

		and finding the roots where B'(t) = 0.  If there is at most one root
		in the interval [0, 1], then the curve B(t) is monotonically increasing.

		It is easier if we use the control vector [ 0 x1 (1-x2) 1 ] since
		this provides more symmetry, yields better equations and constrains
		x1 and x2 to be positive.

		Therefore:

			 B'(t) = [3t^2 2t 1 0] * | -1  3 -3  1 | * | 0    |
									 |  3 -6  3  0 |   | x1   |
									 | -3  3  0  0 |   | 1-x2 |
									 |  1  0  0  0 |   | 1    |

				   = [t^2 t 1 0] * | 3*(3*x1 + 3*x2 - 2)  |
								   | 2*(-6*x1 - 3*x2 + 3) |
								   | 3*x1                 |
								   | 0                    |

		gives t = (2*x1 + x2 -1) +/- sqrt(x1^2 + x2^2 + x1*x2 - 2*(x1 + x2) + 1)
				  --------------------------------------------------------------
								3*x1 + 3* x2 - 2

		If the ellipse [x1^2 + x2^2 + x1*x2 - 2*(x1 + x2) + 1] <= 0, (Note
		the symmetry) x1 and x2 are valid control values and the curve is
		monotonic.  Otherwise, x1 and x2 are invalid and have to be projected
		onto the ellipse.

		It happens that the maximum value that x1 or x2 can be is 4/3.
		If one of the values is less than 4/3, we can determine the
		boundary constraints for the other value.
*/
void
Curve::checkMonotonic (float *x1, float *x2)
{
	float d;

	/*
	 We want a control vector of [ 0 x1 (1-x2) 1 ] since this provides
	 more symmetry. (This yields better equations and constrains x1 and x2
	 to be positive.)
	*/
	*x2 = 1.0 - *x2;

	/* x1 and x2 must always be positive */
	if (*x1 < 0.0) *x1 = 0.0;
	if (*x2 < 0.0) *x2 = 0.0;

	/*
	 If x1 or x2 are greater than 1.0, then they must be inside the
	 ellipse (x1^2 + x2^2 - 2(x1 +x2) + x1*x2 + 1).
	 x1 and x2 are invalid if x1^2 + x2^2 - 2(x1 +x2) + x1*x2 + 1 > 0.0
	*/
	if ((*x1 > 1.0) || (*x2 > 1.0)) {
		d = *x1 * (*x1 - 2.0 + *x2) + *x2 * (*x2 - 2.0) + 1.0;
		if ((d + sMachineTolerance) > 0.0) {
			constrainInsideBounds (x1, x2);
		}
	}

	*x2 = 1.0 - *x2;
}

/****************************************************************************/ /**
		Convert the control values for a polynomial defined in the Bezier
		basis to a polynomial defined in the power basis (t^3 t^2 t 1).
*/
void
Curve::bezierToPower (
	float a1, float b1, float c1, float d1,
	float *a2, float *b2, float *c2, float *d2)
{
	float a = b1 - a1;
	float b = c1 - b1;
	float c = d1 - c1;
	float d = b - a;
	*a2 = c - b - d;
	*b2 = d + d + d;
	*c2 = a + a + a;
	*d2 = a1;
}

/*
   Evaluate a polynomial in array form ( value only )
   input:

      - P               array 
      - deg             degree
      - s               parameter
		.

   output:

      - ag_horner1      evaluated polynomial
		.

   process: 

      - ans = sum (i from 0 to deg) of P[i]*s^i
	.

   restrictions: 

      - deg >= 0           
	.
*/
float
Curve::ag_horner1 (float P[], int deg, float s)
{
	float h = P[deg];
	while (--deg >= 0) h = (s * h) + P[deg];
	return (h);
}

typedef struct ag_polynomial {
	float *p;
	int deg;
} AG_POLYNOMIAL;

/*
   Compute parameter value at zero of a function between limits
       with function values at limits

   input:
       - a, b      real interval
       - fa, fb    double values of f at a, b
       - f         real valued function of t and pars
       - tol       tolerance
       - pars      pointer to a structure
		.

   output:
       - ag_zeroin2   a <= zero of function <= b
		.

   process:

       We find the zeroes of the function f(t, pars).  t is
       restricted to the interval [a, b].  pars is passed in as
       a pointer to a structure which contains parameters
       for the function f.

   restrictions:

       fa and fb are of opposite sign.
       Note that since pars comes at the end of both the
       call to ag_zeroin and to f, it is an optional parameter.
*/
float
Curve::ag_zeroin2 (float a, float b, float fa, float fb, float tol, AG_POLYNOMIAL *pars)
{
	int test;
	float c, d, e, fc, del, m, machtol, p, q, r, s;

	/* initialization */
	machtol = sMachineTolerance;

	/* start iteration */
label1:
	c = a;  fc = fa;  d = b-a;  e = d;
label2:
	if (fabs(fc) < fabs(fb)) {
		a = b;   b = c;   c = a;   fa = fb;   fb = fc;   fc = fa;
	}

	/* convergence test */
	del = 2.0 * machtol * fabs(b) + 0.5*tol;
	m = 0.5 * (c - b);
	test = ((fabs(m) > del) && (fb != 0.0));
	if (test) {
		if ((fabs(e) < del) || (fabs(fa) <= fabs(fb))) {
			/* bisection */
			d = m;  e= d;
		}
		else {
			s = fb / fa;
			if (a == c) {
				/* linear interpolation */
				p = 2.0*m*s;    q = 1.0 - s;
			}
			else {
				/* inverse quadratic interpolation */
				q = fa/fc;
				r = fb/fc;
				p = s*(2.0*m*q*(q-r)-(b-a)*(r-1.0));
				q = (q-1.0)*(r-1.0)*(s-1.0);
			}
			/* adjust the sign */
			if (p > 0.0) q = -q;  else p = -p;
			/* check if interpolation is acceptable */
			s = e;   e = d;
			if ((2.0*p < 3.0*m*q-fabs(del*q))&&(p < fabs(0.5*s*q))) {
				d = p/q;
			}
			else {
				d = m;	e = d;
			}
		}
		/* complete step */
		a = b;	fa = fb;
		if ( fabs(d) > del )   b += d;
		else if (m > 0.0) b += del;  else b -= del;
		fb = ag_horner1 (pars->p, pars->deg, b);
		if (fb*(fc/fabs(fc)) > 0.0 ) {
			goto label1;
		}
		else {
			goto label2;
		}
	}
	return (b);
}

/****************************************************************************/ /**
   Compute parameter value at zero of a function between limits

   input:

       - a, b            real interval
       - f               real valued function of t and pars
       - tol             tolerance
       - pars            pointer to a structure
		.

   output:

       - ag_zeroin       zero of function
		.

   process:

       Call ag_zeroin2 to find the zeroes of the function f(t, pars).
       t is restricted to the interval [a, b].
       pars is passed in as a pointer to a structure which contains
       parameters for the function f.

   restrictions:

       f(a) and f(b) are of opposite sign.
       Note that since pars comes at the end of both the
         call to ag_zeroin and to f, it is an optional parameter.
       If you already have values for fa,fb use ag_zeroin2 directly
*/
float
Curve::ag_zeroin (float a, float b, float tol, AG_POLYNOMIAL *pars)
{
	float fa, fb;

	fa = ag_horner1 (pars->p, pars->deg, a);
	if (fabs(fa) < sMachineTolerance) return(a);

	fb = ag_horner1 (pars->p, pars->deg, b);
	if (fabs(fb) < sMachineTolerance) return(b);

	return (ag_zeroin2 (a, b, fa, fb, tol, pars));
} 

/****************************************************************************/ /**
   Find the zeros of a polynomial function on an interval
   input:

       - Poly                 array of coefficients of polynomial
       - deg                  degree of polynomial
       - a, b                 interval of definition a < b
       - a_closed             include a in interval (TRUE or FALSE)
       - b_closed             include b in interval (TRUE or FALSE)
		.

   output: 

       - polyzero             number of roots 
                            -1 indicates Poly == 0.0
       - Roots                zeroes of the polynomial on the interval
		.

   process:

       Find all zeroes of the function on the open interval by 
       recursively finding all of the zeroes of the derivative
       to isolate the zeroes of the function.  Return all of the 
       zeroes found adding the end points if the corresponding side
       of the interval is closed and the value of the function 
       is indeed 0 there.

   restrictions:

       The polynomial p is simply an array of deg+1 doubles.
       p[0] is the constant term and p[deg] is the coef 
       of t^deg.
       The array roots should be dimensioned to deg+2. If the number
       of roots returned is greater than deg, suspect numerical
       instabilities caused by some nearly flat portion of Poly.
*/
int
Curve::polyZeroes (float Poly[], int deg, float a, int a_closed, float b, int b_closed, float Roots[])
{
	int i, left_ok, right_ok, nr, ndr, skip;
	float e, f, s, pe, ps, tol, *p, p_x[22], *d, d_x[22], *dr, dr_x[22];
	AG_POLYNOMIAL ply;

	e = pe = 0.0;  
	f = 0.0;

	for (i = 0 ; i < deg + 1; ++i) {
		f += fabs(Poly[i]);
	}
	tol = (fabs(a) + fabs(b))*(deg+1)*sMachineTolerance;

	/* Zero polynomial to tolerance? */
	if (f <= tol)  return(-1);

	p = p_x;  d = d_x;  dr = dr_x;
	for (i = 0 ; i < deg + 1; ++i) {
		p[i] = 1.0/f * Poly[i];
	}

	/* determine true degree */
	while ( fabs(p[deg]) < tol) deg--;

	/* Identically zero poly already caught so constant fn != 0 */
	nr = 0;
	if (deg == 0) return (nr);

	/* check for linear case */
	if (deg == 1) {
		Roots[0] = -p[0] / p[1];
		left_ok  = (a_closed) ? (a<Roots[0]+tol) : (a<Roots[0]-tol);
		right_ok = (b_closed) ? (b>Roots[0]-tol) : (b>Roots[0]+tol);
		nr = (left_ok && right_ok) ? 1 : 0;
		if (nr) {
			if (a_closed && Roots[0]<a) Roots[0] = a;
			else if (b_closed && Roots[0]>b) Roots[0] = b;
		}
		return (nr);
	}
	/* handle non-linear case */
	else {
		ply.p = p;  ply.deg = deg;

		/* compute derivative */
		for (i=1; i<=deg; i++) d[i-1] = i*p[i];

		/* find roots of derivative */
		ndr = polyZeroes ( d, deg-1, a, 0, b, 0, dr );
		if (ndr == -1) return (0);

		/* find roots between roots of the derivative */
		for (i=skip=0; i<=ndr; i++) {
			if (nr>deg) return (nr);
			if (i==0) {
				s=a; ps = ag_horner1( p, deg, s);
				if ( fabs(ps)<=tol && a_closed) Roots[nr++]=a;
			}
			else { s=e; ps=pe; }
			if (i==ndr) { e = b; skip = 0;}
			else e=dr[i];
			pe = ag_horner1( p, deg, e );
			if (skip) skip = 0;
			else {
				if ( fabs(pe) < tol ) {
					if (i!=ndr || b_closed) {
						Roots[nr++] = e;
						skip = 1;
					}
				}
				else if ((ps<0 && pe>0)||(ps>0 && pe<0)) {
					Roots[nr++] = ag_zeroin(s, e, 0.0, &ply );
					if ((nr>1) && Roots[nr-2]>=Roots[nr-1]-tol) { 
						Roots[nr-2] = (Roots[nr-2]+Roots[nr-1]) * 0.5;
						nr--;
					}
				}
			}
		}
	}

	return (nr);
} 

/****************************************************************************/ /**
		Create a constrained single span cubic 2d bezier curve using the
		specified control points.  The curve interpolates the first and
		last control point.  The internal two control points may be
		adjusted to ensure that the curve is monotonic.
*/
void
Curve::engineBezierCreate (float x[4], float y[4])
{
	static bool sInited = false;
	float rangeX, dx1, dx2, nX1, nX2, oldX1, oldX2;

	if (!sInited) {
		init_tolerance ();
		sInited = true;
	}

	rangeX = x[3] - x[0];
	if (rangeX == 0.0) {
		return;
	}
	dx1 = x[1] - x[0];
	dx2 = x[2] - x[0];

	/* normalize X control values */
	nX1 = dx1 / rangeX;
	nX2 = dx2 / rangeX;

	/* if all 4 CVs equally spaced, polynomial will be linear */
	if ((nX1 == kOneThird) && (nX2 == kTwoThirds)) {
		m_isLinear = true;
	} else {
		m_isLinear = false;
	}

	/* save the orig normalized control values */
	oldX1 = nX1;
	oldX2 = nX2;

	/*
	 check the inside control values yield a monotonic function.
	 if they don't correct them with preference given to one of them.
	
	 Most of the time we are monotonic, so do some simple checks first
	*/
	if (nX1 < 0.0) nX1 = 0.0;
	if (nX2 > 1.0) nX2 = 1.0;
	if ((nX1 > 1.0) || (nX2 < -1.0)) {
		checkMonotonic (&nX1, &nX2);
	}

	/* compute the new control points */
	if (nX1 != oldX1) {
		x[1] = x[0] + nX1 * rangeX;
		if (oldX1 != 0.0) {
			y[1] = y[0] + (y[1] - y[0]) * nX1 / oldX1;
		}
	}
	if (nX2 != oldX2) {
		x[2] = x[0] + nX2 * rangeX;
		if (oldX2 != 1.0) {
			y[2] = y[3] - (y[3] - y[2]) * (1.0 - nX2) / (1.0 - oldX2);
		}
	}

	/* save the control points */
	m_fX1 = x[0];
	m_fX4 = x[3];

	/* convert Bezier basis to power basis */
	bezierToPower (
		0.0, nX1, nX2, 1.0,
		&(m_fCoeff[3]), &(m_fCoeff[2]), &(m_fCoeff[1]), &(m_fCoeff[0])
	);
	bezierToPower (
		y[0], y[1], y[2], y[3],
		&(m_fPolyY[3]), &(m_fPolyY[2]), &(m_fPolyY[1]), &(m_fPolyY[0])
	);
}

/****************************************************************************/ /**
		Given the time between fX1 and fX4, return the
		value of the curve at that time.
*/
float
Curve::engineBezierEvaluate (float time)
{
	float t, s, poly[4], roots[5];
	int numRoots;


	if (m_fX1 == time) {
		s = 0.0;
	}
	else if (m_fX4 == time) {
		s = 1.0;
	}
	else {
		s = (time - m_fX1) / (m_fX4 - m_fX1);
	}

	if (m_isLinear) {
		t = s;
	}
	else {
		poly[3] = m_fCoeff[3];
		poly[2] = m_fCoeff[2];
		poly[1] = m_fCoeff[1];
		poly[0] = m_fCoeff[0] - s;

		numRoots = polyZeroes (poly, 3, 0.0, 1, 1.0, 1, roots);
		if (numRoots == 1) {
			t = roots[0];
		}
		else {
			t = 0.0;
		}
	}
	return (t * (t * (t * m_fPolyY[3] + m_fPolyY[2]) + m_fPolyY[1]) + m_fPolyY[0]);
}

void
Curve::engineHermiteCreate (float x[4], float y[4])
{
	float dx, dy, tan_x, m1, m2, length, d1, d2;

	/* save the control points */
	m_fX1 = x[0];

	/*	
	 *	Compute the difference between the 2 keyframes.					
	 */
	dx = x[3] - x[0];
	dy = y[3] - y[0];

	/* 
	 * 	Compute the tangent at the start of the curve segment.			
	 */
	tan_x = x[1] - x[0];
	m1 = m2 = (float)kMaxTan;
	if (tan_x != 0.0) {
		m1 = (y[1] - y[0]) / tan_x;
	}

	tan_x = x[3] - x[2];
	if (tan_x != 0.0) {
		m2 = (y[3] - y[2]) / tan_x;
	}

	length = 1.0 / (dx * dx);
	d1 = dx * m1;
	d2 = dx * m2;
	m_fCoeff[0] = (d1 + d2 - dy - dy) * length / dx;
	m_fCoeff[1] = (dy + dy + dy - d1 - d1 - d2) * length;
	m_fCoeff[2] = m1;
	m_fCoeff[3] = y[0];
}

/****************************************************************************/ /**
		Given the time between fX1 and fX2, return the function
		value of the curve
*/
float
Curve::engineHermiteEvaluate (float time)
{
	float t;
	t = time - m_fX1;
	return (t * (t * (t * m_fCoeff[0] + m_fCoeff[1]) + m_fCoeff[2]) + m_fCoeff[3]);
}

/****************************************************************************/ /**
		A static helper function to evaluate the infinity portion of an
	animation curve.  The infinity portion is the parts of the animation
	curve outside the range of keys.

  Input Arguments:

		- float time					The time (in seconds) to evaluate
		- EtBoolean evalPre
			- true				evaluate the pre-infinity portion
			- false			evaluate the post-infinity portion
		.

  Return Value:

		float value				The evaluated value of the curve at time
*/
float
Curve::evaluateInfinities (float time, bool evalPre)
{
	float value = 0.0;
	float	valueRange;
	float	factoredTime, firstTime, lastTime, timeRange;
	float	remainder, tanX, tanY;
	double numCycles, notUsed;

	/* find the number of cycles of the base animation curve */
	firstTime = m_keyList[0].time;
	lastTime = m_keyList[m_numKeys - 1].time;
	timeRange = lastTime - firstTime;
	if (timeRange == 0.0) {
		/*
		 Means that there is only one key in the curve.. Return the value
		 of that key..
		*/
		return (m_keyList[0].value);
	}
	if (time > lastTime) {
		remainder = fabs (modf ((time - lastTime) / timeRange, &numCycles));
	}
	else {
		remainder = fabs (modf ((time - firstTime) / timeRange, &numCycles));
	}
	factoredTime = timeRange * remainder;
	numCycles = fabs (numCycles) + 1;

	if (evalPre) {
		/* evaluate the pre-infinity */
		if (m_preInfinity == kInfinityOscillate) {
			if ((remainder = modf (numCycles / 2.0, &notUsed)) != 0.0) {
				factoredTime = firstTime + factoredTime;
			}
			else {
				factoredTime = lastTime - factoredTime;
			}
		}
		else if ((m_preInfinity == kInfinityCycle)
		||	(m_preInfinity == kInfinityCycleRelative)) {
			factoredTime = lastTime - factoredTime;
		}
		else if (m_preInfinity == kInfinityLinear) {
			factoredTime = firstTime - time;
			tanX = m_keyList[0].inTanX;
			tanY = m_keyList[0].inTanY;
			value = m_keyList[0].value;
			if (tanX != 0.0) {
				value -= ((factoredTime * tanY) / tanX);
			}
			return (value);
		}
	}
	else {
		/* evaluate the post-infinity */
		if (m_postInfinity == kInfinityOscillate) {
			if ((remainder = modf (numCycles / 2.0, &notUsed)) != 0.0) {
				factoredTime = lastTime - factoredTime;
			}
			else {
				factoredTime = firstTime + factoredTime;
			}
		}
		else if ((m_postInfinity == kInfinityCycle)
		||	(m_postInfinity == kInfinityCycleRelative)) {
			factoredTime = firstTime + factoredTime;
		}
		else if (m_postInfinity == kInfinityLinear) {
			factoredTime = time - lastTime;
			tanX = m_keyList[m_numKeys - 1].outTanX;
			tanY = m_keyList[m_numKeys - 1].outTanY;
			value = m_keyList[m_numKeys - 1].value;
			if (tanX != 0.0) {
				value += ((factoredTime * tanY) / tanX);
			}
			return (value);
		}
	}

	value = evaluate(factoredTime);

	/* Modify the value if infinityType is cycleRelative */
	if (evalPre && (m_preInfinity == kInfinityCycleRelative)) {
		valueRange = m_keyList[m_numKeys - 1].value -
						m_keyList[0].value;
		value -= (numCycles * valueRange);
	}
	else if (!evalPre && (m_postInfinity == kInfinityCycleRelative)) {
		valueRange = m_keyList[m_numKeys - 1].value -
						m_keyList[0].value;
		value += (numCycles * valueRange);
	}
	return (value);
}

/****************************************************************************/ /**
		A static helper method to find a key prior to a specified time

  Input Arguments:

		- float time					The time (in seconds) to find
		- int *index				The index of the key prior to time

  Return Value:

	- EtBoolean result
		- true				time is represented by an actual key
										(with the index in index)
		- false			the index key is the key less than time
	.

	Note:

		keys are sorted by ascending time, which means we can use a binary
	search to find the key
*/
bool
Curve::find (float time, int *index)
{
	int len, mid, low, high;

	/* make sure we have something to search */
	if ((index == NULL)) {
		return (false);
	}

	/* use a binary search to find the key */
	*index = 0;
	len = m_numKeys;
	if (len > 0) {
		low = 0;
		high = len - 1;
		do {
			mid = (low + high) >> 1;
			if (time < m_keyList[mid].time) {
				high = mid - 1;			/* Search lower half */
			} else if (time > m_keyList[mid].time) {
				low  = mid + 1;			/* Search upper half */
			}
			else {
				*index = mid;	/* Found item! */
				return (true);
			}
		} while (low <= high);
		*index = low;
	}
	return (false);
}

/****************************************************************************/ /**
		A function to evaluate an animation curve at a specified time

  Input Arguments:

		- float time					The time (in seconds) to evaluate
		.

  Return Value:

		- float value				The evaluated value of the curve at time
		.
*/
float
Curve::evaluate(float time)
{
	bool withinInterval = false;
	Key *nextKey;
	int index = 0;
	float value = 0.0;
	float x[4];
	float y[4];

	/* make sure we have something to evaluate */
	if ((m_numKeys == 0)) {
		return (value);
	}

	/* check if the time falls into the pre-infinity */
	if (time < m_keyList[0].time) {
		if (m_preInfinity == kInfinityConstant) {
			return (m_keyList[0].value);
		}
		return (evaluateInfinities (time, true));
	}

	/* check if the time falls into the post-infinity */
	if (time > m_keyList[m_numKeys - 1].time) {
		if (m_postInfinity == kInfinityConstant) {
			return (m_keyList[m_numKeys - 1].value);
		}
		return (evaluateInfinities (time, false));
	}

	/* check if the animation curve is static */
	if (m_isStatic) {
		return (m_keyList[0].value);
	}

	/* check to see if the time falls within the last segment we evaluated */
	if (m_lastKey != NULL) {
		if ((m_lastIndex < (m_numKeys - 1))
		&&	(time > m_lastKey->time)) {
			nextKey = &(m_keyList[m_lastIndex + 1]);
			if (time == nextKey->time) {
				m_lastKey = nextKey;
				m_lastIndex++;
				return (m_lastKey->value);
			}
			if (time < nextKey->time ) {
				index = m_lastIndex + 1;
				withinInterval = true;
			}
		}
		else if ((m_lastIndex > 0)
			&&	(time < m_lastKey->time)) {
			nextKey = &(m_keyList[m_lastIndex - 1]);
			if (time > nextKey->time) {
				index = m_lastIndex;
				withinInterval = true;
			}
			if (time == nextKey->time) {
				m_lastKey = nextKey;
				m_lastIndex--;
				return (m_lastKey->value);
			}
		}
	}

	/* it does not, so find the new segment */
	if (!withinInterval) {
		if (find (time, &index) || (index == 0)) {
			/*
				Exact match or before range of this action,
				return exact keyframe value.
			*/
			m_lastKey = &(m_keyList[index]);
			m_lastIndex = index;
			return (m_lastKey->value);
		}
		else if (index == m_numKeys) {
			/* Beyond range of this action return end keyframe value */
			m_lastKey = &(m_keyList[0]);
			m_lastIndex = 0;
			return (m_keyList[m_numKeys - 1].value);
		}
	}

	/* if we are in a new segment, pre-compute and cache the bezier parameters */
	if (m_lastInterval != (index - 1)) {
		m_lastInterval = index - 1;
		m_lastIndex = m_lastInterval;
		m_lastKey = &(m_keyList[m_lastInterval]);
		if ((m_lastKey->outTanX == 0.0)
		&&	(m_lastKey->outTanY == 0.0)) {
			m_isStep = true;
		}
		else {
			m_isStep = false;
			x[0] = m_lastKey->time;
			y[0] = m_lastKey->value;
			x[1] = x[0] + (m_lastKey->outTanX * kOneThird);
			y[1] = y[0] + (m_lastKey->outTanY * kOneThird);

			nextKey = &(m_keyList[index]);
			x[3] = nextKey->time;
			y[3] = nextKey->value;
			x[2] = x[3] - (nextKey->inTanX * kOneThird);
			y[2] = y[3] - (nextKey->inTanY * kOneThird);

			if (m_isWeighted) {
				engineBezierCreate (x, y);
			}
			else {
				engineHermiteCreate (x, y);
			}
		}
	}

	/* finally we can evaluate the segment */
	if (m_isStep) {
		value = m_lastKey->value;
	}
	else if (m_isWeighted) {
		value = engineBezierEvaluate (time);
	}
	else {
		value = engineHermiteEvaluate (time);
	}
	return (value);
}
