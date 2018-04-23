#include "nvafx.h"

//  this is a direct evaluation of the spherical harmonic basis functions for
//  lambertian diffuse, taken from Hanrahan & Ramamoorthi. "An Efficient 
//  Representation for Irradiance Environment Maps" (SIGGRAPH 2001)
double Lambert_Al_Evaluator::operator ()( int l ) const
{
    if (l<0)       // bogus case
        return 0.;

    if ( (l&1)==1 )
    {
        if (l==1) 
            return 2.*M_PI / 3.;
        else 
            return 0.;
    }
    else  // l is even
    {
        double l_fac = factorial((unsigned int)l);
        double l_over2_fac = factorial((unsigned int)(l>>1));
        double denominator = (l+2)*(l-1);
        double sign = ( (l>>1) & 1 )?1.f : -1.f;  // -1^(l/2 - 1) = -1 when l is a multiple of 4, 1 for other multiples of 2
        double exp2_l = (1 << (unsigned int)l);
        return (sign*2.*M_PI*l_fac) / (denominator*exp2_l*l_over2_fac);
    }
}

//  this is a direct evaluation of the spherical harmonic basis functions for
//  a phong reflector with exponent s, taken from Hanrahan & Ramamoorthi. 
//  "A Signal-Processing Framework for Inverse Rendering" (SIGGRAPH 2001)
//  this is an approximation to the actual coefficients, good when s >> l
//  don't compute 10th order SH for (R.L)^5 with this...
double Phong_Al_Evaluator::operator ()( int l ) const
{
    if ( l<0 ) return 0;

    double lambda = sqrt(4*M_PI/(2.*double(l)+1.));
    double x = -double(l*l) / (2.*m_specular);
    return exp(x) / lambda;
}