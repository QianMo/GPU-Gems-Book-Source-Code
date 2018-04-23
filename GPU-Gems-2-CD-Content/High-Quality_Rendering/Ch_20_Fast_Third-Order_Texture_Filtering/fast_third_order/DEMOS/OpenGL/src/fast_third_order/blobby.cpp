#include "blobby.h"

#include <time.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <limits>

#include <glh/glh_linear.h>


/// read the space-separated components of a vector from a stream
template <int N,typename T> inline std::istream& 
operator>>(std::istream& is, glh::vec<N,T>& _v)
{
	for(int i=0;i<_v.size();++i) 
		is >> _v[i];
	return is;
}


/// output a vector by printing its space-separated compontens
template <int N,typename T> inline std::ostream& 
operator<<(std::ostream& os, const glh::vec<N,T>& _v) 
{
	int i;
	for(i=0;i<_v.size()-1;++i) 
		os << _v[i] << " ";
	os << _v[i];
	return os;
}


/**
 * computes blobby data. 
 *
 * _is expects the following stream:
 *    resolution first.xyz last.xyz shift
 *    blob[0].center.xyz blob[0].weight
 *    blob[1].center.xyz blob[1].weight
 *    ...
 *
 * Type <Vec3f> should be readable from stream.
 * The output type <Int> should be some integer type.
 **/
template <typename Vec3f, typename Int>
Int* synthetic(std::istream& _is)
{
	Int s_max = std::numeric_limits<Int>::max();

	float r, s, r_min=std::numeric_limits<float>::max(), r_max=-r_min;
	Vec3f first, last;
	size_t n, i, j, k;

	_is >> n >> first >> last >> s;

	std::vector<Vec3f> c_vec;
	std::vector<float> w_vec;

	Vec3f x, h((last-first)*(1.0f/(n-1)));

	while(_is >> x >> r)
	{
		c_vec.push_back(x);
		w_vec.push_back(r);
	}

	std::vector<Vec3f>::const_iterator c_it;
	std::vector<float>::const_iterator w_it, w_end = w_vec.end();

	float *f = new float[n*n*n], *f_it = f;

	for(k=0,x[2]=first[2]; k<n; ++k, x[2]+=h[2])
		for(j=0,x[1]=first[1]; j<n; ++j, x[1]+=h[1])
			for(i=0,x[0]=first[0]; i<n; ++i, x[0]+=h[0], ++f_it)
			{
				w_it = w_vec.begin();
				c_it = c_vec.begin();
				for(r=s; w_it != w_end; ++c_it, ++w_it)
					r += *w_it * exp(-(x-*c_it).length());
				r_min = std::min(r, r_min);
				r_max = std::max(r, r_max);
				*f_it = r;
			}

	r = s_max/(r_max - r_min);
	Int *o = new Int[n*n*n], *o_it = o; f_it = f;
	for(i=0; i<n*n*n; ++i, ++f_it, ++o_it)
		*o_it = Int((*f_it - r_min) * r);

	delete[] f;
	return o;
}

unsigned short* dumbell(int _size)
{
	std::stringstream sstr;

	sstr.width(15);

	sstr << _size << " -1 -1 -1 1 1 1 0.0 ";

	float r = 0.5f, w = -1.16f;
	sstr <<  r << " " <<  r << " " <<  r << " " << w << " ";
	sstr << -r << " " << -r << " " << -r << " " << w << " "; 

	return synthetic<glh::vec3f, unsigned short>(sstr);
}

unsigned short* pyramid(int _size)
{
	std::stringstream sstr;

	sstr.width(15);

	sstr << _size << " -0.7 -0.7 -0.7 0.7 0.7 0.7 0.0 ";

	float r = 1.0f, s = r/sqrt(3.0f), t = s/sqrt(2.0f), w = -0.82f;
	sstr <<    0 << " " <<   0 << " " << t*3/2 << " " << w << " ";
	sstr <<    0 << " " <<  -s << " " <<  -t/2 << " " << w << " ";
	sstr <<  r/2 << " " << s/2 << " " <<  -t/2 << " " << w << " ";
	sstr << -r/2 << " " << s/2 << " " <<  -t/2 << " " << w << " ";
//	sstr << "0 0 0 " << -w/2 << " ";

	return synthetic<glh::vec3f, unsigned short>(sstr);
}

unsigned short* cube(int _size)
{
	std::stringstream sstr;

	sstr.width(15);

	sstr << _size << " -1 -1 -1 1 1 1 0.0 ";

	float r = 1.0f, w = -0.67f;
	sstr <<  r/2 << " " <<  r/2 << " " <<  r/2 << " " << w << " ";
	sstr << -r/2 << " " <<  r/2 << " " <<  r/2 << " " << w << " ";
	sstr <<  r/2 << " " << -r/2 << " " <<  r/2 << " " << w << " ";
	sstr << -r/2 << " " << -r/2 << " " <<  r/2 << " " << w << " ";
	sstr <<  r/2 << " " <<  r/2 << " " << -r/2 << " " << w << " ";
	sstr << -r/2 << " " <<  r/2 << " " << -r/2 << " " << w << " ";
	sstr <<  r/2 << " " << -r/2 << " " << -r/2 << " " << w << " ";
	sstr << -r/2 << " " << -r/2 << " " << -r/2 << " " << w << " ";
	sstr << "0 0 0 " << -w*0.75f << " ";

	return synthetic<glh::vec3f, unsigned short>(sstr);
}

unsigned short* random(int _size)
{
	std::stringstream sstr;

	sstr.width(15);

	sstr << _size << " 0 0 0 6 6 6 0.0 ";

	float s = 4.0f/RAND_MAX;

	for(int i=0; i<5; ++i)
		sstr << 1+rand()*s << " " << 1+rand()*s << " " << 1+rand()*s << " " << -1-rand()*s << " ";

	return synthetic<glh::vec3f, unsigned short>(sstr);
}

