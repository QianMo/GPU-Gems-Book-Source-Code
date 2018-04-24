// ************************************************
// ziggurat.hpp
// authors: Lee Howes and David B. Thomas
//
// Class to support calling a Mersenne Twister/
// Ziggurat implementation on the CPU from the
// test framework.
// ************************************************


#ifndef ziggurat_hpp
#define ziggurat_hpp

#include "../wallace_base.hpp"

class Ziggurat:public RNG
{
  protected:
	boost::mt19937 m_mt;

	enum
	{ BUFFER_SIZE = 512 };
	float m_buffer[512];

	float UNI()
	{
		return (.5 + ((signed) m_mt()) * .2328306e-9);
	}

	unsigned long kn[128];
	float wn[128], fn[128];

	float nfix(void)
	{
		long hz = m_mt();
		unsigned long iz = hz & 127;
		if (((unsigned long) abs(hz)) < kn[iz])
			return hz * wn[iz];

		const float r = 3.442620f;
		float x, y;
		for (;;)
		{
			x = hz * wn[iz];
			if (iz == 0)
			{
				do
				{
					x = -logf(UNI()) * 0.2904764;
					y = -logf(UNI());
				}
				while (y + y < x * x);
				return (hz > 0) ? r + x : -r - x;
			}
			if (fn[iz] + UNI() * (fn[iz - 1] - fn[iz]) < expf(-.5 * x * x))
				return x;
			hz = m_mt();
			iz = hz & 127;
			if (((unsigned long) abs(hz)) < kn[iz])
				return (hz * wn[iz]);
		}
	}


	void zigset()
	{
		const double m1 = 2147483648.0;
		double dn = 3.442619855899, tn = dn, vn = 9.91256303526217e-3, q;
		int i;
		/* Tables for RNOR: */
		q = vn / exp(-.5 * dn * dn);
		kn[0] = (unsigned long) ((dn / q) * m1);
		kn[1] = 0;
		wn[0] = q / m1;
		wn[127] = dn / m1;
		fn[0] = 1.;
		fn[127] = exp(-.5 * dn * dn);
		for (i = 126; i >= 1; i--)
		{
			dn = sqrt(-2. * log(vn / dn + exp(-.5 * dn * dn)));
			kn[i + 1] = (unsigned long) ((dn / tn) * m1);
			tn = dn;
			fn[i] = exp(-.5 * dn * dn);
			wn[i] = dn / m1;
		}
	}

	virtual void UpdateRngOutput()
	{
		for (unsigned i = 0; i < BUFFER_SIZE; i++)
		{
			m_buffer[i] = nfix();
		}
		m_rngOutputCurr = m_buffer;
	}
  public:
  Ziggurat():m_mt(time(NULL))
	{
		zigset();
	}

	const char *Name()
	{
		return "Ziggurat";
	}

	const char *Description()
	{
		return "Does a Ziggurat based on a mersenne twister.";
	}

	virtual void Generate(unsigned count, float *values)
	{
		for (unsigned i = 0; i < count; i++)
		{
			values[i] = nfix();
		}
	}
	virtual float Generate()
	{
		return nfix();
	}

	virtual float operator() ()
	{
		return nfix();
	}
};

#endif
