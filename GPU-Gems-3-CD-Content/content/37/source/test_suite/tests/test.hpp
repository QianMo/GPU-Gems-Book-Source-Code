// ************************************************
// simulation.hpp
// authors: Lee Howes and David B. Thomas
//
// Basic code to support statistical tests.
// Includes abstract Test superclass,
// result class and the uniform distribution to
// normal distribution mapper used for
// providing uniformly distributed output
// to feed into statistical tests.
// ***********************************************

#ifndef test_hpp
#define test_hpp

#include "../wallace_base.hpp"

#include "gsl/gsl_cdf.h"

struct TestResult
{
	std::vector < char >generatorName;
	  std::vector < char >testName;
	  std::vector < char >testPart;
	double nsamples;
	bool fail;
	double pvalue;

	void WriteRow(FILE * dst)
	{
		fprintf(dst, "%s, %s, %s, %lf, %s, %lg\n", &generatorName[0], &testName[0], &testPart[0], nsamples, fail ? "FAIL" : "pass", pvalue);
	}
};

struct TestOptions
{
	FILE *log;
	bool abortOnFirstFail;

	  TestOptions():log(stderr), abortOnFirstFail(true)
	{
	}
};

class Test
{
  public:
	virtual const char *Name() = 0;
	virtual const char *Description() = 0;

	virtual void Execute(RNG * pSrc, TestOptions & opts, std::vector < TestResult > &results) = 0;

	  virtual ~ Test()
	{
	}
};

class FakeTest:public Test
{
  public:
	virtual const char *Name()
	{
		return "FakeTest";
	};
	virtual const char *Description()
	{
		return "Doesn't do anything, just generates numbers forever.";
	};

	virtual void Execute(RNG * pSrc, TestOptions & opts, std::vector < TestResult > &results)
	{
		fprintf(stderr, "Running FakeTest. This will never end...\n");

		while (1)
		{
			pSrc->Generate();
		}
	}
};

class NormalToUniformIntMapper
{
  private:
	RNG * m_pImpl;
	boost::mt19937 m_fill;

	float Phi(float x)
	{
		float s = x, t = 0, b = x, q = x * x, i = 1;
		while (s != t)
			  s = (t = s) + (b *= q / (i += 2));
		  return .5f + s * exp(-.5f * q - .91893853320467274178f);
	}
  public:
	  NormalToUniformIntMapper(RNG * pImpl):m_pImpl(pImpl), m_fill(time(NULL) + lrand48())
	{
#ifndef NDEBUG
		for (unsigned i = 1; i <= 1000; i++)
		{
			double x = i / 1000.0;
			double p = gsl_cdf_ugaussian_Pinv(x);
			double gx = Phi(p);
			if (fabs(gx - x) > 1e-6)
			{
				fprintf(stderr, "  Chi(%lg)=%lg, should be %lg, err=%lg\n", p, gx, x, fabs(gx - x));
				assert(0);
			}
		}
#endif
	}

	unsigned Generate()
	{
		float u = Phi(m_pImpl->Generate());
		assert(u >= 0 && u <= 1);
		unsigned fu = (unsigned) (u * 0xFFFFFFFF);
		return (fu & 0xFFFFFC00) | (m_fill() & 0x3FF);
	}

	unsigned operator() ()
	{
		return Generate();
	}

	void Generate(unsigned n, unsigned *dst)
	{
		BOOST_STATIC_ASSERT(sizeof(float) == sizeof(unsigned));

		float *fdst = (float *) dst;
		m_pImpl->Generate(n, fdst);

		for (unsigned i = 0; i < n; i++)
		{
			float u = Phi(fdst[i]);
			unsigned fu = (unsigned) (u * 0xFFFFFFFF);
			dst[i] = (fu & 0xFFFFFC00) | (m_fill() & 0x3FF);
		}
	}
};

#endif
