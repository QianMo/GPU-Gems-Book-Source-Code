// ************************************************
// chi2_hist.hpp
// authors: Lee Howes and David B. Thomas
//
// Support for generation of chi-squared test
// histograms.
// ************************************************

#ifndef chi2_hist_hpp
#define chi2_hist_hpp

#include "gsl/gsl_cdf.h"

#include <algorithm>
#include <vector>
#include <numeric>

class Chi2Hist
{
  public:
	virtual ~ Chi2Hist()
	{
	}

	virtual const char *Name() const = 0;

	virtual unsigned MinSamplesBeforeOutput() const = 0;
	virtual unsigned RequiredPreroll() const = 0;

	virtual void Reset() = 0;
	virtual void AddSamples(unsigned n, const float *point) = 0;

	virtual double TotalSamples() const = 0;
	virtual double TotalDataPoints() const = 0;

	// return pair of p-value,statistic
	virtual std::pair < double, double >CalcPValue() const = 0;
};

namespace detail
{
	template < unsigned L, unsigned K > struct CompileTimePow
	{
		enum
		{ val = K * CompileTimePow < L - 1, K >::val };
	};

	  template < unsigned K > struct CompileTimePow <0, K >
	{
		enum
		{ val = 1 };
	};
};

template < unsigned D, unsigned K > class Chi2HistImpl:public Chi2Hist
{
  public:
	enum
	{ N = detail::CompileTimePow < D, K >::val };
  private:

	float m_bounds[K + 1];
	unsigned m_counts[N];
	double m_total;

	unsigned m_preroll;

	char m_name[256];

	unsigned FindPartition(float x) const
	{
		if (K == 2)
		{
			return x < 0 ? 1 : 0;
		}
		else if (K == 3)
		{
			const float p_third = 0.430727299295458f;
			  return (x < -p_third) ? 0 : (x > p_third ? 2 : 1);
		}
		else if (K == 4)
		{
			const float p_quarter = 0.674489750196082f;
			return x < 0 ? (x < -p_quarter ? 0 : 1) : (x < p_quarter ? 2 : 3);
		}
		else if (K == 5)
		{
			const float p_fifth = 0.841621233572914f;
			const float p_two_fifths = 0.2533471031358f;
			if (x < -p_two_fifths)
			{
				return x < -p_fifth ? 0 : 1;
			}
			else
			{
				return x < p_two_fifths ? 2 : (x < p_fifth ? 3 : 4);
			}
		}
		else
		{
			unsigned i = std::lower_bound(m_bounds, m_bounds + K, x) - m_bounds - 1;
			assert(i < K);
			assert(m_bounds[i] <= x);
			assert(x <= m_bounds[i + 1]);
			return i;
		}
	}

	unsigned BuildIndex(const float *point) const
	{
		unsigned acc = 0;
		for (unsigned i = 0; i < D; i++)
		{
			acc = acc * K + FindPartition(point[i]);
		}
		return acc;
	}
  public:
	Chi2HistImpl()
	{
		sprintf(m_name, "Chi2Hist(D=%u,K=%u,N=%u)", D, K, N);

		for (unsigned i = 0; i <= K; i++)
		{
			m_bounds[i] = gsl_cdf_ugaussian_Pinv(double (i) / K);
		}
		m_bounds[0] = -8;
		m_bounds[K] = +8;

		Reset();
	}

	virtual const char *Name() const
	{
		return m_name;
	}

	virtual unsigned MinSamplesBeforeOutput() const
	{
		return 10 * N * D;
	}

	virtual unsigned RequiredPreroll() const
	{
		return D;
	}

	virtual double TotalSamples() const
	{
		return m_total * D;
	}

	virtual double TotalDataPoints() const
	{
		return m_total;
	}

	virtual void Reset()
	{
		std::fill(m_counts, m_counts + N, 0);
		m_total = 0;
		m_preroll = 0;
	}

	void AddSamples(unsigned n, const float *point)
	{
		if (m_preroll)
		{
			n += m_preroll;
			point -= m_preroll;
		}
		m_preroll = n % D;

		unsigned todo = n / D;
		for (unsigned i = 0; i < todo; i++, point += D)
			m_counts[BuildIndex(point)]++;
		m_total += todo;
	}

	std::pair < double, double >CalcPValue() const
	{
		double expected = m_total / (N);
		if (expected > (1U << 31))
		{
			fprintf(stderr, "Too many samples accumulated for size of bucket. Bucket counters are likely to overflow");
			exit(1);
		}
		if (expected < 10)
		{
			fprintf(stderr, "expected of %lg < 10 for Chi2(%d,%d), numSamples=%lg\n", expected, D, K, m_total);
			exit(1);
		}

		assert(m_total == std::accumulate(m_counts, m_counts + N, 0.0));

		std::vector < double >got(m_counts, m_counts + N);
		for (unsigned i = 0; i < N; i++)
		{
			double diff = (got[i] - expected);
			got[i] = diff * diff / expected;
		}
		double stat = std::accumulate(got.begin(), got.end(), 0.0);
		return std::make_pair(gsl_cdf_chisq_P(stat, N - 1), stat);
	}
};

#endif
