// ************************************************
// wallace_base.hpp
// authors: Lee Howes and David B. Thomas
//
// Base class for Wallace generator implementations
// containing much of the basic management code
// for maintaining the source pools and generating
// Chi2 correction values and so on.
// ************************************************

#ifndef wallace_base_hpp
#define wallace_base_hpp

#include <cmath>
#include <numeric>

#include "rng.hpp"

#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"

double RandN()
{
	static bool cached = false;
	static double cn;

	if (cached)
	{
		cached = false;
		return cn;
	}

	double a = sqrt(-2 * log(drand48()));
	double b = 6.283185307179586476925286766559 * drand48();
	cn = sin(b) * a;
	cached = true;
	return cos(b) * a;
}

double MakeChi2Scale(unsigned N, double r = RandN())
{
	double chic1 = sqrt(sqrt(1.0 - 1.0 / N));
	double chic2 = sqrt(1.0 - chic1 * chic1);
	return chic1 + chic2 * r;
}

void Hadamard4x4a(float &p, float &q, float &r, float &s)
{
	float t = (p + q + r + s) / 2;
	p = p - t;
	q = q - t;
	r = t - r;
	s = t - s;
}

void Hadamard4x4b(float &p, float &q, float &r, float &s)
{
	float t = (p + q + r + s) / 2;
	p = t - p;
	q = t - q;
	r = r - t;
	s = s - t;
}

class Wallace
{
  public:
	virtual ~ Wallace()
	{
	}

	virtual const char *Name() = 0;
	virtual const char *Description() = 0;

	virtual void Generate(unsigned count, float *values) = 0;
	virtual float Generate() = 0;
	virtual float operator() () = 0;
};

template < unsigned tPOOL_SIZE, unsigned tNUM_THREADS, unsigned tCYCLES_PER_PASS, unsigned tOUTPUTS_PER_TRANSFORM = tPOOL_SIZE,
#ifdef NDEBUG
	bool tVERIFY = false,
#else
	bool tVERIFY = true,
#endif
unsigned tNUM_PASSES = 1 > class WallaceBase:public RNG {
  public:
	enum
	{ POOL_SIZE = tPOOL_SIZE };
	enum
	{ NUM_THREADS = tNUM_THREADS };
	enum
	{ CYCLES_PER_PASS = tCYCLES_PER_PASS };
	enum
	{ OUTPUTS_PER_TRANSFORM = tOUTPUTS_PER_TRANSFORM };
	enum
	{ VERIFY = tVERIFY };
	enum
	{ NUM_PASSES = tNUM_PASSES };

	enum
	{ POOL_SIZE_MASK = POOL_SIZE - 1 };

	enum
	{ WARP_SIZE = NUM_THREADS < 16 ? 1 : 16 };	// blank it down for non-wallace impls
	enum
	{ NUM_WARPS = NUM_THREADS / WARP_SIZE };

	//BOOST_STATIC_ASSERT(NUM_THREADS%WARP_SIZE==0);

  private:
	float m_buffers[2][POOL_SIZE];
	float m_output[OUTPUTS_PER_TRANSFORM];

	float *m_pool, *m_temporary;
	float *m_dest;

	unsigned m_transformsSinceReseed;

	enum
	{ V_POOL_SIZE = VERIFY ? POOL_SIZE : 1 };
	enum
	{ V_NUM_WARPS = VERIFY ? NUM_WARPS : 1 };
	enum
	{ V_CYCLES_PER_PASS = VERIFY ? CYCLES_PER_PASS : 1 };
	unsigned m_vReadCounts[V_POOL_SIZE], m_vWriteCounts[V_POOL_SIZE];
	unsigned m_vBankAccesses[V_NUM_WARPS][V_CYCLES_PER_PASS][WARP_SIZE];
	std::vector < unsigned >m_vPropCountStg;	//[V_POOL_SIZE];
	unsigned *m_vPropCounts;
	unsigned m_vPassesSincePropSeed;

	void ClearPerPassCounters()
	{
		if (VERIFY)
		{
			std::fill(m_vReadCounts, m_vReadCounts + V_POOL_SIZE, 0);
			std::fill(m_vWriteCounts, m_vWriteCounts + V_POOL_SIZE, 0);
			std::fill(&m_vBankAccesses[0][0][0], &m_vBankAccesses[0][0][0] + V_NUM_WARPS * V_CYCLES_PER_PASS * WARP_SIZE, 0);
		}
	}

	void CheckPerPassCounters()
	{
		if (VERIFY)
		{

			for (unsigned i = 0; i < POOL_SIZE; i++)
			{
				if (m_vReadCounts[i] == 0)
					printf("  Pool[%u] was not read.\n", i);
				if (m_vWriteCounts[i] == 0)
					printf("  Pool[%u] was not written.\n", i);
			}

			unsigned totalWarpAccesses = 0, totalThreadAccesses = 0, totalCycles = 0;
			for (unsigned w = 0; w < NUM_WARPS; w++)
			{
				for (unsigned c = 0; c < CYCLES_PER_PASS; c++)
				{
					unsigned accesses = 0, cycles = 0;
					for (unsigned i = 0; i < WARP_SIZE; i++)
					{
						unsigned tt = m_vBankAccesses[w][c][i];
						if (tt)
						{
							accesses += tt;
							if (cycles == 0)
								cycles = 1;
							cycles += tt - 1;
						}
					}
					if (cycles > 1)
					{
						printf("    Conflict[%u][%u], accesses=%u, cycles=%u\n", w, c, accesses, cycles);
					}
					totalWarpAccesses += accesses ? 1 : 0;
					totalThreadAccesses += accesses;
					totalCycles += cycles;
				}
			}
			if (totalCycles > totalWarpAccesses)
			{
				printf
					("  acc/warp=%u, acc/thread=%u, cycles=%u, extra=%u, ratio=%lg\n",
					 totalWarpAccesses, totalThreadAccesses, totalCycles,
					 totalCycles - totalWarpAccesses, (totalCycles - totalWarpAccesses) / double (totalWarpAccesses));
			}

			m_vPassesSincePropSeed++;
			unsigned propCount = std::accumulate(m_vPropCounts, m_vPropCounts + V_POOL_SIZE,
												 0);
			if (propCount > POOL_SIZE)
			{
				for (unsigned i = 0; i < V_POOL_SIZE; i++)
				{
					printf("%u\n", m_vPropCounts[i]);
				}
			}
			assert(propCount <= V_POOL_SIZE);
			if (propCount == V_POOL_SIZE)
			{
				//printf("  propogation complete in %u passes.\n", m_vPassesSincePropSeed);
				std::fill(m_vPropCounts, m_vPropCounts + V_POOL_SIZE, 0);
				m_vPropCounts[lrand48() % V_POOL_SIZE] = 1;
				m_vPassesSincePropSeed = 0;
			}
			else if (m_vPassesSincePropSeed > V_POOL_SIZE / 16)
			{
				printf("  propogated to %u after %u passes.\n", propCount, m_vPassesSincePropSeed);
			}
			fflush(stdout);
		}
	}
  protected:
	//////////////////////////////////////////////////////////////////
	// Accessors for implementations
	// Provides access to the pool (i.e. shared memory)
	float ReadPool(unsigned tid, unsigned cycle, unsigned address)
	{
		assert(address < POOL_SIZE);
		assert(tid < NUM_THREADS);
		assert(cycle < CYCLES_PER_PASS);
		if (VERIFY)
		{
			m_vReadCounts[address]++;
			if (m_vReadCounts[address] > 1)
			{
				printf("Pool[%u] read %u times.\n", address, m_vReadCounts[address]);
			}

			if ((tid / WARP_SIZE) >= NUM_WARPS)
			{
				fprintf(stderr, "tid=%u, WARP_SIZE=%u, NUM_WARPS=%u, NUM_THREADS=%u\n", tid, WARP_SIZE, NUM_WARPS, NUM_THREADS);
			}
			assert(tid / WARP_SIZE < V_NUM_WARPS);

			m_vBankAccesses[tid / WARP_SIZE][cycle][address % WARP_SIZE]++;
		}
		return m_pool[address];
	}
	void WritePool(unsigned tid, unsigned cycle, unsigned address, float value, unsigned sourceCount, const unsigned *sources)
	{
		assert(address < POOL_SIZE);
		assert(tid < NUM_THREADS);
		assert(cycle < CYCLES_PER_PASS);
		if (VERIFY)
		{
			m_vWriteCounts[address]++;
			if (m_vWriteCounts[address] > 1)
			{
				printf("Pool[%u] read %u times.\n", address, m_vWriteCounts[address]);
			}
			m_vBankAccesses[tid / WARP_SIZE][cycle][address % WARP_SIZE]++;

			for (unsigned i = 0; i < sourceCount; i++)
			{
				assert(sources[i] < POOL_SIZE);
				assert(m_vPropCounts[address] <= 1);
				assert(m_vPropCounts[sources[i]] <= 1);
				m_vPropCounts[address] |= m_vPropCounts[sources[i]];
			}
		}
		m_temporary[address] = value;
	}
	void WriteOutput(unsigned tid, unsigned cycle, unsigned address, float value)
	{
		assert(address < OUTPUTS_PER_TRANSFORM);
		assert(tid < NUM_THREADS);
		assert(cycle < CYCLES_PER_PASS);
		m_dest[address] = value;
	}

	const float *GetRawInputPool()
	{
		BOOST_STATIC_ASSERT(!VERIFY);
		return m_pool;
	}

	float *GetRawOutputPool()
	{
		BOOST_STATIC_ASSERT(!VERIFY);
		return m_temporary;
	}

	float *GetRawOutputBuffer()
	{
		BOOST_STATIC_ASSERT(!VERIFY);
		return m_dest;
	}

	virtual void PreTransformSetup()
	{
	}
	virtual void PostTransformCleanup()
	{
	}

	// Runs a kernel thread. Per thread arguments are managed by the inheritor.
	virtual void TransformKernel(unsigned tid, unsigned pass) = 0;

	void Transform(float *dest = 0)
	{

		m_dest = dest ? dest : m_output;

		PreTransformSetup();
		for (unsigned k = 0; k < NUM_PASSES; k++)
		{
			ClearPerPassCounters();
			for (unsigned i = 0; i < NUM_THREADS; i++)
			{
				TransformKernel(i, k);
			}
			std::swap(m_pool, m_temporary);
			CheckPerPassCounters();
		}
		PostTransformCleanup();

		if (!dest)
		{
			m_rngOutputCurr = m_output;
			m_rngOutputEnd = m_output + OUTPUTS_PER_TRANSFORM;
		}
	}


	virtual void UpdateRngOutput()
	{
		Transform();
	}


	unsigned OutputSamplesLeft()
	{
		assert(m_rngOutputCurr >= m_output && m_rngOutputCurr <= m_output + OUTPUTS_PER_TRANSFORM);
		return (m_output + OUTPUTS_PER_TRANSFORM) - m_rngOutputCurr;
	}

	virtual void ReseedPoolImpl(float *pool)
	{
		double sumSquares = 0;
		for (unsigned i = 0; i < POOL_SIZE; i++)
		{
			pool[i] = RandN();
			sumSquares += pool[i] * pool[i];
		}
		double scale = sqrt(POOL_SIZE / sumSquares);
		for (unsigned i = 0; i < POOL_SIZE; i++)
		{
			pool[i] *= scale;
		}
	}
  public:
  WallaceBase():m_pool(m_buffers[0]), m_temporary(m_buffers[1]), m_vPropCountStg(POOL_SIZE, 0), m_vPropCounts(&m_vPropCountStg[0])
	{
		ReseedPool();
	}

	double PoolSumOfSquares() const
	{
		double sumSquares = 0;
		for (unsigned i = 0; i < POOL_SIZE; i++)
		{
			sumSquares += m_pool[i] * m_pool[i];
		}
		return sumSquares;
	}

	virtual void ReseedPool()
	{
		ReseedPoolImpl(m_pool);
		m_transformsSinceReseed = 0;
		m_rngOutputCurr = m_output + OUTPUTS_PER_TRANSFORM;	// force a transform before any output

		if (VERIFY)
		{
			std::fill(m_vPropCounts, m_vPropCounts + V_POOL_SIZE, 0);
			m_vPropCounts[lrand48() % V_POOL_SIZE] = 1;
		}
	}

	void Generate(unsigned count, float *values)
	{
		if (count > OutputSamplesLeft())
		{
			std::copy(m_rngOutputCurr, m_output + OUTPUTS_PER_TRANSFORM, values);
			count -= OutputSamplesLeft();
			values += OutputSamplesLeft();
			m_rngOutputCurr = m_output + OUTPUTS_PER_TRANSFORM;	// -> OutputSamplesLeft()==0

			// transform directly into output buffer
			while (count >= OUTPUTS_PER_TRANSFORM)
			{
				Transform(values);
				count -= OUTPUTS_PER_TRANSFORM;
				values += OUTPUTS_PER_TRANSFORM;
			}

			// always leave scope with a full pool
			Transform();
		}

		assert(count <= OutputSamplesLeft());

		std::copy(m_rngOutputCurr, m_rngOutputCurr + count, values);
		m_rngOutputCurr += count;
	}

	float Generate()
	{
		if (OutputSamplesLeft() == 0)
		{
			Transform();
		}
		return *m_rngOutputCurr++;
	}

	float operator() ()
	{
		return Generate();
	}
};

#endif // wallace_base_hpp
