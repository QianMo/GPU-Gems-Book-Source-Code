// ************************************************
// batched_rng_base.hpp
// authors: Lee Howes and David B. Thomas
//
// Superclass to support implementations of
// batched random number generators such
// as the CUDA implementations.
// ************************************************


#ifndef impls__batched_rng_base_hpp
#define impls__batched_rng_base_hpp

#include "../wallace_base.hpp"

#include <cmath>
#include <numeric>

#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"

template < unsigned tBATCH_SIZE > class BatchedRngBase:public RNG
{
  public:
	enum
	{ BATCH_SIZE = tBATCH_SIZE };

  private:
	float m_output[BATCH_SIZE];
	float *m_outputReadPos;
  protected:
	  virtual void GenerateImpl(float *dest) = 0;

	unsigned OutputSamplesLeft()
	{
		assert(m_outputReadPos >= m_output && m_outputReadPos <= m_output + BATCH_SIZE);
		return (m_output + BATCH_SIZE) - m_outputReadPos;
	}
	virtual void UpdateRngOutput()
	{
		GenerateImpl(m_output);
		m_rngOutputCurr = m_output;
	}
  public:
  BatchedRngBase():m_outputReadPos(m_output + BATCH_SIZE)
	{
	}

	void Generate(unsigned count, float *values)
	{
		if (count > OutputSamplesLeft())
		{
			std::copy(m_outputReadPos, m_output + BATCH_SIZE, values);
			count -= OutputSamplesLeft();
			values += OutputSamplesLeft();
			m_outputReadPos = m_output + BATCH_SIZE;	// -> OutputSamplesLeft()==0

			// transform directly into output buffer
			while (count >= BATCH_SIZE)
			{
				GenerateImpl(values);
				count -= BATCH_SIZE;
				values += BATCH_SIZE;
			}

			// always leave scope with a full pool
			GenerateImpl(m_output);
			m_outputReadPos = m_output;
		}

		assert(count <= OutputSamplesLeft());

		std::copy(m_outputReadPos, m_outputReadPos + count, values);
		m_outputReadPos += count;
	}

	float Generate()
	{
		if (OutputSamplesLeft() == 0)
		{
			GenerateImpl(m_output);
			m_outputReadPos = m_output;
		}
		return *m_outputReadPos++;
	}

	float operator() ()
	{
		return Generate();
	}
};

#endif
