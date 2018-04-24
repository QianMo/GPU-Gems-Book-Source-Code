// ************************************************
// wallace_cuda.hpp
// authors: Lee Howes and David B. Thomas
//
// Wrapper for allowing the calling of CUDA
// based Wallace generators from the test
// framework.
// ************************************************

#ifndef wallace_cuda_hpp
#define wallace_cuda_hpp

#include "../wallace_base.hpp"

#include <cmath>

#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"

typedef void (*cuda_transform_func_t) (unsigned POOL_SIZE, unsigned BATCH_SIZE, unsigned BATCH_COUNT, unsigned seed, float *chi2Corr,	// [in] BATCH_SIZE*BATCH_COUNT
									   float *poolIn,	// [in] POOL_SIZE*BATCH_COUNT (can be modified).
									   float *poolOut,	// [out] POOL_SIZE*BATCH_COUNT. poolIn!=poolOut
									   float *output	// [out] POOL_SIZE*BATCH_COUNT*BATCH_SIZE
	);

template < cuda_transform_func_t TRANSFORM_FUNC, unsigned POOL_SIZE, unsigned BATCH_SIZE = 16, unsigned BATCH_COUNT = 16 > class WallaceCUDA:public WallaceBase < POOL_SIZE * BATCH_COUNT, 1, 1,
	POOL_SIZE * BATCH_SIZE * BATCH_COUNT,
	false >
{
  protected:
	// We pretend to WallaceBase that we have one thread, 
	typedef WallaceBase < POOL_SIZE * BATCH_COUNT, 1, 1, POOL_SIZE * BATCH_SIZE * BATCH_COUNT, false > base_t;

	float m_chi2Corr[BATCH_SIZE * BATCH_COUNT];
	unsigned m_seed;

	virtual void PreTransformSetup()
	{
		for (unsigned i = 0; i < BATCH_SIZE * BATCH_COUNT; i++)
		{
			m_chi2Corr[i] = MakeChi2Scale(POOL_SIZE);
		}
		m_seed = (1664525U * m_seed + 1013904223U) & 0xFFFFFFFF;
	}

	// Runs a kernel thread. Per thread arguments are managed by the inheritor.
	virtual void TransformKernel(unsigned tid, unsigned pass)
	{
		TRANSFORM_FUNC(POOL_SIZE, BATCH_SIZE, BATCH_COUNT, m_seed,
					   m_chi2Corr, const_cast < float *>(base_t::GetRawInputPool()), base_t::GetRawOutputPool(), base_t::GetRawOutputBuffer());
	}

	virtual void ReseedPoolImpl(float *poolStg)
	{
		for (unsigned k = 0; k < BATCH_SIZE; k++)
		{
			float *pool = poolStg + k * POOL_SIZE;
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
	}

	const char *m_name, *m_desc;
  public:
	WallaceCUDA(const char *name, const char *desc):m_seed(lrand48()), m_name(name), m_desc(desc)
	{
		base_t::ReseedPool();
	}

	const char *Name()
	{
		return m_name;
	}

	const char *Description()
	{
		return m_desc;
	}
};

#endif
