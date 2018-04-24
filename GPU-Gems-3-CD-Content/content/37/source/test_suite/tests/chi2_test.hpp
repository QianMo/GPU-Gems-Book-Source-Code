// ************************************************
// chi2_test.hpp
// authors: Lee Howes and David B. Thomas
//
// Implementation of a chi-squared test with
// construction functions suppotring 
// a number of variants.
// ************************************************

#include "../common.hpp"

#include "test.hpp"
#include "chi2_hist.hpp"

class Chi2TestSet:public Test
{
  private:
	std::vector < char >m_setName, m_setDescription;
	  std::vector < boost::shared_ptr < Chi2Hist > >m_hists;
	double m_maxSamples;

	bool IsBadPValue(double x)
	{
		return (std::min(x, 1 - x) < 1e-6);
	}
  public:
	  Chi2TestSet(const char *name, double maxSamples,
				  std::vector < boost::shared_ptr < Chi2Hist > >&hists):m_setName(StrToVec(name)), m_hists(hists), m_maxSamples(maxSamples)
	{
		m_setDescription.resize(1024, 0);	// yay, buffer overflows!
		sprintf(&m_setDescription[0], "For up to %lg samples (2^%0.3lg) apply the following tests:", m_maxSamples, std::log(m_maxSamples) / std::log(2));
		for (unsigned i = 0; i < m_hists.size(); i++)
		{
			m_setDescription.resize(m_setDescription.size() + strlen(m_hists[i]->Name()) + 10);
			strcat(&m_setDescription[0], " ");
			strcat(&m_setDescription[0], m_hists[i]->Name());
			strcat(&m_setDescription[0], ".");
		}
	}

	virtual ~ Chi2TestSet()
	{
	}

	virtual const char *Name()
	{
		return &m_setName[0];
	}

	virtual const char *Description()
	{
		return &m_setDescription[0];
	}

	virtual void Execute(RNG * pSrc, TestOptions & opts, std::vector < TestResult > &results)
	{
		bool breakOnOneFail = opts.abortOnFirstFail;
		FILE *log = opts.log;

		unsigned preroll = 0, minSamples = 0;
		for (unsigned i = 0; i < m_hists.size(); i++)
		{
			preroll = std::max(preroll, m_hists[i]->RequiredPreroll());
			minSamples = std::max(minSamples, m_hists[i]->MinSamplesBeforeOutput());
			m_hists[i]->Reset();
		}
		if (log)
			fprintf(log, "  preroll=%u, minSamples=%u\n", preroll, minSamples);

		unsigned batchSize = 16384;

		std::vector < float >bufferStorage(batchSize + preroll);
		float *buffer = &bufferStorage[preroll];

		std::vector < double >pvalues(m_hists.size());
		std::vector < bool > failed(m_hists.size(), false);

		double target = std::min(m_maxSamples, (double) std::max(minSamples, batchSize));
		double done = 0;
		while (done < m_maxSamples)
		{
			if (log)
				fprintf(log, " Target=%lg, done=%lg\n", target, done);
			while (done < target)
			{
				unsigned todo = (unsigned) std::min((double) batchSize, target - done);
				pSrc->Generate(todo, buffer);

				for (unsigned i = 0; i < m_hists.size(); i++)
				{
					if (!failed[i])
					{
						m_hists[i]->AddSamples(todo, buffer);
					}
				}
				done += todo;

				// put the preroll at start
				std::copy(buffer + todo - preroll, buffer + todo, &bufferStorage[0]);
			}

			if (log)
				fprintf(log, "nsamples=2^%lg\n", std::log(done) / std::log(2));
			for (unsigned i = 0; i < m_hists.size(); i++)
			{
				if (!failed[i])
				{
					std::pair < double, double >curr = m_hists[i]->CalcPValue();
					if (log)
						fprintf(log, "  %s, pvalue=%lg, stat=%lg\n", m_hists[i]->Name(), curr.first, curr.second);
					bool nowFailed = IsBadPValue(curr.first);
					if (nowFailed && !failed[i] && log)
						fprintf(log, "    FAILED\n");
					failed[i] = nowFailed;
				}
			}

			bool allFailed = true, oneFailed = false;
			for (unsigned i = 0; i < m_hists.size(); i++)
			{
				allFailed = allFailed && failed[i];
				oneFailed = oneFailed || failed[i];
			}

			if (allFailed)
			{
				if (log)
					fprintf(log, "ALL FAILED.\n");
				break;
			}

			if (breakOnOneFail && oneFailed)
			{
				if (log)
					fprintf(log, "ONE FAILED, breaking.\n");
				break;
			}
			fflush(log);

			target = std::min(m_maxSamples, ceil(target * 1.5));
		}

		TestResult r;
		r.generatorName = StrToVec(pSrc->Name());
		r.testName = m_setName;
		for (unsigned i = 0; i < m_hists.size(); i++)
		{
			r.testPart = StrToVec(m_hists[i]->Name());
			r.nsamples = m_hists[i]->TotalSamples();
			r.fail = failed[i];
			r.pvalue = m_hists[i]->CalcPValue().first;
			results.push_back(r);
		}
	}
};

boost::shared_ptr < Test > MakeQuickChi2Test()
{
	std::vector < boost::shared_ptr < Chi2Hist > >hists;
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 1, 64 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 2, 8 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 3, 5 > ()));
	return boost::shared_ptr < Test > (new Chi2TestSet("QuickChi2", 1U << 26, hists));
}

boost::shared_ptr < Test > MakeLongUniDimChi2Test()
{
	std::vector < boost::shared_ptr < Chi2Hist > >hists;
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 1, 4096 > ()));
	return boost::shared_ptr < Test > (new Chi2TestSet("LongUniDimChi2", pow(2, 32), hists));
}

boost::shared_ptr < Test > MakeLongSpectralChi2Test()
{
	std::vector < boost::shared_ptr < Chi2Hist > >hists;
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 2, 128 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 3, 25 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 4, 11 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 5, 7 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 6, 5 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 7, 4 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 8, 3 > ()));
	hists.push_back(boost::shared_ptr < Chi2Hist > (new Chi2HistImpl < 14, 2 > ()));
	return boost::shared_ptr < Test > (new Chi2TestSet("LongSpectralChi2", pow(2, 32), hists));
}
