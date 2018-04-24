// ************************************************
// test.cpp
// authors: Lee Howes and David B. Thomas
//
// Main source file for the execution of performance
// and statistical quality tests for the random
// number generators and options simulations
// linked to the GPU Gems 3 chapter
// Gaussian Random Number Generation and 
// Financial Simulation in CUDA
// ************************************************

#include "rng.hpp"

#include "impls/ziggurat.hpp"
#include "impls/wallace_v4.hpp"
#include "impls/wallace_cuda_v4.hpp"
#include "impls/nrng_cuda_tausworthe.hpp"

#include "tests/chi2_test.hpp"

#include "thing_mapper.hpp"

#include "sims/gem_lookback_option.hpp"
#include "sims/gem_asian_option.hpp"

void speedTests();

void benchmarkSoftwareOption(char *name, char *rngname, ThingMapper < RNG > &rngs, ThingMapper < Simulation > &sims)
{

	// Benchmark software lookback option
	boost::shared_ptr < RNG > rng = rngs.Get(rngname);
	boost::shared_ptr < Simulation > sim = sims.Get(name);
	fprintf(stderr, "Benchmarking %s using %s...\n", sim->Name(), rng->Name());
	std::pair < double, double >res;
	double todo = 128, done;
	int start = clock(), end = start;
	while ((end - start) < CLOCKS_PER_SEC * 5)
	{
		start = clock();
		res = sim->Execute(rng.get(), todo);
		done = todo;
		end = clock();
		fprintf(stderr, "2^%0.3lf, %lf, %lf, %lf\n", log2(done), res.first, res.second, res.second / sqrt(done));
		todo *= 2;
	}
	double elapsed = (end - start) / double (CLOCKS_PER_SEC);
	fprintf(stdout,
			"%s, time=%lf, sims=%0.0lf, MSims/s=%lf, MSteps/s=%lf, MRngs/s=%lf, ",
			rng->Name(), elapsed, done, done / (elapsed * 1000000),
			done / (elapsed * 1000000) * sim->StepsPerSim(), done / (elapsed * 1000000) * sim->RngsPerSim());
	fprintf(stdout, "mean=%lf, stddev=%lf\n", res.first, res.second);
}

void benchmarkSoftwareRNG(char *name, ThingMapper < RNG > &rngs)
{

	// Speed test CPU Wallace
	boost::shared_ptr < RNG > rng = rngs.Get(name);
	std::vector < float >buffer(4096);
	fprintf(stderr, "Benchmarking %s...\n", rng->Name());
	double todo = buffer.size(), done;
	int start = clock(), end = start;
	while ((end - start) < CLOCKS_PER_SEC * 5)
	{
		// warm up
		rng->Generate(buffer.size(), &buffer[0]);
		start = clock();
		done = 0;
		while (done < todo)
		{
			rng->Generate(buffer.size(), &buffer[0]);
			done += buffer.size();
		}
		end = clock();
		todo *= 2;
	}
	double elapsed = (end - start) / double (CLOCKS_PER_SEC);
	fprintf(stdout, "%s, time=%lf, num=%0.0lf, MSamples/s=%lf, lastFloat=%f\n", rng->Name(), elapsed, done, done / (elapsed * 1000000), rng->Generate());
}

void chi2Test(char *name, ThingMapper < RNG > &rngs, ThingMapper < Test > &tests)
{
	boost::shared_ptr < RNG > rng = rngs.Get(name);
	boost::shared_ptr < Test > test = tests.Get("QuickChi2");
	fprintf(stderr, "Applying %s to %s\n", test->Name(), rng->Name());
	TestOptions opts;
	opts.log = NULL;
	std::vector < TestResult > results;
	test->Execute(rng.get(), opts, results);
	for (unsigned i = 0; i < results.size(); i++)
	{
		results[i].WriteRow(stdout);
	}
}

int main(int argc, char *argv[])
{
	try
	{
		srand48(time(NULL));
		srand48(time(NULL));
		nice(10);
		fprintf(stderr, "[adding tests and gens]\n");
		ThingMapper < RNG > rngs("Generator");
		rngs.Add(boost::shared_ptr < RNG > (new Ziggurat()));
		rngs.Add(boost::shared_ptr < RNG > (new Wallace_V4 <> ()));
		rngs.Add(MakeWallaceCUDA_V4());
		rngs.Add(MakeNrngCUDA_Tausworthe());
		fprintf(stderr, "[added %d gens]\n", rngs.Count());
		ThingMapper < Test > tests("Test");
		tests.Add(MakeQuickChi2Test());
		fprintf(stderr, "[added %d tests]\n", tests.Count());
		ThingMapper < Simulation > sims("Sim");
		sims.Add(MakeGemLookbackOption());
		sims.Add(MakeGemAsianOption());
		fprintf(stderr, "[added %d sims]\n", sims.Count());

		// Perform speed related CUDA tests
		printf("Performance results for CUDA versions of generators and options:\n");
		speedTests();
		printf("\n\n\n");
		printf("Performance results for CPU versions of generators and options:\n");
		benchmarkSoftwareRNG("Wallace_V4", rngs);
		benchmarkSoftwareRNG("Ziggurat", rngs);
		printf("\n\n\nBenchmark financial simulations::\n");
		benchmarkSoftwareOption("GemLookbackOption", "Ziggurat", rngs, sims);
		benchmarkSoftwareOption("GemAsianOption", "Ziggurat", rngs, sims);
		benchmarkSoftwareOption("GemLookbackOption", "Wallace_V4", rngs, sims);
		benchmarkSoftwareOption("GemAsianOption", "Wallace_V4", rngs, sims);

		// Perform chi2 tests
		printf("\n\n\nChi2 test results for CUDA random number generators:\n");
		chi2Test("WallaceCUDA_V4", rngs, tests);
		chi2Test("NrngCUDA_Tausworthe", rngs, tests);
		return 0;
	}
	catch(std::exception & e)
	{
		fprintf(stderr, "Exception : %s\n", e.what());
		return 1;
	}
}
