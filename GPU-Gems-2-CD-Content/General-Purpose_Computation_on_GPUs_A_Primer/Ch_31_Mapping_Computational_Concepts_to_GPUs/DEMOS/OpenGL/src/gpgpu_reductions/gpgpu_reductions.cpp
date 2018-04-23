/*********************************************************************NVMH3****

Copyright NVIDIA Corporation 2004
TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

****************************************************************************/
// gpgpu_reductions.cpp
// This program tests the parallel reduction functionality of the pug API.
// It simply creates a large 2D array of random values.  It then computes the 
// sums of each row and each column, and the total sum of the array values.
// It does this both on the CPU and the GPU, and compares the results.
//
// Note that because the GPU implements this reduction in a data-parallel 
// using log(n) steps, the order of summation is different than on the CPU.
// This explains much of the error reported in the computation.
// 
#include <string>
#include <stdio.h>
#include <math.h>
#include <shared/pug/pug.h>

std::string path = "../shared/pug/";

inline bool isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

void testReductions(int width, int height)
{
    // CPU data
    float* cpuData = new float[width * height];
    float* cpuReducedRows = new float[height];
    float* cpuReducedColumns = new float[width];
    float* gpuReducedRows = new float[height];
    float* gpuReducedColumns = new float[width];

    float cpuTotalReduction = 0;
    float gpuTotalReduction = 0;
    int i = 0, j = 0;

    // initialize the data
    for (i = 0; i < width * height; ++i)
    {
        cpuData[i] = rand() / (float)RAND_MAX;
    }

    // zero the CPU results
    for (i = 0; i < height; ++i) cpuReducedRows[i]    = 0;
    for (i = 0; i < width; ++i)  cpuReducedColumns[i] = 0;
    cpuTotalReduction = 0;
    float val;

    // CPU reductions
    for (j = 0; j < height; ++j)
    {
        float *thisRow = cpuData + (j * width);
        for (i = 0; i < width; ++i)
        {
            val = thisRow[i];
            cpuReducedRows[j] += val;
            cpuReducedColumns[i] += val;
            cpuTotalReduction += val;
        }
    }

    cpuTotalReduction = 0;
    for (i = 0; i < width; ++i) cpuTotalReduction+= cpuReducedColumns[i];

    // GPU data
    PUGBuffer* gpuData = pugAllocateBuffer(width, height, PUG_READ, 1, false);
    pugInitBuffer(gpuData, cpuData);

    // temp storage
    PUGBuffer* gpuTemp = pugAllocateBuffer(width, height, PUG_READWRITE, 1, true);
   
    // the programs
    PUGProgram* sum2Program = pugLoadReductionProgram((path + "pugreduce.cg").c_str(), 
                                                      "ReduceAdd", 2);

    PUGProgram* sum4Program = pugLoadReductionProgram((path + "pugreduce.cg").c_str(),
                                                      "ReduceAdd", 4);
    // GPU Reductions
    pugReduce1D(sum2Program, gpuData, gpuTemp, PUG_DIMENSION_X, height, width);
    PUGRect rectRows(0, 1, 0, height); // read left column of sums
    pugReadMemory(gpuReducedRows, gpuTemp, rectRows);

    pugReduce1D(sum2Program, gpuData, gpuTemp, PUG_DIMENSION_Y, height, width);
    PUGRect rectColumns(0, width, 0, 1); // read left column of sums
    pugReadMemory(gpuReducedColumns, gpuTemp, rectColumns);

    pugReduce2D(sum4Program, gpuData, gpuTemp, height, width);
    PUGRect rectTotal(0, 1, 0, 1); // read bottom left element
    pugReadMemory(&gpuTotalReduction, gpuTemp, rectTotal);

    // Compare and report results
    float errorMax = 0, errorMin = RAND_MAX, errorAvg = 0;
    float errorPerOpAvg = 0;
    float totalSumCPU = 0;
    float totalSumGPU = 0;
    
    // rows
    for (i = 0; i < height; ++i)
    {
        totalSumCPU += cpuReducedRows[i];
        totalSumGPU += gpuReducedRows[i];
        float error = fabs(cpuReducedRows[i] - gpuReducedRows[i]);
        errorAvg += error;
        errorMax = max(error, errorMax);
        errorMin = min(error, errorMin);

    }
    errorAvg /= height;
    errorPerOpAvg = errorAvg / height;
    printf("Total Row Sums      | gpu: %f, cpu: %f\n", totalSumGPU, totalSumCPU);
    printf("Row reduction error | min: %1.4e, max: %1.4e, avg: %1.4e\n", 
           errorMin, errorMax, errorAvg);
    printf("Avg. Per-Operation Error:  %1.4e\n\n", errorPerOpAvg);

    // columns
    errorMax = 0; errorMin = RAND_MAX; errorAvg = 0;
    totalSumCPU = 0;
    totalSumGPU = 0;
    for (i = 0; i < width; ++i)
    {
        totalSumCPU += cpuReducedColumns[i];
        totalSumGPU += gpuReducedColumns[i];
        float error = fabs(cpuReducedColumns[i] - gpuReducedColumns[i]);
        errorAvg += error;
        errorMax = max(error, errorMax);
        errorMin = min(error, errorMin);
    }
    errorAvg /= width;
    errorPerOpAvg = errorAvg / height;
    printf("Total Col Sums      | gpu: %f, cpu: %f\n", totalSumGPU, totalSumCPU);
    printf("Col reduction error | min: %1.4e, max: %1.4e, avg: %1.4e\n", 
           errorMin, errorMax, errorAvg);
    printf("Avg. Per-Operation Error:  %1.4e\n\n", errorPerOpAvg);

    // reduce to single value  
    float error = fabs(cpuTotalReduction - gpuTotalReduction);
    errorPerOpAvg = error / (width * height);
    printf("Total Sums          | gpu: %f, cpu: %f\n", gpuTotalReduction, cpuTotalReduction);
    printf("Avg. Per-Operation Error:  %1.4e\n\n", errorPerOpAvg);
    printf("Total reduction error:     %1.4e\n", error);

    delete [] cpuData;
    delete [] cpuReducedRows;
    delete [] cpuReducedColumns;
    delete [] gpuReducedRows;
    delete [] gpuReducedColumns;

    pugDeleteBuffer(gpuData);
    pugDeleteBuffer(gpuTemp);
}


int main(int argc, char* argv[])
{
    int width = 31;
    int height = 57;
    bool oneTest = false;

    if (argc >= 3)
    {
        width  = atoi(argv[1]);
        height = atoi(argv[2]);
        oneTest = true;
    }

    if (!pugInit(path.c_str())) {
        fprintf(stderr, "Unable to initialize PUG\n");
        exit(1);
    }

    printf("-----------------------------------------------\n");
    printf("TESTING %d-by-%d SUM REDUCTIONS\n", width, height);
    printf("-----------------------------------------------\n");
    testReductions(width, height);
    
    if (!oneTest)
    {
        width = 512;
        height = 256;
        printf("-----------------------------------------------\n\n");
        printf("-----------------------------------------------\n");
        printf("TESTING %d-by-%d SUM REDUCTIONS\n", width, height);
        printf("-----------------------------------------------\n");
        testReductions(width, height);
        printf("-----------------------------------------------\n\n");
    }

    printf("NOTE: Part of the error is introduced by different order of\n"); 
    printf("operations used to implement reductions on the CPU and GPU.\n");
    
    return 0;
}