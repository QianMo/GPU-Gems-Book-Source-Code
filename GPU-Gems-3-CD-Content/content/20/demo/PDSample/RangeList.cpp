/// Provided Courtesy of Daniel Dunbar

#include <math.h>
#include <stdio.h>

#include "RangeList.h"

///

static const float kSmallestRange = .000001f;

RangeList::RangeList(float min, float max)
{
	numRanges = 0;
	rangesSize = 8;
	ranges = new RangeEntry[rangesSize];
	reset(min, max);
}

RangeList::~RangeList()
{
	delete[] ranges;
}

void RangeList::reset(float min, float max)
{
	numRanges = 1;
	ranges[0].min = min;
	ranges[0].max = max;
}

void RangeList::deleteRange(int pos)
{
	if (pos<numRanges-1) {
		memmove(&ranges[pos], &ranges[pos+1], sizeof(*ranges)*(numRanges-(pos+1)));
	}
	numRanges--;
}

void RangeList::insertRange(int pos, float min, float max)
{
	if (numRanges==rangesSize) {
		RangeEntry *tmp = new RangeEntry[rangesSize];
		memcpy(tmp, ranges, numRanges*sizeof(*tmp));
		delete[] ranges;
		ranges = tmp;
	}

	if (pos<numRanges) {
		memmove(&ranges[pos+1], &ranges[pos], sizeof(*ranges)*(numRanges-pos));
	}

	ranges[pos].min = min;
	ranges[pos].max = max;
	numRanges++;
}

void RangeList::subtract(float a, float b)
{
	static const float twoPi = (float) (M_PI*2);

	if (a>twoPi) {
		subtract(a-twoPi, b-twoPi);
	} else if (b<0) {
		subtract(a+twoPi, b+twoPi);
	} else if (a<0) {
		subtract(0, b);
		subtract(a+twoPi,twoPi);
	} else if (b>twoPi) {
		subtract(a, twoPi);
		subtract(0, b-twoPi);
	} else if (numRanges==0) {
		;
	} else {
		int pos;

		if (a<ranges[0].min) {
			pos = -1;
		} else {
			int lo=0, mid=0, hi=numRanges;

			while (lo<hi-1) {
				mid = (lo+hi)>>1;
				if (ranges[mid].min<a) {
					lo = mid;
				} else {
					hi = mid;
				}
			}

			pos = lo;
		}

		if (pos==-1) {
			pos = 0;
		} else if (a<ranges[pos].max) {
			float c = ranges[pos].min;
			float d = ranges[pos].max;
			if (a-c<kSmallestRange) {
				if (b<d) {
					ranges[pos].min = b;
				} else {
					deleteRange(pos);
				}
			} else {
				ranges[pos].max = a;
				if (b<d) {
					insertRange(pos+1, b, d);
				}
				pos++;
			}
		} else {
			if (pos<numRanges-1 && b>ranges[pos+1].min) {
				pos++;
			} else {
				return;
			}
		}

		while (pos<numRanges && b>ranges[pos].min) {
			if (ranges[pos].max-b<kSmallestRange) {
				deleteRange(pos);
			} else {
				ranges[pos].min = b;
			}
		}
	}
}

void RangeList::print()
{
	printf("[");
	for (int i=0; i<numRanges; i++) {
		printf("(%f,%f)%s", ranges[i].min, ranges[i].max, (i==numRanges-1)?"":", ");
	}
	printf("]\n");
}
