/// Provided Courtesy of Daniel Dunbar

#include <vector>

typedef struct _RangeEntry {
	float min, max;
} RangeEntry;

class RangeList {
public:
	RangeEntry *ranges;
	int numRanges, rangesSize;

public:
	RangeList(float min, float max);
	~RangeList();

	void reset(float min, float max);

	void print();

	void subtract(float min, float max);

private:
	void deleteRange(int pos);
	void insertRange(int pos, float min, float max);
};
