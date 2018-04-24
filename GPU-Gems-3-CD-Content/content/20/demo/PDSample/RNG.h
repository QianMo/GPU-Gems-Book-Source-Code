/// Provided Courtesy of Daniel Dunbar

class RNG {
private:
	/* Period parameters */  
	static const long N = 624;
	static const long M = 397;
	static const unsigned long MATRIX_A = 0x9908b0dfUL;   /* constant vector a */
	static const unsigned long UPPER_MASK = 0x80000000UL; /* most significant w-r bits */
	static const unsigned long LOWER_MASK = 0x7fffffffUL; /* least significant r bits */

private:
	unsigned long mt[N]; /* the array for the state vector  */
	int mti;

public:
	RNG(unsigned long seed=5489UL);
	RNG(unsigned long *init_key, int key_length);

	void seed(unsigned long seed);

		/* generates a random number on [0,0xffffffff]-interval */
	unsigned long getInt32();
		/* generates a random number on [0,0x7fffffff]-interval */
	long getInt31();
		/* generates a random number on [0,1]-real-interval */
	double getDoubleLR();
	float getFloatLR();
		/* generates a random number on [0,1)-real-interval */
	double getDoubleL();
	float getFloatL();
		/* generates a random number on (0,1)-real-interval */
	double getDouble();
	float getFloat();
};
