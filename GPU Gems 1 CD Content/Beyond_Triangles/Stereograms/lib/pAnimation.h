class pAnimation
{
	public:
		int type;
		int keysize;

		int numkey;
		float *keytime;
		float *keyvalue;
		
		float curtime;
		int curtimekey;

	pAnimation();
	virtual ~pAnimation();

	pAnimation(const pAnimation& in);
	void operator=(const pAnimation& in);

	void reset();
	void set_type(int t);
	void set_numkeys(int nk,int zero_mem=1,int keep_old=0);
	float get_maxtime();

	void read(FILE *fp);
	void write(FILE *fp) const;

	void update(float time,float* value);
	void evaluate_bezier(const float *p,int n,float f,float *o) const;
};
