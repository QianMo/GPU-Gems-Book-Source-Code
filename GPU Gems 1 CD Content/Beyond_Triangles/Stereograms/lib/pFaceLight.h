class pFaceLight : public pArray<int>
{
	public:
	pFaceLight()
	{
	}

	virtual ~pFaceLight()
	{
	}

	int operator==(const pFaceLight& in)
	{
		int i;
		if (num!=in.num)
			return 0;
		for( i=0;i<num;i++ )
			if (buf[i]!=in.buf[i])
				return 0;
		return 1;
	}

	int operator!=(const pFaceLight& in)
	{
		int i;
		if (num!=in.num)
			return 1;
		else
		if (num==0)
			return 1;
		for( i=0;i<num;i++ )
			if (buf[i]!=in.buf[i])
				return 1;
		return 0;
	}

	int compare(const pFaceLight& in)
	{
		int i;
		for( i=0;i<num;i++ )
			if (buf[i]<in.buf[i])
				return -1;
			else
			if (buf[i]>in.buf[i])
				return 1;
		return 0;
	}
};
