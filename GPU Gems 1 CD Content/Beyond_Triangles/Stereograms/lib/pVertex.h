class pVertex 
{
public:
	float pos[3];
	float norm[3];
	float tx[2];

	void operator=(const pVertex& in)
	{
		pos[0]=in.pos[0];
		pos[1]=in.pos[1];
		pos[2]=in.pos[2];
		norm[0]=in.norm[0];
		norm[1]=in.norm[1];
		norm[2]=in.norm[2];
		tx[0]=in.tx[0];
		tx[1]=in.tx[1];
	}
};
