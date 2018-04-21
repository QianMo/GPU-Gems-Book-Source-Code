class pLight
{
	public:
		pVector pos;
		pVector color;
		int flags;
		pString name;

		pAnimation anim_pos;
		pAnimation anim_radius;
		pAnimation anim_color;

	pLight() : 
		pos(0,0,0,100), color(1,1,1), flags(0)
	{ }

	pLight(const pLight& in) : 
		pos(in.pos), color(in.color), 
		flags(in.flags), name(in.name),
		anim_pos(in.anim_pos), 
		anim_radius(in.anim_radius), 
		anim_color(in.anim_color)
	{ }

	void operator=(const pLight& in) 
	{ 
		pos = in.pos;
		color = in.color;
		flags = in.flags;
		name = in.name;
		anim_pos = in.anim_pos; 
		anim_radius = in.anim_radius;
		anim_color = in.anim_color;
	}

	void write(FILE *fp) const;
	void read(FILE *fp);
};
