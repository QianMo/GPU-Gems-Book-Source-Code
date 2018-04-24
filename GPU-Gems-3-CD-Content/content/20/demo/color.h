//
// color.h
// Last Updated:		05.01.07
// 
// Mark Colbert & Jaroslav Krivanek
// colbert@cs.ucf.edu
//
// Copyright (c) 2007.
//
// The following code is freely distributed "as is" and comes with 
// no guarantees or required support by the authors.  Any use of 
// the code for commercial purposes requires explicit written consent 
// by the authors.
//


#ifndef COLOR_H
#define COLOR_H

#include <iostream>

/// 3-component color class (NO-PADDING!)
class Color3 {
	public:
		Color3() : r(0.f), g(0.f), b(0.f) {}
		Color3(const float &f) : r(f), g(f), b(f) {}
		Color3(const float &_r, const float &_g, const float &_b) : r(_r), g(_g), b(_b) {}

		inline Color3& operator+=(const Color3 &c)	     { r+=c.r; g+=c.g; b+=c.b; return (*this);				}
		inline Color3& operator-=(const Color3 &c)	     { r-=c.r; g-=c.g; b-=c.b; return (*this);				}
		inline Color3& operator*=(const Color3 &c)	     { r*=c.r; g*=c.g; b*=c.b; return (*this);				}
		inline Color3& operator/=(const Color3 &c)	     { r/=c.r; g/=c.g; b/=c.b; return (*this);				}

		inline Color3  operator+ (const Color3 &c) const { return Color3(r+c.r, g+c.g, b+c.b);					} 
		inline Color3  operator- (const Color3 &c) const { return Color3(r-c.r, g-c.g, b-c.b);					} 
		inline Color3  operator* (const Color3 &c) const { return Color3(r*c.r, g*c.g, b*c.b);					} 
		inline Color3  operator/ (const Color3 &c) const { return Color3(r/c.r, g/c.g, b/c.b);					}

		inline Color3& operator+=(const float &f)		 { r+=f; g+=f; b+=f; return (*this);					}
		inline Color3& operator-=(const float &f)		 { r-=f; g-=f; b-=f; return (*this);					}
		inline Color3& operator*=(const float &f)		 { r*=f; g*=f; b*=f; return (*this);					}
		inline Color3& operator/=(const float &f)		 { r/=f; g/=f; b/=f; return (*this);					}

		inline Color3  operator+ (const float &f) const  { return Color3(r+f, g+f, b+f);							}
		inline Color3  operator- (const float &f) const  { return Color3(r-f, g-f, b-f);							}
		inline Color3  operator* (const float &f) const  { return Color3(r*f, g*f, b*f);							}
		inline Color3  operator/ (const float &f) const  { return Color3(r/f, g/f, b/f);							}

		inline Color3  friend operator+(const float &f, const Color3 &c) { return c+f;							}
		inline Color3  friend operator-(const float &f, const Color3 &c) { return Color3(f-c.r, f-c.g, f-c.b);	}
		inline Color3  friend operator*(const float &f, const Color3 &c) { return c*f;							}
		inline Color3  friend operator/(const float &f, const Color3 &c) { return Color3(f/c.r, f/c.g, f/c.b);	}
		
		inline Color3  operator- ()				 const { return Color3(-r, -g, -b);								}

		friend std::ostream& operator<< (std::ostream& fout, const Color3& c) {
			fout << "( " << c.r << " " << c.g << " " << c.b << " )";
			return fout;
		}

		inline float  y() { return r*0.2126f + g*0.7152f + b*0.0722f; }

		float r,g,b;

	private:
		//char padding[4];
};

/// 4-component color class
class Color4 {
	public:
		Color4() : r(0.f), g(0.f), b(0.f), a(1.f) {}
		Color4(const float &f) : r(f), g(f), b(f), a(1.f) {}
		Color4(const Color3 &c) : r(c.r), g(c.g), b(c.b), a(1.f) {}
		Color4(const float &_r, const float &_g, const float &_b) : r(_r), g(_g), b(_b), a(1.f) {}
		Color4(const float &_r, const float &_g, const float &_b, const float &_a) : r(_r), g(_g), b(_b), a(_a) {}

		inline Color4& operator+=(const Color4 &c)	     { r+=c.r; g+=c.g; b+=c.b; a+=c.a; return (*this);		}
		inline Color4& operator-=(const Color4 &c)	     { r-=c.r; g-=c.g; b-=c.b; a-=c.a; return (*this);		}
		inline Color4& operator*=(const Color4 &c)	     { r*=c.r; g*=c.g; b*=c.b; a*=c.a; return (*this);		}
		inline Color4& operator/=(const Color4 &c)	     { r/=c.r; g/=c.g; b/=c.b; a/=c.a; return (*this);		}

		inline Color4  operator+ (const Color4 &c) const { return Color4(r+c.r, g+c.g, b+c.b);					} 
		inline Color4  operator- (const Color4 &c) const { return Color4(r-c.r, g-c.g, b-c.b);					} 
		inline Color4  operator* (const Color4 &c) const { return Color4(r*c.r, g*c.g, b*c.b);					} 
		inline Color4  operator/ (const Color4 &c) const { return Color4(r/c.r, g/c.g, b/c.b);					}

		inline Color4& operator+=(const float &f)		 { r+=f; g+=f; b+=f; return (*this);					}
		inline Color4& operator-=(const float &f)		 { r-=f; g-=f; b-=f; return (*this);					}
		inline Color4& operator*=(const float &f)		 { r*=f; g*=f; b*=f; return (*this);					}
		inline Color4& operator/=(const float &f)		 { r/=f; g/=f; b/=f; return (*this);					}

		inline Color4  operator+ (const float &f) const  { return Color4(r+f, g+f, b+f);							}
		inline Color4  operator- (const float &f) const  { return Color4(r-f, g-f, b-f);							}
		inline Color4  operator* (const float &f) const  { return Color4(r*f, g*f, b*f);							}
		inline Color4  operator/ (const float &f) const  { return Color4(r/f, g/f, b/f);							}

		inline Color4  friend operator+(const float &f, const Color4 &c) { return c+f;							}
		inline Color4  friend operator-(const float &f, const Color4 &c) { return Color4(f-c.r, f-c.g, f-c.b);	}
		inline Color4  friend operator*(const float &f, const Color4 &c) { return c*f;							}
		inline Color4  friend operator/(const float &f, const Color4 &c) { return Color4(f/c.r, f/c.g, f/c.b);	}
		
		inline Color4  operator- ()				 const { return Color4(-r, -g, -b);								}

		friend std::ostream& operator<< (std::ostream& fout, const Color4& c) {
			fout << "( " << c.r << " " << c.g << " " << c.b << " " << c.a << " )";
			return fout;
		}

		inline float  y() { return r*0.2126f + g*0.7152f + b*0.0722f; }

		float r,g,b,a;
};

typedef Color4 Color;


#endif
