/*
 File: quasisampler_prototype.h
 Quasisampler prototype.

 This is a toy (non-optimized) implementation of the importance sampling
 technique proposed in the paper:
 "Fast Hierarchical Importance Sampling with Blue Noise Properties",
 by Victor Ostromoukhov, Charles Donohue and Pierre-Marc Jodoin,
 to be presented at SIGGRAPH 2004.

 
 Implementation by Charles Donohue,
 Based on Mathematica code by Victor Ostromoukhov.
 Universite de Montreal
 18.08.04

*/


#ifndef QUASISAMPLER_PROTOTYPE_H
#define QUASISAMPLER_PROTOTYPE_H

#include <math.h>
#include <vector>

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))

#define LUT_SIZE 21 // Number of Importance Index entries in the Lookup table.
#define NUM_STRUCT_INDEX_BITS 6 // Number of significant bits taken from F-Code.

#define GOLDEN_RATIO PHI // Phi is the Golden Ratio.
#define PHI     1.6180339887498948482045868343656 // ( 1 + sqrt(5) ) / 2
#define PHI2    2.6180339887498948482045868343656 // Phi squared
#define LOG_PHI 0.48121182505960347 // log(Phi)
#define SQRT5   2.2360679774997896964091736687313 // sqrt(5.0)

// Two-bit sequences.
#define B00 0
#define B10 1
#define B01 2

/// The six tile types.
enum TileType {
  TileTypeA,TileTypeB,TileTypeC,TileTypeD,TileTypeE,TileTypeF
};

/// Simple 2D point and vector type.
class Point2D
{
public:
  double x,y;

  Point2D(){};
  Point2D(const double x, const double y) { this->x=x; this->y=y; }
  Point2D(const double vect[2]) { x=vect[0]; y=vect[1]; }

  Point2D operator+(const Point2D& pt) const{ return Point2D(x+pt.x,y+pt.y); }
  Point2D operator-(const Point2D& pt) const{ return Point2D(x-pt.x,y-pt.y); }
  Point2D operator*(double factor) const{ return Point2D(x*factor,y*factor); }
  Point2D operator/(double factor) const{ return Point2D(x/factor,y/factor); }

  /// Returns the squared distance to the origin, or the squared length of a vector.
  double d2() const { return x*x+y*y; }
};

/// This is a base class that implements the Quasi-Sampler importance sampling
/// system, as presented in the paper :
/// "Fast Hierarchical Importance Sampling with Blue Noise Properties",
/// by Victor Ostromoukhov, Charles Donohue and Pierre-Marc Jodoin,
/// to be presented at SIGGRAPH 2004.
/// This is a pure-virtual class, and you must implement the "getImportanceAt()" function
/// in order to use the sampling system.
/// The mechanics of the system can be observed in the given source code.
class Quasisampler
{

protected:

  //
  // Static tables.
  //


  /// Fibonacci sequence (first 32 numbers).
  static const unsigned fiboTable[32]; // defined at end of file.

  /// Unit vectors rotated around origin, in \f$ \frac{\pi}{10} \f$ increments, 
  /// counter-clockwise. 0 = North.
  /// This table can be used to accelerate the trigonomic operations within the tile
  /// subdivision process, since all angles can only take these values.
  static const Point2D vvect[20]; // defined at end of file.

  /// Pre-calculated correction vectors lookup table.
  /// These are available in ASCII format on the web-site.
  static const double lut[LUT_SIZE][21][2]; // defined at end of file.



  //
  //  Static functions.
  //

  /// Fibonacci number at a given position.
  /// The value returned is \f$ F_i = F_{i-1} + F_{i-2}  \f$.
  static unsigned fibonacci(unsigned i)
  {
    if (i<1) return 1;
    if (i<=32) return fiboTable[i-1]; // pre-calculated.
    return fibonacci(i-1)+fibonacci(i-2);
  }

  /// Returns the required level of subdivision for a given importance value.
  /// The value returned is \f$ \lceil{\log_{\phi^2}(importance)}\rceil \f$,
  /// where \f$ \phi=\frac{1 + {\sqrt{5}}}{2}\f$  is the Golden Ratio.
  static unsigned getReqSubdivisionLevel( unsigned importance )
  {
    if (importance==0) return 0;
    unsigned nbits = (unsigned)(log( (double)importance*SQRT5 + 1.0 ) / LOG_PHI) - 1;
    if (nbits<1) nbits = 1;
    return (unsigned)ceil(0.5*nbits);
  }  

  /// Returns the decimal value of an F-Code, over a given number of bits.
  /// The value returned is \f$ \sum_{j=2}^{m} b_{j} F_{j} \f$.
  static unsigned calcFCodeValue(unsigned bitsequence,unsigned nbits)
  {
    unsigned i_s = 0;
    for (unsigned i=0; i<nbits ;i++ )
    {
      if ( bitsequence & ( 1u<<(nbits-i-1) ) ) i_s += fibonacci(i+2);
    }
    return i_s;
  }


  /// Returns the Structural Index (i_s) for a given F-Code.
  static unsigned calcStructuralIndex(unsigned bitsequence)
  {
    return calcFCodeValue(bitsequence,NUM_STRUCT_INDEX_BITS);
  }

  /// Returns the Importance Index (i_v) for a given importance value.
  /// The value returned is \f$ \lfloor n \cdot ({\log_{\phi^2} \sqrt{5} \cdot x}) ~ {\bf mod} ~ 1 \rfloor \f$.
  static unsigned calcImportanceIndex( unsigned importance )
  {
    double t = log(1.0 + sqrt(5.0)*importance) / log(PHI2);
    t -= floor(t); // modulo 1.0
    return (unsigned)(LUT_SIZE*t); 
  }


  /// Fetches the appropriate vector from the lookup table.
  static Point2D calcDisplacementVector(unsigned importance, unsigned f_code, int dir)
  {
    unsigned i_s = calcStructuralIndex(f_code);
    unsigned i_v = calcImportanceIndex(importance);

    return  
      vvect[dir]        * lut[i_v][i_s][0] + // u component
      vvect[(dir+5)%20] * lut[i_v][i_s][1] ; // v component
  }  


  //
  // Inner classes.
  //


  /// Individual tile elements, which also serve as nodes for the tile subdivision tree.
  class TileNode
  {

    unsigned level; // Depth in the tree.
    int tileType; // Types A through F.
    int dir; // Tile orientation, 0=North, in Pi/10 increments, CCW.
    double scale; 
    Point2D p1,p2,p3; // Three points of the triangle. Counter-clockwise.

    /// The F-Code binary sequence.
    unsigned f_code; 
    
    // tiling tree structure
    TileNode* parent;
    unsigned parent_slot; // position in parent's list (needed for iterators)
    bool terminal; // true for leaf nodes
    std::vector<TileNode*> children;

  public:

    /// Builds a tile according to the given specifications.
    TileNode(  
      TileNode* parent = NULL, 
      int tileType = TileTypeF,
      Point2D refPt = Point2D(0,0),
      int dir = 15, // 15 = East.
      unsigned newbits = 0,
      int parent_slot = 0,
      double scale = 1.0)
    {
      this->parent = parent;
      this->tileType = tileType;
      this->p1 = refPt;
      this->dir = dir%20;
      this->parent_slot = parent_slot;
      this->scale = scale;
      this->level = parent ? parent->level + 1 : 0; // Increment the level.


      // Build triangle, according to type.
      switch(tileType)
      {
      case TileTypeC:
      case TileTypeD:
        // "Skinny" triangles
        p2 = p1 + vvect[dir%20]*scale;
        p3 = p1 + vvect[(dir+4)%20]*(PHI*scale);
        break;
      case TileTypeE:
      case TileTypeF:
        // "Fat" triangles
        p2 = p1 + vvect[dir%20]*(PHI2*scale);
        p3 = p1 + vvect[(dir+2)%20]*(PHI*scale);
        break;
      default:
        // Pentagonal tiles (triangle undefined)
        p2 = p1 + vvect[dir%20]*scale;
        p3 = p1 + vvect[(dir+5)%20]*scale;
      }
      
      // Append 2 new bits to the F-Code.
      if (parent)
        f_code = (parent->f_code<<2)^newbits;
      else 
        f_code = newbits;
      
      // Set as leaf node
      terminal = true;
      children.clear();
    }

    /// Helper constructor.
    /// Creates an initial tile that is certain to contain the ROI.
    /// The starting tile is of type F (arbitrary).
    TileNode( double roi_width,  double roi_height)
    {
      double side = MAX(roi_width,roi_height);
      double scale = 2.0 * side;
      Point2D offset(PHI*PHI/2.0-0.25,0.125);
      *this = TileNode(NULL, TileTypeF,offset*-scale,15,0,0,scale);
    }
    
    ~TileNode() { collapse(); }

    /// Splits a tile according to the given subdivision rules.
    /// Please refer to the code for further details.
    void refine()
    {
      if (!terminal) return; // Can only subdivide leaf nodes.
      
      terminal=false; // The tile now has children.
      double newscale = scale / GOLDEN_RATIO; // The scale factor between levels is constant.
      
      switch(tileType)
      {

      // Each new tile is created using the following information:
      // A pointer to its parent, the type of the new tile (a through f),
      // the origin of the new tile, the change in orientation of the new tile with
      // respect to the parent's orientation, the two bits to be pre-pended to the F-Code,
      // the parent's slot (for traversal purposes), and the new linear scale of the tile,
      // which is always the parent's scale divided by the golden ratio.

      case TileTypeA:
        children.push_back(
          new TileNode(this, TileTypeB, p1, dir+0, B00, 0, newscale));
        break;

      case TileTypeB:
        children.push_back(
          new TileNode(this, TileTypeA, p1, dir+10, B00, 0, newscale) );
        break;
        
      case TileTypeC:
        children.push_back(
          new TileNode(this, TileTypeF, p3, dir+14, B00, 0, newscale) );
        children.push_back(
          new TileNode(this, TileTypeC, p2, dir+6, B10, 1, newscale) );      
        children.push_back(
          new TileNode(this, TileTypeA, children[0]->p3, dir+1, B10, 2, newscale) );
        break;

      case TileTypeD:
        children.push_back(
          new TileNode(this, TileTypeE, p2, dir+6, B00, 0, newscale) );
        children.push_back(
          new TileNode(this, TileTypeD, children[0]->p3, dir+14, B10, 1, newscale) );
        break;

      case TileTypeE:
        children.push_back(
          new TileNode(this, TileTypeC, p3, dir+12, B10, 0, newscale) );
        children.push_back(
          new TileNode(this, TileTypeE, p2, dir+8 , B01, 1, newscale) );
        children.push_back(
          new TileNode(this, TileTypeF, p1, dir+0 , B00, 2, newscale) );
        children.push_back(
          new TileNode(this, TileTypeA, children[0]->p2, dir+7, B10, 3, newscale) );
        break;

      case TileTypeF:
        children.push_back(
          new TileNode(this, TileTypeF, p3, dir+12, B01, 0, newscale) );
        children.push_back(
          new TileNode(this, TileTypeE, children[0]->p3, dir+0, B00, 1, newscale) );
        children.push_back(
          new TileNode(this, TileTypeD, children[1]->p3, dir+8, B10, 2, newscale) );
        children.push_back(
          new TileNode(this, TileTypeA, children[0]->p3, dir+15, B01, 3, newscale) );
        break;
      }
    }

    /// Prunes the subdivision tree at this node.
    void collapse()
    {
      // Recursively prune the tree.
      for (unsigned i=0; i<children.size(); i++) delete children[i];
      terminal = true;
      children.clear();
    }

    /// Returns the next node of the tree, in depth-first traversal.
    /// Returns NULL if it is at the last node.
    TileNode* nextNode()
    {
      if (!terminal) return children[0];
      
      if (level == 0) return NULL; // single node case.

      if ( parent_slot < parent->children.size()-1 )
        return parent->children[parent_slot+1];

      // last child case
      TileNode* tmp = this;
      do
      {
        tmp = tmp->parent;
      }
      while ( (tmp->level != 0) && (tmp->parent_slot ==  tmp->parent->children.size()-1) );

      if (tmp->level == 0) return NULL; // last node
      return tmp->parent->children[tmp->parent_slot+1];

    }

    /// Returns the next closest leaf to a node.
    /// Returns NULL if it's the last leaf.
    TileNode* nextLeaf()
    {
      TileNode* tmp = this;
      do
      {
        tmp = tmp->nextNode();
        if ( !tmp ) return NULL;
        if ( tmp->terminal ) return tmp;
      }
      while (1);
    }

    // Public accessors

    Point2D getP1() const { return p1; }
    Point2D getP2() const { return p2; }
    Point2D getP3() const { return p3; }
    Point2D getCenter() const { return (p1+p2+p3)/3.0; }
    unsigned getFCode() const { return f_code; }
    bool isSamplingType() const {
      return ( (tileType == TileTypeA) || (tileType == TileTypeB) ); }
    unsigned getLevel() { return level; }
    bool isTerminal() const { return terminal; }
    TileNode* getParent() { return parent; }
    TileNode* getChild(unsigned i) {  return children[i]; }

    /// Obtains the correction vector from the lookup table,
    /// then scales and adds it to the reference point.
    Point2D getDisplacedSamplingPoint(unsigned importance)
    {
      return p1 + calcDisplacementVector(importance,f_code,dir) * scale;
    }  

  }; // end of class TileNode.

  /// Leaf iterator for the tile subdivision tree.
  /// The traversal is made in a depth-first manner.
  /// Warning: This does not behave like STL style iterators.
  class TileLeafIterator
  {
    TileNode* shape;
  public:
    TileLeafIterator() { shape=NULL; }
    TileLeafIterator(TileNode* s ) { begin(s); }

    TileNode* operator*() { return shape; }
    TileNode* operator->() { return shape; }

    void begin(TileNode* s)
    {
      TileNode* tmp = s;  
      while ( ! tmp->isTerminal() ) tmp = tmp->getChild(0); // find first leaf
      shape = tmp;
    }

    /// Subdivides the tile and moves to its 1st child.
    void refine() 
    {
      shape->refine();
      shape = shape->getChild(0);
    }

    /// Prunes the subdivision tree.
    void collapse()
    {
      if (shape->getParent())
      {
        shape = shape->getParent();
        shape->collapse();
      }
    }

    /// Moves to the next node in the subdivision tree, in depth-first traversal.
    /// Returns false iff there is no such node.
    bool next()
    {
      TileNode* s = shape->nextLeaf();
      if (s)
      {
        shape = s;
        return true;
      }
      else
      {
        shape = s;
        return false;
      }
    }

    /// Checks if there is a next tile, in depth-first traversal.
    bool hasNext()
    {
      TileNode* s = shape->nextLeaf();
      if (s) return true;
      else return false;
    }
  };


  //
  // Instance members.
  //

  /// Root node of the tile subdivision tree.
  TileNode *root;
  
  /// Extents of the region of interest.
  double width, height;

  /// Protected constructor, which initializes the Region of Interest.
  Quasisampler(double width=0.0, double height=0.0) 
  { this->width=width; this->height=height; root=NULL; }
  
  virtual ~Quasisampler() { if (root) delete root; }



  /// This is a helper function which constrains the incoming points
  /// to the region of interest.
  unsigned getImportanceAt_bounded(Point2D pt)
  {
    if (pt.x>=0 && pt.x<width && pt.y>=0 && pt.y<height) 
      return getImportanceAt(pt);
    else 
      return 0;
  }

  /// Subdivides all tiles down a level, a given number of times.
  void subdivideAll(int times=1)
  {
    if (!root) return;
    TileNode *tmp;
    for (int i=0;i<times;i++)  
    {
      TileLeafIterator it(root);
      do { 
        tmp = *it;
        it.next();
        tmp->refine();
      }
      while (*it);
    }
  }

  /// Generates the hierarchical structure.
  void buildAdaptiveSubdivision( unsigned minSubdivisionLevel = 6 )
  {
    root = new TileNode(width,height);
    
    // Since we are approximating the MAX within each tile by the values at
    // a few key points, we must provide a sufficiently dense initial
    // tiling. This would not be necessary with a more thorough scan of each
    // tile.
    subdivideAll(minSubdivisionLevel);

    TileLeafIterator it(root);
    TileNode *tmp;

    // Recursively subdivide all triangles until each triangle's
    // required level is reached.
    unsigned level;
    do {
      level = it->getLevel();
      
      if ( it->isSamplingType() ) // Sampling tiles are infinitesimal
      {
        if ( level < getReqSubdivisionLevel(getImportanceAt_bounded(it->getP1())) )
        {
          tmp = *it;
          tmp->refine();
        }
      }
      else
      {
        if ( 
          ( level < getReqSubdivisionLevel(getImportanceAt_bounded(it->getP1())) ) ||
          ( level < getReqSubdivisionLevel(getImportanceAt_bounded(it->getP2())) ) ||
          ( level < getReqSubdivisionLevel(getImportanceAt_bounded(it->getP3())) ) ||
          ( level < getReqSubdivisionLevel(getImportanceAt_bounded(it->getCenter())) )
          )
        {
          tmp = *it;
          tmp->refine();
        }
      }
    } while ( it.next() );
  }


  /// Collect the resulting point set.
  void collectPoints( 
    std::vector<Point2D> &pointlist,
    bool filterBounds = true )
  {
    pointlist.clear();

    Point2D pt, pt_displaced;
    unsigned importance;
    TileLeafIterator it(root);
    do {
      pt = it->getP1();  
      if ( it->isSamplingType() ) // Only "pentagonal" tiles generate sampling points.
      {   
        importance = getImportanceAt_bounded( pt );

        // Threshold the function against the F-Code value.           
        if ( importance >= calcFCodeValue( it->getFCode() , 2*it->getLevel() ) )
        {
          // Get the displaced point using the lookup table.
          pt_displaced = it->getDisplacedSamplingPoint(importance);

          if ( !filterBounds ||
            (pt_displaced.x>=0 && pt_displaced.x<width && 
            pt_displaced.y>=0 && pt_displaced.y<height) )
          {
            pointlist.push_back(pt_displaced); // collect point.
          }
        }
      }

    } while ( it.next() );

  }

public:

  /// This virtual function must be implemented in order to use the sampling system.
  /// It should return the value of the importance function at the given point.
  virtual unsigned getImportanceAt( Point2D pt ) = 0;

  /// Builds and collects the point set generated be the sampling system,
  /// using the previously defined importance function.
  std::vector<Point2D> getSamplingPoints()
  {
    if (root) delete root;
    std::vector<Point2D> pointlist;
    
    buildAdaptiveSubdivision(); 
    collectPoints(pointlist);
    return pointlist;
  }

  /// \example example.cpp
  /// This a simple example of how to use the Quasisampler class.

  /// \example example2.cpp
  /// This another simple example of how to use the Quasisampler class.

  /// \example example3.cpp
  /// This example shows how to use the system on a grayscale image.
}; 



/*

Static Member initialization

*/

const unsigned Quasisampler::fiboTable[32]=
{ 1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,
1597,2584,4181,6765,10946,17711,28657,46368,75025,
121393,196418,317811,514229,832040,1346269,2178309 };

const Point2D Quasisampler::vvect[]={
  Point2D(0,1), Point2D(-0.309017,0.951057), Point2D(-0.587785,0.809017),
  Point2D(-0.809017,0.587785), Point2D(-0.951057,0.309017), Point2D(-1,0),
  Point2D(-0.951057,-0.309017), Point2D(-0.809017,-0.587785),
  Point2D(-0.587785,-0.809017), Point2D(-0.309017,-0.951057), Point2D(0,-1),
  Point2D(0.309017,-0.951057), Point2D(0.587785,-0.809017), Point2D(0.809017,-0.587785),
  Point2D(0.951057,-0.309017), Point2D(1,0), Point2D(0.951057,0.309017),
  Point2D(0.809017,0.587785), Point2D(0.587785,0.809017), Point2D(0.309017,0.951057)
};

const double Quasisampler::lut[LUT_SIZE][21][2] =
{{{0.0130357, 0.0419608}, {-0.0241936, 0.0152706}, {-0.00384601, -0.311212}, {-0.000581893, -0.129134}, 
  {-0.0363269, 0.0127624}, {0.0999483, 0.408639}, {-0.0526517, 0.4385}, {-0.128703, 0.392}, {0.0132026, 1.0818}, 
  {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, 
 {{0.00793289, 0.0148063}, {0.0206067, -0.0809589}, {0.0110103, -0.430433}, {0.0000473169, -0.293185}, 
  {-0.0593578, 0.019457}, {0.34192, 0.291714}, {-0.286696, 0.386017}, {-0.345313, 0.311961}, {0.00606029, 1.00877}, 
  {0.04757, 0.05065}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, 
 {{0.00454493, -0.00805726}, {0.0545058, -0.140953}, {0.00960599, -0.493483}, {0.000527191, -0.354496}, 
  {-0.0742085, -0.0477178}, {0.436518, 0.218493}, {-0.422435, 0.275524}, {-0.425198, 0.257027}, 
  {0.0127468, 0.979585}, {0.128363, 0.139522}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
  {0, 0}, {0, 0}, {0, 0}}, {{-0.0014899, -0.0438403}, {0.122261, -0.229582}, {-0.00497263, -0.580537}, 
  {-0.00489546, -0.424237}, {-0.107601, -0.133695}, {0.526304, 0.125709}, {-0.558461, 0.0679206}, 
  {-0.511708, 0.153397}, {0.0271526, 0.950065}, {0.298021, 0.327582}, {-0.00464701, -0.00362132}, {0, 0}, {0, 0}, 
  {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, 
 {{-0.0182024, -0.0837012}, {0.226792, -0.318088}, {-0.0416745, -0.663614}, {-0.0253331, -0.455424}, 
  {-0.159087, -0.20807}, {0.552691, 0.0525824}, {-0.617244, -0.197362}, {-0.561762, 0.00314535}, 
  {0.0522991, 0.928754}, {0.376689, 0.429912}, {-0.0180693, -0.00792235}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
  {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, {{-0.0308901, -0.108719}, {0.362157, -0.377329}, 
  {-0.0918077, -0.742776}, {-0.0571567, -0.453854}, {-0.242014, -0.230347}, {0.542952, -0.00542364}, 
  {-0.614735, -0.35591}, {-0.565238, -0.204834}, {0.084241, 0.900632}, {0.403207, 0.481046}, 
  {-0.0459391, -0.00743248}, {0.0143212, 0.0776031}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
  {0, 0}, {0, 0}}, {{-0.0429758, -0.112222}, {0.470514, -0.41007}, {-0.139291, -0.797567}, {-0.0930261, -0.382258}, 
  {-0.30831, -0.210972}, {0.504387, -0.05265}, {-0.578917, -0.4354}, {-0.545885, -0.40618}, {0.122368, 0.852639}, 
  {0.377534, 0.476884}, {-0.0712593, 0.0238995}, {0.0349156, 0.248696}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
  {0, 0}, {0, 0}, {0, 0}, {0, 0}}, {{-0.0297026, -0.0818903}, {0.514634, -0.426843}, {-0.161039, -0.817284}, 
  {-0.099245, -0.221824}, {-0.359506, -0.135015}, {0.433957, -0.0878639}, {-0.541453, -0.46714}, 
  {-0.526484, -0.556459}, {0.1735, 0.771396}, {0.353023, 0.455358}, {-0.07854, 0.0885735}, {0.0714601, 0.591673}, 
  {-0.0147015, 0.0839976}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, 
 {{-0.0204607, -0.0433266}, {0.515056, -0.428386}, {-0.153717, -0.803384}, {-0.0874438, 0.032819}, 
  {-0.370233, 0.00469937}, {0.331072, -0.0951004}, {-0.507368, -0.487422}, {-0.533403, -0.648977}, 
  {0.243233, 0.652577}, {0.33663, 0.406983}, {-0.0624495, 0.167064}, {0.0527702, 0.808443}, {-0.0444704, 0.258347}, 
  {0.030331, -0.00128903}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, 
 {{-0.0184965, 0.00557424}, {0.495666, -0.40889}, {-0.136052, -0.781115}, {-0.0493628, 0.265293}, 
  {-0.337945, 0.202038}, {0.193353, -0.0835904}, {-0.479971, -0.497456}, {-0.574003, -0.71938}, 
  {0.32445, 0.514949}, {0.331709, 0.341565}, {-0.034108, 0.244375}, {0.0149632, 0.910353}, {-0.104428, 0.60938}, 
  {0.0948414, -0.00216379}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, 
 {{-0.0436899, 0.0294207}, {0.469933, -0.372015}, {-0.153852, -0.756531}, {0.00920944, 0.393625}, 
  {-0.270292, 0.392355}, {0.0540646, -0.0473047}, {-0.466651, -0.492248}, {-0.647575, -0.793479}, 
  {0.394352, 0.385016}, {0.330852, 0.272582}, {-0.0125759, 0.30811}, {-0.0407447, 0.902855}, {-0.136947, 0.8021}, 
  {0.227048, -0.0014045}, {0.0261797, 0.0109521}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, 
 {{-0.0602358, 0.0215278}, {0.43301, -0.338538}, {-0.233311, -0.71494}, {0.0916642, 0.433266}, 
  {-0.173199, 0.474801}, {-0.0384285, 0.024931}, {-0.475596, -0.469989}, {-0.739327, -0.866143}, 
  {0.440049, 0.277063}, {0.326099, 0.207864}, {-0.00488013, 0.365323}, {-0.0890991, 0.872087}, 
  {-0.159106, 0.889116}, {0.311406, 0.0126425}, {0.081674, 0.0403966}, {0.01391, 0.00573611}, {0, 0}, {0, 0}, 
  {0, 0}, {0, 0}, {0, 0}}, {{-0.0723894, -0.00927744}, {0.354855, -0.326512}, {-0.329593, -0.647058}, 
  {0.169384, 0.42962}, {-0.0250381, 0.472328}, {-0.108748, 0.122704}, {-0.507741, -0.424372}, 
  {-0.805866, -0.896362}, {0.48306, 0.211626}, {0.314407, 0.142681}, {-0.00348365, 0.415081}, 
  {-0.125494, 0.836485}, {-0.183247, 0.847226}, {0.366439, 0.0391043}, {0.18978, 0.100287}, {0.0401008, 0.018797}, 
  {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, {{-0.0748666, -0.0517059}, {0.237999, -0.333105}, 
  {-0.391007, -0.558425}, {0.223599, 0.428175}, {0.159284, 0.420084}, {-0.17834, 0.234411}, {-0.553952, -0.353981}, 
  {-0.821481, -0.848098}, {0.527132, 0.175271}, {0.312397, 0.0908259}, {0.00190795, 0.441568}, 
  {-0.149358, 0.790424}, {-0.226469, 0.765995}, {0.383259, 0.0740479}, {0.243694, 0.15335}, {0.0901877, 0.0475938}, 
  {-0.00963625, 0.00819101}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}, 
 {{-0.0862318, -0.0937052}, {0.132383, -0.310846}, {-0.420153, -0.463782}, {0.261956, 0.440763}, 
  {0.290379, 0.392449}, {-0.264095, 0.349189}, {-0.576491, -0.274722}, {-0.797096, -0.724963}, 
  {0.565701, 0.153393}, {0.315376, 0.0546255}, {0.0149326, 0.430477}, {-0.167772, 0.702404}, {-0.283244, 0.645617}, 
  {0.383304, 0.0988087}, {0.248786, 0.17877}, {0.103708, 0.0729573}, {-0.0286781, 0.0298329}, 
  {-0.00878083, 0.0189161}, {0, 0}, {0, 0}, {0, 0}}, 
 {{-0.0911025, -0.116785}, {0.058151, -0.268943}, {-0.424486, -0.374671}, {0.288764, 0.470621}, 
  {0.362681, 0.386055}, {-0.327219, 0.436709}, {-0.585384, -0.202215}, {-0.772145, -0.5936}, {0.580061, 0.135496}, 
  {0.313963, 0.0305349}, {0.0109925, 0.360967}, {-0.181933, 0.552414}, {-0.300836, 0.508161}, 
  {0.364265, 0.0976394}, {0.210088, 0.176749}, {0.096516, 0.0958074}, {-0.0658733, 0.0731591}, 
  {-0.0280071, 0.057776}, {0.0158411, 0.00325704}, {0, 0}, {0, 0}}, 
 {{-0.0974734, -0.0918732}, {0.0139633, -0.212455}, {-0.406371, -0.282796}, {0.296357, 0.483457}, 
  {0.381376, 0.39536}, {-0.333854, 0.503081}, {-0.58254, -0.14516}, {-0.763625, -0.49765}, {0.567887, 0.121286}, 
  {0.30413, 0.0127316}, {-0.00152308, 0.270083}, {-0.191895, 0.352083}, {-0.283727, 0.35145}, 
  {0.326415, 0.0742237}, {0.163984, 0.15982}, {0.0726181, 0.108651}, {-0.0800514, 0.114725}, 
  {-0.0673361, 0.138093}, {0.0402953, 0.00961117}, {-0.0193168, 0.0236477}, {0, 0}}, 
 {{-0.0790912, -0.0163216}, {-0.00448123, -0.162101}, {-0.352873, -0.196134}, {0.271462, 0.449512}, 
  {0.35836, 0.383875}, {-0.286884, 0.565229}, {-0.550438, -0.0846486}, {-0.75899, -0.42121}, {0.528606, 0.119818}, 
  {0.280538, 0.00168322}, {-0.0349212, 0.150096}, {-0.171099, 0.193366}, {-0.250974, 0.211407}, 
  {0.280682, 0.0548899}, {0.126017, 0.143427}, {0.0562988, 0.110436}, {-0.0785227, 0.145239}, 
  {-0.0937526, 0.190149}, {0.0791086, 0.0227095}, {-0.0545744, 0.0707386}, {0, 0}}, 
 {{-0.0518157, 0.0510771}, {-0.00760212, -0.128097}, {-0.253754, -0.111841}, {0.205436, 0.354864}, 
  {0.295866, 0.325402}, {-0.192075, 0.64807}, {-0.4774, -0.00676484}, {-0.722069, -0.332801}, {0.470923, 0.131373}, 
  {0.244358, -0.00366888}, {-0.0555535, 0.0625726}, {-0.128642, 0.0933316}, {-0.239777, 0.136585}, 
  {0.234046, 0.0562388}, {0.105223, 0.134278}, {0.0497268, 0.106459}, {-0.0606163, 0.175207}, 
  {-0.106271, 0.232174}, {0.0538097, 0.0296093}, {-0.122383, 0.16238}, {-0.0113815, 0.0340113}}, 
 {{-0.0304857, 0.0883196}, {0.00193379, -0.129688}, {-0.148195, -0.0572436}, {0.128477, 0.258454}, 
  {0.18546, 0.230594}, {-0.120249, 0.694404}, {-0.326488, 0.130702}, {-0.599671, -0.166452}, {0.371228, 0.215584}, 
  {0.18765, -0.00862734}, {-0.0530754, 0.00501476}, {-0.0781737, 0.0495139}, {-0.215913, 0.0922068}, 
  {0.202485, 0.0708782}, {0.103985, 0.125369}, {0.0553649, 0.1009}, {-0.0397036, 0.199708}, {-0.0966645, 0.253069}, 
  {-0.0153489, 0.0350904}, {-0.134291, 0.193388}, {-0.0315258, 0.0780417}}, 
 {{-0.00909437, 0.0971829}, {0.00766774, -0.145809}, {-0.0755563, -0.0337505}, {0.0700629, 0.188928}, 
  {0.109764, 0.175155}, {-0.084045, 0.707208}, {-0.200288, 0.246694}, {-0.431284, 0.0136518}, {0.274276, 0.314326}, 
  {0.138397, -0.0136486}, {-0.033298, -0.019655}, {-0.0429267, 0.0341841}, {-0.195447, 0.0692005}, 
  {0.188428, 0.0886883}, {0.112392, 0.115937}, {0.0568682, 0.0920568}, {-0.0238131, 0.214855}, 
  {-0.0754228, 0.259851}, {-0.0881413, 0.0371697}, {-0.127762, 0.194639}, {-0.0700573, 0.173426}}};


#endif //QUASISAMPLER_PROTOTYPE_H

