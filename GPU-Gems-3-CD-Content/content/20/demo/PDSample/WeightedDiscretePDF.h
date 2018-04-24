/// Provided Courtesy of Daniel Dunbar


template <class T>
class WDPDF_Node
{
private:
	bool m_mark;

public:
	WDPDF_Node<T> *parent, *left, *right;
	T key;
	float weight, sumWeights;

public:
	WDPDF_Node(T key_, float weight_, WDPDF_Node<T> *parent_);
	~WDPDF_Node();

	WDPDF_Node<T> *sibling() { return this==parent->left?parent->right:parent->left; }

	void markRed() { m_mark = true; }
	void markBlack() { m_mark = false; }
	bool isRed() { return m_mark; }
	bool isBlack() { return !m_mark; }
	bool leftIsBlack() { return !left || left->isBlack(); }
	bool rightIsBlack() { return !right || right->isBlack(); }
	bool leftIsRed() { return !leftIsBlack(); }
	bool rightIsRed() { return !rightIsBlack(); }
	void setSum() { sumWeights = weight + (left?left->sumWeights:0) + (right?right->sumWeights:0); }
};

template <class T>
class WeightedDiscretePDF
{
private:
	WDPDF_Node<T> *m_root;

public:
	WeightedDiscretePDF();
	~WeightedDiscretePDF();

	void insert(T item, float weight);
	void update(T item, float newWeight);
	void remove(T item);
	bool inTree(T item);
	
		/* pick a tree element according to its
		 * weight. p should be in [0,1).
		 */
	T choose(float p);

private:
	WDPDF_Node<T> **lookup(T item, WDPDF_Node<T> **parent_out);
	void split(WDPDF_Node<T> *node);
	void rotate(WDPDF_Node<T> *node);
	void lengthen(WDPDF_Node<T> *node);
	void propogateSumsUp(WDPDF_Node<T> *n);
};

#include "WeightedDiscretePDF.cpp"
