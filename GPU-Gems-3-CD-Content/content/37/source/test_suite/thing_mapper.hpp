// ************************************************
// thing_mapper.hpp
// authors: Lee Howes and David B. Thomas
//
// A mapper for mapping various things to strings.
// ************************************************

#include "common.hpp"

template < class T > class ThingMapper
{
  private:
	std::vector < char >m_thingType;
	std::vector < boost::shared_ptr < T > >m_things;
  public:
	ThingMapper(const char *thingType):m_thingType(StrToVec(thingType))
	{
	}

	unsigned Count() const
	{
		return m_things.size();
	}

	void Add(boost::shared_ptr < T > thing)
	{
		m_things.push_back(thing);
	}

	void ListNames(FILE * dst)
	{
		for (unsigned i = 0; i < m_things.size(); i++)
		{
			fprintf(dst, "%i : %s\n", i, m_things[i]->Name());
		}
	}

	void List(FILE * dst)
	{
		for (unsigned i = 0; i < m_things.size(); i++)
		{
			fprintf(dst, "%i : %s\n\t%s\n\n", i, m_things[i]->Name(), m_things[i]->Description());
		}
	}

	boost::shared_ptr < T > Get(const char *index)
	{
		char *tmp = 0;
		unsigned v = strtoul(index, &tmp, 0);
		if (tmp != index)
		{
			if (v >= m_things.size())
			{
				fprintf(stderr, "Index %u out of range for %s\n", v, &m_thingType[0]);
			}
			return m_things[v];
		}

		for (unsigned i = 0; i < m_things.size(); i++)
		{
			if (0 == strcmp(index, m_things[i]->Name()))
				return m_things[i];
		}

		fprintf(stderr, "Couldn't find %s with index %s\n", &m_thingType[0], index);
		exit(1);
	}
};
