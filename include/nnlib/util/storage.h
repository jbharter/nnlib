#ifndef STORAGE_H
#define STORAGE_H

#include <initializer_list>

namespace nnlib
{

/// Unique, contigious storage that manages its own memory.
/// May be shared across multiple objects.
/// Used by tensors.
template <typename T>
class Storage
{
public:
	Storage(size_t n = 0, const T &defaultValue = 0) :
		m_ptr(new T[n]),
		m_size(n),
		m_capacity(n)
	{
		for(size_t i = 0; i < n; ++i)
			m_ptr[i] = defaultValue;
	}
	
	Storage(const Storage &copy) :
		m_ptr(new T[copy.size()]),
		m_size(copy.size()),
		m_capacity(copy.size())
	{
		size_t index = 0;
		for(const T &value : copy)
		{
			m_ptr[index] = value;
			++index;
		}
	}
	
	Storage(const std::initializer_list<T> &values) :
		m_ptr(new T[values.size()]),
		m_size(values.size()),
		m_capacity(values.size())
	{
		size_t index = 0;
		for(const T &value : values)
		{
			m_ptr[index] = value;
			++index;
		}
	}
	
	~Storage()
	{
		delete[] m_ptr;
	}
	
	Storage &operator=(const Storage &copy)
	{
		resize(copy.size());
		size_t index = 0;
		for(const T &value : copy)
		{
			m_ptr[index] = value;
			++index;
		}
		return *this;
	}
	
	Storage &operator=(const std::initializer_list<T> &values)
	{
		resize(values.size());
		size_t index = 0;
		for(const T &value : values)
		{
			m_ptr[index] = value;
			++index;
		}
		return *this;
	}
	
	void resize(size_t n, const T &defaultValue = 0)
	{
		if(n > m_capacity)
		{
			T *ptr = new T[n];
			for(size_t i = 0; i < m_size; ++i)
				ptr[i] = m_ptr[i];
			for(size_t i = m_size; i < n; ++i)
				ptr[i] = defaultValue;
			delete[] m_ptr;
			m_ptr = ptr;
			m_capacity = n;
		}
		m_size = n;
	}
	
	T *ptr()
	{
		return m_ptr;
	}
	
	size_t size() const
	{
		return m_size;
	}
	
	T &operator[](size_t i)
	{
		return m_ptr[i];
	}
	
	const T &operator[](size_t i) const
	{
		return m_ptr[i];
	}
	
	const T *begin() const
	{
		return m_ptr;
	}
	
	const T *end() const
	{
		return m_ptr + m_size;
	}
private:
	T *m_ptr;			///< The data itself.
	size_t m_size;		///< Number of elements being used.
	size_t m_capacity;	///< Number of elements available in buffer.
};

}

#endif