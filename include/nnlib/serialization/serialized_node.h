#ifndef SERIALIZED_NODE_H
#define SERIALIZED_NODE_H

#include <map>
#include <string>
#include <vector>
#include "factory.h"
#include "traits.h"

namespace nnlib
{

/// \brief A serialized piece of data.
///
/// SerializedNode is an intermediary value between classes and fully serialized data.
/// Serializers can load nodes from or save nodes to a file.
/// Classes can load in a node or save a node.
/// A node can take one of five value types:
///     null, a number, a string, an array, or an object
/// Numbers are always stored as floating point values, but can be cast to any arithmetic type.
/// Arrays store sequences of node pointers and objects map strings to node pointers.
class SerializedNode
{
public:
	/// A tag indicating a node's data type.
	enum class Type { Null, Number, String, Array, Object };
	
	/// The stored value when type is Array.
	using Array = std::vector<SerializedNode *>;
	
	/// The stored value when type is Object.
	using Object = std::map<std::string, SerializedNode *>;
	
	/// Create a null node.
	SerializedNode() :
		m_type(Type::Null)
	{}
	
	/// Create a non-null node.
	template <typename T>
	SerializedNode(T &&value) :
		m_type(Type::Null)
	{
		set(std::forward<T>(value));
	}
	
	/// Destructor; delete children if type is array or object.
	~SerializedNode()
	{
		if(m_type == Type::Array)
		{
			for(SerializedNode *n : m_array)
				delete n;
		}
		else if(m_type == Type::Object)
		{
			for(auto &nvp : m_object)
				delete nvp.second;
		}
	}
	
	/// Assignment operator.
	SerializedNode &operator=(const SerializedNode &other)
	{
		if(this != &other)
		{
			type(other.m_type);
			switch(m_type)
			{
			case Type::Null:
				break;
			case Type::Number:
				m_number = other.m_number;
				break;
			case Type::String:
				m_string = other.m_string;
				break;
			case Type::Array:
				m_array.clear();
				for(SerializedNode *n : other.m_array)
					m_array.push_back(new SerializedNode(*n));
				break;
			case Type::Object:
				m_object.clear();
				for(auto &it : other.m_object)
					m_object.emplace(it.first, new SerializedNode(*it.second));
				break;
			};
		}
		return *this;
	}
	
	/// Get the current type.
	Type type() const
	{
		return m_type;
	}
	
	/// \brief Set the current type.
	///
	/// Because some of the unioned types are classes, we have to explicitly
	/// call the constructor and destructor when switching between them.
	/// See http://en.cppreference.com/w/cpp/language/union.
	void type(Type type)
	{
		if(type == m_type)
			return;
		
		if(m_type == Type::String)
			m_string.~basic_string<char>();
		if(m_type == Type::Array)
			m_array.~Array();
		else if(m_type == Type::Object)
			m_object.~Object();
		
		if(type == Type::String)
			new (&m_string) std::string;
		if(type == Type::Array)
			new (&m_array) Array;
		else if(type == Type::Object)
			new (&m_object) Object;
		
		m_type = type;
	}
	
	/// Set a number value.
	template <typename T>
	typename std::enable_if<std::is_arithmetic<T>::value>::type set(T value)
	{
		type(Type::Number);
		m_number = value;
	}
	
	/// Set a string value.
	template <typename T>
	typename std::enable_if<std::is_convertible<T, std::string>::value>::type set(const T &value)
	{
		type(Type::String);
		m_string = value;
	}
	
	/// Set an array value.
	template <typename T>
	typename std::enable_if<std::is_convertible<T, Array>::value>::type set(const T &value)
	{
		type(Type::Array);
		m_array = value;
	}
	
	
	/// Set an array value from a pair of iterators.
	template <typename T>
	void set(T i, const T &end)
	{
		Array arr;
		arr.reserve(std::distance(i, end));
		
		while(i != end)
		{
			arr.push_back(new SerializedNode(*i));
			++i;
		}
		
		set(arr);
	}
	
	/// Set an object value.
	template <typename T>
	typename std::enable_if<std::is_convertible<T, Object>::value>::type set(const T &value)
	{
		type(Type::Object);
		m_object = value;
	}
	
	/// Set a serializble value (through a reference).
	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value>::type set(const T &value)
	{
		bool isPolymorphic = true;
		
		try
		{
			set("type", Factory<typename traits::BaseOf<T>::type>::derivedName(typeid(value)));
		}
		catch(const Error &e)
		{
			isPolymorphic = false;
		}
		
		if(isPolymorphic)
		{
			SerializedNode *n = new SerializedNode();
			value.save(*n);
			set("value", n);
			set("polymorphic", true);
		}
		else
		{
			value.save(*this);
		}
	}
	
	/// Set a serializable value (through a pointer).
	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value>::type set(const T *value)
	{
		set(*value);
	}
	
	/// Assignment.
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedNode>::value>::type set(const T &value)
	{
		*this = value;
	}
	
	/// Get a number value.
	template <typename T>
	typename std::enable_if<std::is_arithmetic<T>::value, T>::type as() const
	{
		NNHardAssertEquals(m_type, Type::Number, "Invalid type!");
		return m_number;
	}
	
	/// Get a string value.
	template <typename T>
	typename std::enable_if<std::is_same<T, std::string>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Type::String, "Invalid type!");
		return m_string;
	}
	
	/// Get an array value.
	template <typename T>
	typename std::enable_if<std::is_same<T, Array>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Type::Array, "Invalid type!");
		return m_array;
	}
	
	/// Get an object value.
	template <typename T>
	typename std::enable_if<std::is_same<T, Object>::value, const T &>::type as() const
	{
		NNHardAssertEquals(m_type, Type::Object, "Invalid type!");
		return m_object;
	}
	
	/// Get a serializable value.
	template <typename T>
	typename std::enable_if<traits::HasLoadAndSave<T>::value, T>::type as() const
	{
		T value;
		if(m_type == Type::Object && m_object.find("polymorphic") != m_object.end())
			value.load(get<SerializedNode>("value"));
		else
			value.load(*this);
		return value;
	}
	
	/// Get a polymorphic serializable value (through a pointer).
	template <typename T>
	typename std::enable_if<std::is_pointer<T>::value, T>::type as() const
	{
		NNHardAssert(get<bool>("polymorphic"), "Cannot get pointer to a non-polymorphic type!");
		
		T value = dynamic_cast<T>(Factory<typename traits::BaseOf<typename std::remove_pointer<T>::type>::type>::construct(get<std::string>("type")));
		NNHardAssertNotEquals(value, nullptr, "Failed to get pointer to derived class!");
		
		value->load(get<SerializedNode>("value"));
		return value;
	}
	
	/// Get identity (useful for convenience).
	template <typename T>
	typename std::enable_if<std::is_same<T, SerializedNode>::value, const T &>::type as() const
	{
		return *this;
	}
	
	/// \brief In an object, make a name-value pair.
	///
	/// If the current type is not already object, this will change the type to object.
	void set(const std::string &name, SerializedNode *value)
	{
		type(Type::Object);
		m_object.emplace(name, value);
	}
	
	/// \brief In an object, make a name-value pair.
	///
	/// If the current type is not already object, this will change the type to object.
	template <typename T>
	void set(const std::string &name, const T &value)
	{
		type(Type::Object);
		m_object.emplace(name, new SerializedNode(value));
	}
	
	/// \brief In an object, make an array name-value pair from a pair of iterators.
	///
	/// If the current type is not already object, this will change the type to object.
	template <typename T>
	void set(const std::string &name, T i, const T &end)
	{
		Array arr;
		arr.reserve(std::distance(i, end));
		
		while(i != end)
		{
			arr.push_back(new SerializedNode(*i));
			++i;
		}
		
		set(name, arr);
	}
	
	/// \brief In an object, get a number or serializable type from a name-value pair.
	///
	/// If the current type is not object, this will throw an Error.
	template <typename T>
	typename std::enable_if<std::is_arithmetic<T>::value || traits::HasLoadAndSave<T>::value || std::is_pointer<T>::value, T>::type get(const std::string &name) const
	{
		NNHardAssertEquals(m_type, Type::Object, "Invalid type!");
		
		auto i = m_object.find(name);
		NNHardAssertNotEquals(i, m_object.end(), "No key '" + name + "' in this object!");
		
		return i->second->as<T>();
	}
	
	/// \brief In an object, get a non-number, non-serializable, non-pointer type from a name-value pair.
	///
	/// If the current type is not object, this will throw an Error.
	template <typename T>
	typename std::enable_if<!std::is_arithmetic<T>::value && !traits::HasLoadAndSave<T>::value && !std::is_pointer<T>::value, const T &>::type get(const std::string &name) const
	{
		NNHardAssertEquals(m_type, Type::Object, "Invalid type!");
		
		auto i = m_object.find(name);
		NNHardAssertNotEquals(i, m_object.end(), "No key '" + name + "' in this object!");
		
		return i->second->as<T>();
	}
	
	/// \brief In an object, load a value from a name-value pair into a variable.
	///
	/// If the current type is not object, this will throw an Error.
	template <typename T>
	void get(const std::string &name, T &value) const
	{
		NNHardAssertEquals(m_type, Type::Object, "Invalid type!");
		
		auto i = m_object.find(name);
		NNHardAssertNotEquals(i, m_object.end(), "No key '" + name + "' in this object!");
		
		value = i->second->as<T>();
	}
	
	/// \brief In an object, load an array from a name-value pair into a pair of iterators.
	///
	/// If the current type is not object, this will throw an Error.
	template <typename T>
	void get(const std::string &name, T i, const T &end) const
	{
		NNHardAssertEquals(std::distance(i, end), get<Array>(name).size(), "Invalid range!");
		for(SerializedNode *n : get<Array>(name))
		{
			*i = n->as<typename std::remove_reference<decltype(*i)>::type>();
			++i;
		}
	}
	
	/// Load an array into a pair of iterators.
	template <typename T>
	void get(T i, const T &end) const
	{
		NNHardAssertEquals(m_type, Type::Array, "Invalid type!");
		NNHardAssertEquals(std::distance(i, end), m_array.size(), "Invalid range!");
		for(SerializedNode *n : m_array)
		{
			*i = n->as<typename std::remove_reference<decltype(*i)>::type>();
			++i;
		}
	}
	
private:
	/// A tag indicating the active type.
	Type m_type;
	
	/// The actual data.
	union
	{
		double m_number;      ///< The number value. All numbers are stored as double here.
		std::string m_string; ///< The string value.
		Array m_array;        ///< The array value (a vector of SerializedNodes).
		Object m_object;      ///< The object value (a map of string -> SerializedNodes).
	};
};

}

#endif
