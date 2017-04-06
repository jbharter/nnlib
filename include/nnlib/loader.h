#ifndef LOADER_H
#define LOADER_H

#include <string>
#include <string.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include "matrix.h"
#include "error.h"

namespace nnlib
{

template <typename T = double>
class Loader
{
public:
	struct Relation
	{
		std::string name;
		std::vector<std::string> attrNames;
		std::vector<std::unordered_map<std::string, size_t>> attrVals;
	};
	static T unknown;
	
	/// Load a weka .arff file.
	static Matrix<T> loadArff(const std::string &filename, Relation *relPtr = nullptr)
	{
		Vector<Tensor<T> *> rows;
		Relation rel;
		
		std::ifstream fin(filename.c_str());
		NNHardAssert(fin.is_open(), "Could not open file '" + filename + "'!");
		
		std::string line;
		while(!fin.fail())
		{
			std::getline(fin, line);
			NNHardAssert(line[0] == '\0' || line[0] == '@' || line[0] == '%', "Invalid arff file!");
			
			if(line[0] == '@')
			{
				if(startsWith(line, "@relation"))
				{
					char *ptr = const_cast<char *>(line.c_str());
					skipWhitespace(&ptr);
					char *end = tokenEnd(ptr);
					rel.name = std::string(ptr, end - ptr);
				}
				else if(startsWith(line, "@attribute"))
				{
					char *ptr = const_cast<char *>(line.c_str() + 10);
					skipWhitespace(&ptr);
					char *end = tokenEnd(ptr);
					rel.attrNames.push_back(std::string(ptr, end - ptr));
					
					ptr = end;
					skipWhitespace(&ptr);
					
					std::unordered_map<std::string, size_t> attrVals;
					if(*ptr == '{')
					{
						++ptr;
						size_t val = 0;
						while(*ptr != '}' && *ptr != '\0')
						{
							skipWhitespace(&ptr);
							char *end = tokenEnd(ptr, ",}");
							attrVals[std::string(ptr, end - ptr)] = val++;
							ptr = end;
							skipWhitespace(&ptr);
							if(*ptr != '\0' && *ptr != '}')
								++ptr;
						}
					}
					else
						NNHardAssert(strncmp(ptr, "numeric", 7) == 0 || strncmp(ptr, "integer", 7) == 0 || strncmp(ptr, "real", 4) == 0, "Unrecognized attribute type!");
					
					rel.attrVals.push_back(attrVals);
				}
				else if(startsWith(line, "@data"))
					break;
			}
		}
		
		while(!fin.fail())
		{
			std::getline(fin, line);
			char *ptr = const_cast<char *>(line.c_str());
			
			// end-of-file
			if(fin.fail())
				break;
			
			// empty line
			if(*ptr == '\0')
				continue;
			
			Vector<T> *rowPtr = new Vector<T>(rel.attrNames.size());
			Vector<T> &row = *rowPtr;
			rows.push_back(rowPtr);
			
			size_t i = 0;
			while(*ptr != '\0')
			{
				skipWhitespace(&ptr);
				if(*ptr == '\0')
					break;
				NNHardAssert(i < rel.attrNames.size(), "Too many columns on row " + std::to_string(rows.size()));
				if(rel.attrVals[i].size() == 0)
				{
					if(*ptr == '?')
					{
						row[i] = unknown;
						++ptr;
					}
					else
						row[i] = std::strtod(ptr, &ptr);
					
					if(*ptr == ',')
						++ptr;
				}
				else
				{
					char *end = tokenEnd(ptr, ",");
					auto j = rel.attrVals[i].find(std::string(ptr, end - ptr));
					NNHardAssert(j != rel.attrVals[i].end(), "Invalid nominal value '" + std::string(ptr, end - ptr) + "'");
					row[i] = j->second;
					ptr = end;
				}
				++i;
			}
			NNHardAssert(i == rel.attrNames.size(), "Not enough columns on row " + std::to_string(rows.size()));
		}
		fin.close();
		
		if(relPtr != nullptr)
			*relPtr = rel;
		
		Matrix<T> flattened(Vector<T>::flatten(rows), rows.size(), rel.attrNames.size());
		for(auto *i : rows)
			delete i;
		
		for(size_t i = 0; i < flattened.cols(); ++i)
		{
			double sum = 0.0;
			size_t count = 0;
			for(size_t j = 0; j < flattened.rows(); ++j)
			{
				if(flattened(j, i) != unknown)
				{
					sum += flattened(j, i);
					++count;
				}
			}
			if(count > 0)
			{
				double mean = sum / count;
				for(size_t j = 0; j < flattened.rows(); ++j)
					if(flattened(j, i) == unknown)
						flattened(j, i) = mean;
			}
		}
		
		return flattened;
	}

private:
	static bool startsWith(std::string str, std::string prefix)
	{
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);
		std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::tolower);
		return str.compare(0, prefix.length(), prefix) == 0;
	}
	
	static void skipWhitespace(char **ptr)
	{
		while(**ptr == ' ' || **ptr == '\t')
			++*ptr;
	}
	
	static char *tokenEnd(char *start, const char *delim = " \t")
	{
		char *ptr = start;
		skipWhitespace(&ptr);
		
		// Skip token; may be quoted
		if(*ptr == '\'')
		{
			++ptr;
			while(*ptr != '\'' && *ptr != '\0')
				++ptr;
			NNHardAssert(*ptr == '\'', "Invalid token!");
			++ptr;
		}
		else if(*ptr == '"')
		{
			++ptr;
			while(*ptr != '"' && *ptr != '\0')
				++ptr;
			NNHardAssert(*ptr == '"', "Invalid token!");
			++ptr;
		}
		else
		{
			while(strspn(ptr, delim) == 0 && *ptr != '\0')
				++ptr;
		}
		
		return ptr;
	}
};

template <typename T>
T Loader<T>::unknown = -1e308;

}

#endif
