#ifndef TEST_ARCHIVE_H
#define TEST_ARCHIVE_H

#include <sstream>
#include "nnlib/serialization/basic.h"
#include "nnlib/tensor.h"
#include "nnlib/nn/linear.h"
using namespace nnlib;

template <typename InputArchive, typename OutputArchive>
void TestArchive()
{
	std::stringstream ss;
	ss.precision(16);
	
	Tensor<> a = Tensor<>(3, 2, 4).rand();
	Tensor<> b;
	
	InputArchive in(ss);
	OutputArchive out(ss);
	
	out(a);
	in(b);
	
	NNAssertEquals(a.shape(), b.shape(), "Serialization failed! Wrong shape!");
	for(auto i = a.begin(), j = b.begin(), k = a.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed! Wrong data!");
	
	std::string s;
	out("string with spaces");
	in(s);
	NNAssertEquals(s, "string with spaces", "Serialization failed!");
	
	bool ok = false;
	try
	{
		Tensor<> t;
		std::stringstream empty("");
		InputArchive ar(empty);
		ar(t);
	}
	catch(const Error &e)
	{
		ok = true;
	}
	NNAssert(ok, "Intentionally bad deserialization did not throw an error!");
}

template <typename T>
void TestSerializationOfModule(T &module)
{
	std::stringstream ss1, ss2;
	ss1.precision(16);
	ss2.precision(16);
	
	{
		std::stringstream ss;
		BasicOutputArchive out(ss);
		BasicInputArchive in(ss);
		std::string test;
		out("string with spaces");
		in(test);
		NNAssertEquals(test, "string with spaces", "Serialization failed!");
	}
	
	{
		BasicOutputArchive out(ss1);
		out(module);
		ss2 << ss1.str();
	}
	
	{
		T deserialized;
		
		BasicInputArchive in(ss1);
		in(deserialized);
		
		NNAssertEquals(deserialized.inputs(), module.inputs(), "Serialization failed! Mismatching inputs.");
		NNAssertEquals(deserialized.outputs(), module.outputs(), "Serialization failed! Mismatching outputs.");
		
		auto &p1 = module.parameters();
		auto &p2 = deserialized.parameters();
		
		for(auto i = p1.begin(), j = p2.begin(), k = p1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed! Mismatching parameters.");
	}
	
	{
		Module<> *deserialized;
		
		BasicInputArchive in(ss2);
		in(deserialized);
		
		NNAssertEquals(deserialized->inputs(), module.inputs(), "Generic serialization failed! Mismatching inputs.");
		NNAssertEquals(deserialized->outputs(), module.outputs(), "Generic serialization failed! Mismatching outputs.");
		
		Tensor<double> &p1 = module.parameters();
		Tensor<double> &p2 = deserialized->parameters();
		
		for(auto i = p1.begin(), j = p2.begin(), k = p1.end(); i != k; ++i, ++j)
			NNAssertAlmostEquals(*i, *j, 1e-12, "Generic serialization failed! Mismatching parameters.");
		
		delete deserialized;
	}
}

template <typename T>
void TestSerializationOfIterable(T &iterable)
{
	std::stringstream ss;
	ss.precision(16);
	
	T deserialized;
	
	BasicOutputArchive out(ss);
	out(iterable);
	
	BasicInputArchive in(ss);
	in(deserialized);
	
	for(auto i = iterable.begin(), j = deserialized.begin(), k = iterable.end(); i != k; ++i, ++j)
		NNAssertAlmostEquals(*i, *j, 1e-12, "Serialization failed!");
}

#endif
