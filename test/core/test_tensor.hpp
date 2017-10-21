#ifndef TEST_TENSOR_H
#define TEST_TENSOR_H

#include "nnlib/core/tensor.hpp"
#include "nnlib/util/tensor_util.hpp"
using namespace nnlib;

void TestTensor()
{
	size_t TRAVIS_TEST_IDX = 0;
	
	// test empty concat
	
	{
		Tensor<> empty = Tensor<>::concatenate({});
		NNAssertEquals(empty.size(), 0, "Tensor::concatenate on an empty list failed!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	// test resizing to the same size
	
	{
		Tensor<> t(3);
		t.resizeDim(0, 3);
		NNAssertEquals(t.size(0), 3, "Tensor::resizeDim to the same size failed!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	// test constructors
	
	Tensor<> empty;
	NNAssertEquals(empty.size(), 0, "Tensor::Tensor() failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	Tensor<> vector(3, 4);
	NNAssertEquals(vector.dims(), 2, "Tensor::Tensor(size_t, size_t) failed! Wrong dimensionality!");
	NNAssertEquals(vector.size(), 12, "Tensor::Tensor(size_t, size_t) failed! Wrong size!");
	NNAssertEquals(*vector.ptr(), 0.0, "Tensor::Tensor(size_t, size_t) failed! Wrong value!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	Tensor<> initFromStorage(Storage<double>({ 3.14, 42.0 }));
	NNAssertEquals(initFromStorage.dims(), 1, "Tensor::Tensor(Storage) failed! Wrong dimensionality!");
	NNAssertEquals(initFromStorage.size(), 2, "Tensor::Tensor(Storage) failed! Wrong size!");
	NNAssertEquals(*initFromStorage.ptr(), 3.14, "Tensor::Tensor(Storage) failed! Wrong value!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	Tensor<> initFromList({ 1.0, 2.0, 3.0, 4.0 });
	NNAssertEquals(initFromList.dims(), 1, "Tensor::Tensor(initializer_list) failed! Wrong dimensionality!");
	NNAssertEquals(initFromList.size(), 4, "Tensor::Tensor(initializer_list) failed! Wrong size!");
	NNAssertEquals(*initFromList.ptr(), 1.0, "Tensor::Tensor(initializer_list) failed! Wrong value!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	Tensor<> initWithDims({ 4, 2, 3 }, true);
	NNAssertEquals(initWithDims.dims(), 3, "Tensor::Tensor(Storage, bool) failed! Wrong dimensionality!");
	NNAssertEquals(initWithDims.size(), 24, "Tensor::Tensor(Storage, bool) failed! Wrong size!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	Tensor<> view(vector);
	NNAssertEquals(view.shape(), vector.shape(), "Tensor::Tensor(Tensor &) failed! Wrong shape!");
	NNAssertEquals(view.ptr(), vector.ptr(), "Tensor::Tensor(Tensor &) failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	Tensor<> viewOfMoved(std::move(initWithDims));
	NNAssertEquals(viewOfMoved.shape(), initWithDims.shape(), "Tensor::Tensor(Tensor &&) failed! Wrong shape!");
	NNAssertEquals(viewOfMoved.ptr(), initWithDims.ptr(), "Tensor::Tensor(Tensor &&) failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	// test assignment
	
	empty = Storage<double>({ 1.0, 2.0, 3.0 });
	NNAssertEquals(empty.dims(), 1, "Tensor::operator=(Storage) failed! Wrong dimensionality!");
	NNAssertEquals(empty.size(), 3, "Tensor::operator=(Storage) failed! Wrong shape!");
	NNAssertEquals(*empty.ptr(), 1.0, "Tensor::operator=(Storage) failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty = { 3.0, 6.0, 9.0, 12.0 };
	NNAssertEquals(empty.dims(), 1, "Tensor::operator=(initializer_list) failed! Wrong dimensionality!");
	NNAssertEquals(empty.size(), 4, "Tensor::operator=(initializer_list) failed! Wrong shape!");
	NNAssertEquals(*empty.ptr(), 3.0, "Tensor::operator=(Storage) failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty = vector;
	NNAssertEquals(empty.shape(), vector.shape(), "Tensor::operator=(Tensor &) failed! Wrong shape!");
	NNAssertEquals(empty.ptr(), vector.ptr(), "Tensor::operator=(Tensor &) failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty = std::move(initFromStorage);
	NNAssertEquals(empty.shape(), initFromStorage.shape(), "Tensor::operator=(Tensor &&) failed! Wrong shape!");
	NNAssertEquals(empty.ptr(), initFromStorage.ptr(), "Tensor::operator=(Tensor &&) failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	// test element access
	
	vector(2, 2) = 3.14;
	NNAssertEquals(vector(2, 2), 3.14, "Tensor::operator() failed!");
	NNAssertEquals(view(2, 2), 3.14, "Tensor::operator() failed!");
	NNAssertEquals(&vector(2, 2), &view(2, 2), "Tensor::operator() failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	vector.fill(6.26);
	for(auto &v : vector)
		NNAssertEquals(v, 6.26, "Tensor::fill failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	// test other methods
	
	empty.resize(11, 22, 33);
	NNAssertEquals(empty.dims(), 3, "Tensor::resize(size_t, size_t, size_t) failed! Wrong dimensionality!");
	NNAssertEquals(empty.size(0), 11, "Tensor::resize(size_t, size_t, size_t) failed!");
	NNAssertEquals(empty.size(1), 22, "Tensor::resize(size_t, size_t, size_t) failed!");
	NNAssertEquals(empty.size(2), 33, "Tensor::resize(size_t, size_t, size_t) failed!");
	NNAssertEquals(empty.size(), 11 * 22 * 33, "Tensor::resize(size_t, size_t, size_t) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty.resize(Storage<size_t>({ 2, 4, 6 }));
	NNAssertEquals(empty.shape(), Storage<size_t>({ 2, 4, 6 }), "Tensor::resize(Storage) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty.resizeDim(1, 18);
	NNAssertEquals(empty.size(0), 2, "Tensor::resizeDim failed!");
	NNAssertEquals(empty.size(1), 18, "Tensor::resizeDim failed!");
	NNAssertEquals(empty.size(2), 6, "Tensor::resizeDim failed!");
	NNAssertEquals(empty.size(), 2 * 18 * 6, "Tensor::resizeDim failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	view = empty.view(2, 2, 2, 2);
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 2, 2, 2 }), "Tensor::view failed! Wrong shape!");
	NNAssertEquals(view.ptr(), empty.ptr(), "Tensor::view failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	view = initWithDims.view(Storage<size_t>({ 3, 6 }));
	NNAssertEquals(view.shape(), Storage<size_t>({ 3, 6 }), "Tensor::view failed! Wrong shape!");
	NNAssertEquals(view.ptr(), initWithDims.ptr(), "Tensor::view failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	view = empty.reshape(Storage<size_t>({ 2, 2, 9, 2, 3 }));
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 2, 9, 2, 3 }), "Tensor::reshape failed! Wrong shape!");
	NNAssertNotEquals(view.ptr(), empty.ptr(), "Tensor::reshape failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	view = empty.reshape(18, 12);
	NNAssertEquals(view.shape(), Storage<size_t>({ 18, 12 }), "Tensor::reshape failed! Wrong shape!");
	NNAssertNotEquals(view.ptr(), empty.ptr(), "Tensor::reshape failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	view = empty.select(1, 16);
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 6 }), "Tensor::select failed! Wrong shape!");
	NNAssertEquals(&view(1, 4), &empty(1, 16, 4), "Tensor::select failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	view = empty.narrow(2, 3, 3);
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 18, 3 }), "Tensor::narrow failed! Wrong shape!");
	NNAssertEquals(&view(1, 2, 1), &empty(1, 2, 4), "Tensor::narrow failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty.resize(2, 1, 4);
	view = empty.expand(1, 3);
	NNAssertEquals(view.shape(), Storage<size_t>({ 2, 3, 4 }), "Tensor::expand failed! Wrong shape!");
	NNAssertEquals(&view(1, 0, 2), &view(1, 1, 2), "Tensor::expand failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	vector.resize(6, 9);
	vector.sub(view, { { 2, 3 }, { 3, 3 } });
	NNAssertEquals(&view.storage(), &vector.storage(), "Tensor::sub(Tensor, initializer_list) failed! Wrong data!");
	NNAssertEquals(&view(1, 1), &vector(3, 4), "Tensor::sub(Tensor, initializer_list) failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	viewOfMoved = vector.sub({ { 1, 3 }, { 4, 3 } });
	NNAssertEquals(&viewOfMoved.storage(), &vector.storage(), "Tensor::sub(initializer_list) failed! Wrong data!");
	NNAssertEquals(&viewOfMoved(1, 1), &vector(2, 5), "Tensor::sub(initializer_list) failed! Wrong data!");
	NNAssertEquals(&viewOfMoved(1, 1), &view(0, 2), "Tensor::sub(initializer_list) failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty = view.copy();
	NNAssertEquals(empty.shape(), view.shape(), "Tensor::copy() failed! Wrong shape!");
	NNAssertNotEquals(&empty.storage(), &view.storage(), "Tensor::copy() failed! Wrong data!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty.copy(viewOfMoved);
	NNAssertEquals(empty.shape(), viewOfMoved.shape(), "Tensor::copy(Tensor &) failed! Wrong shape!");
	for(auto x = empty.begin(), y = viewOfMoved.begin(); x != empty.end(); ++x, ++y)
	{
		NNAssertEquals(*x, *y, "Tensor::copy(Tensor &) failed! Wrong data!");
		NNAssertNotEquals(&*x, &*y, "Tensor::copy(Tensor &) failed! Wrong data!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty = Tensor<>(3, 4).fill(1.0);
	view = Tensor<>(3, 4).fill(2.0);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty.swap(view);
	for(auto x = empty.begin(), y = view.begin(); x != empty.end(); ++x, ++y)
	{
		NNAssertEquals(*x, 2.0, "Tensor::swap(Tensor &) failed!");
		NNAssertEquals(*y, 1.0, "Tensor::swap(Tensor &) failed!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	empty.swap(std::move(view));
	for(auto x = empty.begin(), y = view.begin(); x != empty.end(); ++x, ++y)
	{
		NNAssertEquals(*x, 1.0, "Tensor::swap(Tensor &&) failed!");
		NNAssertEquals(*y, 2.0, "Tensor::swap(Tensor &&) failed!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	vector.resize(3, 2);
	vector.select(1, 0).fill(1.0);
	vector.select(1, 1).fill(2.0);
	vector = vector.transpose();
	for(auto &v : vector.select(0, 0))
		NNAssertEquals(v, 1.0, "Tensor::transpose failed!");
	for(auto &v : vector.select(0, 1))
		NNAssertEquals(v, 2.0, "Tensor::transpose failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	vector.zeros();
	for(auto &v : vector)
		NNAssertEquals(v, 0.0, "Tensor::zeros failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	vector.ones();
	for(auto &v : vector)
		NNAssertEquals(v, 1.0, "Tensor::zeros failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	vector.rand(5, 10);
	for(auto &v : vector)
	{
		NNAssertGreaterThanOrEquals(v, 5, "Tensor::rand failed!");
		NNAssertLessThanOrEquals(v, 10, "Tensor::rand failed!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;
	
	vector.randn(700, 1);
	for(auto &v : vector)
		NNAssertAlmostEquals(v, 700, 100, "Tensor::randn(T, T) failed! It produced very distant outliers.");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	vector.randn(5280, 2, 5);
	for(auto &v : vector)
		NNAssertAlmostEquals(v, 5280, 5, "Tensor::randn(T, T, T) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view = vector.copy().scale(2);
	for(auto x = view.begin(), y = vector.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, 2 * *y, 1e-12, "Tensor::scale failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view = vector.copy().add(2);
	for(auto x = view.begin(), y = vector.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, 2 + *y, 1e-12, "Tensor::add(T) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view.resize(12).rand();
	vector.resize(12).rand();
	empty = view.copy().addV(vector);
	for(auto x = view.begin(), y = vector.begin(), z = empty.begin(); x != view.end(); ++x, ++y, ++z)
		NNAssertAlmostEquals(*x + *y, *z, 1e-12, "Tensor::addV failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	// test tensor math
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view = Tensor<>(3, 100).rand();
	vector = Tensor<>(100).rand();
	viewOfMoved = Tensor<>(view.size(0));
	empty = Tensor<>(view.size(0));
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty.assignMV(view, vector);
	for(size_t i = 0; i < view.size(0); ++i)
		for(size_t j = 0; j < view.size(1); ++j)
			viewOfMoved(i) += view(i, j) * vector(j);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	for(auto x = viewOfMoved.begin(), y = empty.begin(); x != viewOfMoved.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::assignMV failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	vector.resize(3);
	viewOfMoved.resize(100).zeros();
	empty.resize(100);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty.assignMTV(view, vector);
	for(size_t i = 0; i < view.size(1); ++i)
		for(size_t j = 0; j < view.size(0); ++j)
			viewOfMoved(i) += view(j, i) * vector(j);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	for(auto x = viewOfMoved.begin(), y = empty.begin(); x != viewOfMoved.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::assignMTV failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty.resize(3, 3);
	viewOfMoved.resize(3, 3);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty.assignVV(vector, vector);
	for(size_t i = 0; i < vector.size(); ++i)
		for(size_t j = 0; j < vector.size(); ++j)
			viewOfMoved(i, j) = vector(i) * vector(j);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	for(auto x = viewOfMoved.begin(), y = empty.begin(); x != viewOfMoved.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::assignVV failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view = empty.copy().addM(viewOfMoved);
	for(size_t i = 0; i < view.size(0); ++i)
		for(size_t j = 0; j < view.size(1); ++j)
			NNAssertAlmostEquals(view(i, j), empty(i, j) + viewOfMoved(i, j), 1e-12, "Tensor::addM failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty.resize(10, 5).rand();
	vector.resize(10, 5).rand();
	view.resize(10, 10);
	viewOfMoved.resize(10, 10);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	for(size_t i = 0; i < 10; ++i)
	{
		for(size_t j = 0; j < 10; ++j)
		{
			viewOfMoved(i, j) = 0;
			for(size_t k = 0; k < 5; ++k)
				viewOfMoved(i, j) += empty(i, k) * vector(j, k);
		}
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view.assignMM(empty, vector.transpose().copy());
	for(auto x = viewOfMoved.begin(), y = view.begin(); x != viewOfMoved.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::assignMM failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view.assignMTM(empty.transpose().copy(), vector.transpose().copy(), 0.5);
	for(auto x = viewOfMoved.begin(), y = view.begin(); x != viewOfMoved.end(); ++x, ++y)
		NNAssertAlmostEquals(0.5 * *x, *y, 1e-12, "Tensor::assignMM failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view.assignMMT(empty, vector, 1, 1);
	for(auto x = viewOfMoved.begin(), y = view.begin(); x != viewOfMoved.end(); ++x, ++y)
		NNAssertAlmostEquals(1.5 * *x, *y, 1e-12, "Tensor::assignMM failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view = empty.copy().pointwiseProduct(vector);
	for(auto x = view.begin(), y = empty.begin(), z = vector.begin(); x != view.end(); ++x, ++y, ++z)
		NNAssertAlmostEquals(*x, *y * *z, 1e-12, "Tensor::pointwiseProduct failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view = empty.copy().add(vector, 0.75);
	for(auto x = view.begin(), y = empty.begin(), z = vector.begin(); x != view.end(); ++x, ++y, ++z)
		NNAssertAlmostEquals(*x, *y + 0.75 * *z, 1e-12, "Tensor::add(Tensor, T) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view = empty.copy().square();
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y * *y, 1e-12, "Tensor::square failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty = view.copy();
	view.apply([](double &v) { v = tanh(v); });
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertEquals(*x, tanh(*y), "Tensor::apply failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	double sum = view.sum();
	for(auto &v : view)
		sum -= v;
	NNAssertAlmostEquals(sum, 0, 1e-12, "Tensor::sum() failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	vector.resize(2, 3).copy({ 2, 1, 3, 4, 6, 5 });
	vector.sum(view.resize(3), 0);
	empty.resize(3).copy({ 6, 7, 8 });
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::sum(Tensor, size_t) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view.resize(2) = vector.sum(1);
	empty.resize(2).copy({ 6, 15 });
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::sum(size_t) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	NNAssertAlmostEquals(vector.mean(), 3.5, 1e-12, "Tensor::mean failed!");
	NNAssertAlmostEquals(vector.variance(), 2.917, 1e-3, "Tensor::variance failed!");
	NNAssertEquals(vector.min(), 1, "Tensor::min failed!");
	NNAssertEquals(vector.max(), 6, "Tensor::max failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	vector.normalize(-1, 20);
	NNAssertAlmostEquals(vector.min(), -1, 1e-12, "Tensor::normalize failed!");
	NNAssertAlmostEquals(vector.max(), 20, 1e-12, "Tensor::normalize failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	vector.clip(0, 10);
	NNAssertAlmostEquals(vector.min(), 0, 1e-12, "Tensor::clip failed!");
	NNAssertAlmostEquals(vector.max(), 10, 1e-12, "Tensor::clip failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	NNAssertNotEquals(vector.begin(), view.begin(), "TensorIterator::operator== failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	// test const methods
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view.resize(10, 10);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	const Tensor<> &constant = view;
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	{
		const Tensor<> &constView = constant.view(Storage<size_t>({ 3, 3 }));
		NNAssertEquals(constView.shape(), Storage<size_t>({ 3, 3 }), "const Tensor::view(Storage) failed! Wrong shape!");
		NNAssertEquals(&constView(0, 0), &constant(0, 0), "const Tensor::view(Storage) failed! Wrong data!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	{
		const Tensor<> &constView = constant.view(3, 3);
		NNAssertEquals(constView.shape(), Storage<size_t>({ 3, 3 }), "const Tensor::view(size_t, size_t) failed! Wrong shape!");
		NNAssertEquals(&constView(0, 0), &constant(0, 0), "const Tensor::view(size_t, size_t) failed! Wrong data!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	{
		const Tensor<> &constView = constant.select(1, 1);
		NNAssertEquals(constView.shape(), Storage<size_t>({ 10 }), "const Tensor::select failed! Wrong shape!");
		NNAssertEquals(&constView(2), &constant(2, 1), "const Tensor::select failed! Wrong data!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	{
		const Tensor<> &constView = constant.narrow(1, 2, 2);
		NNAssertEquals(constView.shape(), Storage<size_t>({ 10, 2 }), "const Tensor::narrow failed! Wrong shape!");
		NNAssertEquals(&constView(2, 1), &constant(2, 3), "const Tensor::narrow failed! Wrong data!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	{
		const Tensor<> &constView = constant.narrow(1, 2, 2);
		NNAssertEquals(constView.shape(), Storage<size_t>({ 10, 2 }), "const Tensor::narrow failed! Wrong shape!");
		NNAssertEquals(&constView(2, 1), &constant(2, 3), "const Tensor::narrow failed! Wrong data!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view.resize(10, 1, 10);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	{
		const Tensor<> &constant = view;
		const Tensor<> &constView = constant.expand(1, 50);
		NNAssertEquals(constView.shape(), Storage<size_t>({ 10, 50, 10 }), "const Tensor::expand failed! Wrong shape!");
		NNAssertEquals(&constView(7, 29, 3), &constant(7, 0, 3), "const Tensor::expand failed! Wrong data!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	{
		const Tensor<> &constant = view;
		const Tensor<> &constView = constant.sub({ { 2, 4 }, { 0 }, { 7, 2 } });
		NNAssertEquals(constView.shape(), Storage<size_t>({ 4, 1, 2 }), "const Tensor::sub failed! Wrong shape!");
		NNAssertEquals(&constView(2, 0, 1), &constant(4, 0, 8), "const Tensor::sub failed! Wrong data!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	{
		view.randn();
		empty = view.copy();
		std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

		const Tensor<> &constant = view;
		double sum = 0.0;
		constant.apply([&sum](const double &v) { sum += v; });
		NNAssertEquals(sum, constant.sum(), "const Tensor::apply failed!");
	}
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	// test tensor util
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	view.randn();
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty = view + view;
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(2 * *x, *y, 1e-12, "operator+(Tensor, Tensor) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty += view;
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(3 * *x, *y, 1e-12, "operator+=(Tensor, Tensor) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty = view.copy().scale(2.5) - view;
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(1.5 * *x, *y, 1e-12, "operator-(Tensor, Tensor) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty -= view;
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(0.5 * *x, *y, 1e-12, "operator-=(Tensor, Tensor) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty = view * 2;
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x * 2, *y, 1e-12, "operator*(Tensor, T) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty *= 2;
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x * 4, *y, 1e-12, "operator*=(Tensor, T) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty = view / 2;
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x / 2, *y, 1e-12, "operator/(Tensor, T) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	empty /= 2;
	for(auto x = view.begin(), y = empty.begin(); x != view.end(); ++x, ++y)
		NNAssertAlmostEquals(*x / 4, *y, 1e-12, "operator/=(Tensor, T) failed!");
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	// test serialization
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	Tensor<> serializable = Tensor<>(3, 4, 5, 6).rand();
	Tensor<> serialized;
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	Serialized node;
	serializable.save(node);
	serialized = Tensor<>(node);
	std::cout << "TRAVIS_TEST_IDX: " << TRAVIS_TEST_IDX++ << std::endl;

	for(auto x = serializable.begin(), y = serialized.begin(); x != serializable.end(); ++x, ++y)
		NNAssertAlmostEquals(*x, *y, 1e-12, "Tensor::save and/or Tensor::load failed!");
}

#endif
