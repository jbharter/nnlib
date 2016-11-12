#include "tensor.h"
#include "error.h"
#include "random.h"
#include "linear.h"
#include <iostream>
#include <chrono>
using namespace nnlib;
using namespace std;

void testCorrectness();
double testEfficiency(size_t inps, size_t outs, size_t epochs, function<void()> &start, function<void()> &end);

int main()
{
	size_t inps				= 100000;
	size_t outs				= 1000;
	size_t epochs			= 10;
	
	using clock = chrono::high_resolution_clock;
	clock::time_point start;
	function<void()> startFn = [&](void) { start = clock::now(); };
	function<void()> endFn = [&](void) { cout << "took " << chrono::duration<double>(clock::now() - start).count() / epochs << " seconds per epoch" << endl; };
	
	testCorrectness();
	testEfficiency(inps, outs, epochs, startFn, endFn);
	
	return 0;
}

void testCorrectness()
{
	size_t inps = 2, outs = 3;
	Linear<double> layer(inps, outs);
	Vector<double> input(inps), target(outs);
	
	Vector<double> allWeights = Vector<double>::flatten({ &layer.weights(), &layer.bias() });
	size_t i = 0;
	
	// weights
	allWeights(i++) = 1;
	allWeights(i++) = 0;
	allWeights(i++) = 0;
	allWeights(i++) = 1;
	allWeights(i++) = 1;
	allWeights(i++) = 1;
	
	// bias
	allWeights(i++) = 0;
	allWeights(i++) = 1;
	allWeights(i++) = 2;
	
	input(0) = 3.14;
	input(1) = 10.0;
	
	target(0) = 3.14;
	target(1) = 11.0;
	target(2) = 15.14;
	
	Vector<double> &result = layer.forward(input);
	for(size_t i = 0; i < outs; ++i)
		Assert(result(i) == target(i), "forward failed!");
	
	Vector<double> &blame = layer.backward(input, target);
	Assert(blame(0) == 18.28 && blame(1) == 26.14, "backward failed to assign correct input blame!");
	
	for(size_t i = 0; i < outs; ++i)
		Assert(layer.biasBlame()(i) == target(i), "backward failed to assign correct bias blame!");
	
	Matrix<double> expectedBlame(outs, inps);
	expectedBlame[0] = 9.8596;
	expectedBlame[1] = 31.4;
	expectedBlame[2] = 34.54;
	expectedBlame[3] = 110.0;
	expectedBlame[4] = 47.5396;
	expectedBlame[5] = 151.4;
	
	for(size_t i = 0; i < expectedBlame.size(); ++i)
		Assert(fabs(layer.weightsBlame()[i] - expectedBlame[i]) < 1e-12, "backward failed to assign correct weights blame!");
	
	cout << "Passed all tests!" << endl;
}

double testEfficiency(size_t inps, size_t outs, size_t epochs, function<void()> &start, function<void()> &end)
{
	Matrix<double> weights(outs, inps);
	Vector<double> input(inps), bias(outs), result(outs);
	Random r;
	
	weights.fillNormal(r);
	input.fillNormal(r);
	bias.fillNormal(r);
	result.fill(0.0);
	
	start();
	for(size_t i = 0; i < epochs; ++i)
		result += weights * input + bias;
	end();
	
	double resultSum = 0.0;
	for(size_t i = 0; i < outs; ++i)
		resultSum += result[i];
	
	return resultSum;
}
