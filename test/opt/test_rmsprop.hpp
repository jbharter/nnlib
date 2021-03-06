#ifndef TEST_RMSPROP_H
#define TEST_RMSPROP_H

#include "nnlib/opt/rmsprop.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/critics/mse.hpp"
using namespace nnlib;

void TestRMSProp()
{
	RandomEngine::seed();
	
	Tensor<> feat = Tensor<>(10, 2).rand();
	Tensor<> lab = Tensor<>(10, 3).rand();
	
	Linear<> nn(2, 3);
	MSE<> critic;
	
	double errBefore = critic.forward(nn.forward(feat), lab);
	
	RMSProp<> opt(nn, critic);
	opt.step(feat.narrow(0, 0), lab.narrow(0, 0));
	opt.step(feat, lab);
	
	double errAfter = critic.forward(nn.forward(feat), lab);
	
	NNAssertLessThan(errAfter - errBefore, 0, "Optimization failed!");
}

#endif
