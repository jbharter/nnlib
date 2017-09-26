#ifndef NN_LOGISTIC_H
#define NN_LOGISTIC_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Sigmoidal logistic activation function.
template <typename T = double>
class Logistic : public Map<T>
{
public:
	Logistic() = default;
	Logistic(const Serialized &) {}
	
	/// Single element forward.
	virtual T forwardOne(const T &x) override
	{
		return 1.0 / (1.0 + exp(-x));
	}
	
	/// Single element backward.
	virtual T backwardOne(const T &x, const T &y) override
	{
		return y * (1.0 - y);
	}
};

}

NNRegisterType(Logistic, Module);

#endif
