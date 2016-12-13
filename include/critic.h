#ifndef CRITIC_H
#define CRITIC_H

#include "matrix.h"

namespace nnlib
{

template <typename T>
class Critic
{
public:
	virtual void calculateBlame(const Matrix<T> &inputs, const Matrix<T> &targets) = 0;
	virtual Matrix<T> &blame() = 0;
};

}

#endif