#ifndef CRITIC_H
#define CRITIC_H

#include "../util/tensor.h"

namespace nnlib
{

template <typename T = double>
class Critic
{
public:
	/// Calculate the loss (how far input is from target).
	virtual Tensor<T> &forward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	/// Calculate the gradient of the loss w.r.t. the input.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	/// Output buffer (the loss).
	virtual Tensor<T> &output() = 0;
	
	/// Input gradient buffer.
	virtual Tensor<T> &inGrad() = 0;
	
	/// Get the input shape of this critic, including batch.
	virtual const Storage<size_t> &inputs() const
	{
		return const_cast<Critic<T> *>(this)->inGrad().shape();
	}
	
	/// Set the input shape of this critic, including batch.
	/// By default, this resizes the input gradient and resets the batch to dims[0].
	virtual Critic &inputs(const Storage<size_t> &dims)
	{
		inGrad().resize(dims);
		return batch(dims[0]);
	}
	
	/// Get the output shape of this critic, including batch.
	virtual const Storage<size_t> &outputs() const
	{
		return const_cast<Critic<T> *>(this)->output().shape();
	}
	
	/// Set the output shape of this critic, including batch.
	/// By default, this resizes the output and resets the batch to dims[0].
	virtual Critic &outputs(const Storage<size_t> &dims)
	{
		output().resize(dims);
		return batch(dims[0]);
	}
	
	/// Get the batch size of this critic.
	/// By default, this returns the first dimension of the input shape.
	virtual size_t batch() const
	{
		return const_cast<Critic<T> *>(this)->inGrad().size(0);
	}
	
	/// Set the batch size of this critic.
	/// By default, this resizes the first dimension of the input gradient and output.
	virtual Critic &batch(size_t bats)
	{
		Storage<size_t> dims = inGrad().shape();
		dims[0] = bats;
		inGrad().resize(dims);
		
		dims = output().shape();
		dims[0] = bats;
		output().resize(dims);
		
		return *this;
	}
};

}

#endif