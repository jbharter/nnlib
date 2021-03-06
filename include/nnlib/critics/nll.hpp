#ifndef CRTIICS_NLL_HPP
#define CRTIICS_NLL_HPP

#include "critic.hpp"

namespace nnlib
{

/// \brief Negative log loss critic.
///
/// This critic requires matrix input and single-column matrix output.
template <typename T = double>
class NLL : public Critic<T>
{
public:
	NLL(bool average = true) :
		m_average(average)
	{}
	
	bool average() const
	{
		return m_average;
	}
	
	NLL &average(bool ave)
	{
		m_average = ave;
		return *this;
	}
	
	/// A convenience method for counting misclassifications, since we know the output will be categorical.
	size_t misclassifications(const Tensor<T> &input, const Tensor<T> &target)
	{
		NNAssertEquals(input.size(0), target.size(0), "Incompatible operands!");
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		NNAssertEquals(target.dims(), 2, "Expected matrix target!");
		NNAssertEquals(target.size(1), 1, "Expected single-column target!");
		
		size_t miss = 0;
		for(size_t i = 0, iend = input.size(0), jend = input.size(1); i < iend; ++i)
		{
			NNAssertGreaterThanOrEquals(target(i, 0), 0, "Expected positive target!");
			
			size_t max = 0;
			for(size_t j = 1; j < jend; ++j)
				if(input(i, j) > input(i, max))
					max = j;
			
			if(max != target(i, 0))
				++miss;
		}
		
		return miss;
	}
	
	/// L = 1/n sum_i( -input(target(i)) )
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssertEquals(input.size(0), target.size(0), "Incompatible operands!");
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		NNAssertEquals(target.dims(), 2, "Expected matrix target!");
		NNAssertEquals(target.size(1), 1, "Expected single-column target!");
		
		T sum = 0;
		size_t j;
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			NNAssertGreaterThanOrEquals(target(i, 0), 0, "Expected positive target!");
			j = target(i, 0);
			sum -= input(i, j);
		}
		
		if(m_average)
			sum /= input.size();
		
		return sum;
	}
	
	/// dL/di = target == i ? -1/n : 0
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssertEquals(input.size(0), target.size(0), "Incompatible operands!");
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		NNAssertEquals(target.size(1), 1, "Expected single-column target!");
		
		m_inGrad.resize(input.shape()).fill(0);
		T weight = -1.0;
		
		if(m_average)
			weight /= input.size();
		
		size_t j;
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			NNAssertGreaterThanOrEquals(target(i, 0), 0, "Expected positive target!");
			j = target(i, 0);
			m_inGrad(i, j) = weight;
		}
		
		return m_inGrad;
	}
	
protected:
	using Critic<T>::m_inGrad;
	
private:
	bool m_average;
};

}

#endif
