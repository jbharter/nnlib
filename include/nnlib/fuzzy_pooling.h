#ifndef FUZZY_POOLING_H
#define FUZZY_POOLING_H

#include "module.h"
#include <limits>

namespace nnlib
{

template <typename T = double>
class FuzzyPooling : public Module<T>
{
public:
	FuzzyPooling(size_t inps = 0, size_t batch = 1)
	: m_alpha(inps / 2), m_alphaBlame(inps / 2), m_inputBlame(batch, inps), m_outputs(batch, inps / 2)
	{
		NNAssert(inps % 2 == 0, "Fuzzy pooling layers must have an even number of inputs!");
		resetWeights();
	}
	
	void resetWeights()
	{
		for(double &val : m_alpha)
			val = Random<T>::uniform(std::numeric_limits<T>::epsilon(), 1);
	}
	
	virtual void resize(size_t inps, size_t outs, size_t bats) override
	{
		NNAssert(outs * 2 == inps, "Fuzzy pooling layers must have an input size that is twice its output!");
		bool needsReset = inps != m_inputBlame.cols() || outs != m_outputs.cols();
		m_inputBlame.resize(bats, inps);
		m_outputs.resize(bats, outs);
		m_alpha.resize(outs);
		m_alphaBlame.resize(outs / 2);
		if(needsReset)
			resetWeights();
	}
	
	virtual void resize(size_t inps) override
	{
		NNAssert(inps % 2 == 0, "Fuzzy pooling layers must have an even number of inputs!");
		resize(inps, inps / 2, m_outputs.rows());
	}
	
	virtual void batch(size_t bats) override
	{
		Module<T>::batch(bats);
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		auto i = inputs.begin();
		auto j = m_outputs.begin();
		auto k = m_alpha.begin(), end = m_alpha.end();
		
		for(; k != end; ++k)
			*k = *k < 0 ? 0 : (*k > 1 ? 1 : *k);
		
		for(size_t row = 0; row < inputs.rows(); ++row)
			for(k = m_alpha.begin(); k != end; ++k, ++i, ++j)
				*j = fuzzy(*i, *++i, *k);
		
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows(), "Incorrect batch size!");
		NNAssert(inputs.cols() == m_inputBlame.cols(), "Incorrect input size!");
		NNAssert(blame.rows() == m_outputs.rows(), "Incorrect batch size!");
		NNAssert(blame.cols() == m_outputs.cols(), "Incorrect blame size!");
		auto i = m_outputs.begin(), j = m_inputBlame.begin();
		auto k = blame.begin();
		
		auto inp = inputs.begin();
		
		m_alphaBlame.fill(0);
		
		for(size_t row = 0; row < inputs.rows(); ++row)
		{
			auto alpha = m_alpha.begin(), alphaBlame = m_alphaBlame.begin(), end = m_alpha.end();
			for(; alpha != end; ++alpha, ++alphaBlame, ++i, ++j, ++k, ++inp)
			{
				T aa = std::abs(*alpha);
				T x = *inp, y = *++inp, z = 1.0 / (aa + 1);
				*j   = *k * (y + *alpha) * z;
				*++j = *k * (x + *alpha) * z;
				
				// This makes it robust to the discontinuity in the derivative that occurs when a=0.
				if(aa < 0.001)
					*alpha = -*alpha;
				
				*alphaBlame += *k * (aa * (x + y) - *alpha * (x * y + 1.0)) / (aa * (aa + 1.0) * (aa + 1.0) + 0.001);
			}
		}
		
		return m_inputBlame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_outputs;
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
	}
	
	virtual Vector<Tensor<T> *> parameters() override
	{
		return { &m_alpha };
	}
	
	virtual Vector<Tensor<T> *> blame() override
	{
		return { &m_alphaBlame };
	}
	
private:
	/// If false = -1 and true = 1, then
	/// not(x)		= fuzzy(x, 0, 0)
	/// and(x, y)	= fuzzy(x, y, 1)
	/// or(x, y)	= -fuzzy(x, y, 0)
	/// xor(x, y)	= -fuzzy(x, y, 0)
	/// nand(x, y)	= -fuzzy(x, y, 1)
	/// nor(x, y)	= fuzzy(x, y, 0)
	/// nxor(x, y)	= fuzzy(x, y, 0.5)
	static double fuzzy(double x, double y, double alpha)
	{
		double t = std::abs(alpha);
		return (x + alpha) * (y + alpha) / (t + 1) - t;
	}
	
	Vector<T> m_alpha;		///< Alpha parameter for each unit in the network.
	Vector<T> m_alphaBlame;	///< Blame of the alpha parameter.
	
	Matrix<T> m_inputBlame;	///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_outputs;	///< The output of this layer.
};

}

#endif