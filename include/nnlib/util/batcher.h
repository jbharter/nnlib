#ifndef BATCHER_H
#define BATCHER_H

#include "random.h"
#include "tensor.h"

namespace nnlib
{

/// Takes two tensors and returns random slices along the major dimensions, one slice at at time.
/// This is useful for optimization.
/// Batcher requires non-const inputs and will shuffle them unless the copy flag is true.
template <typename T = double>
class Batcher
{
public:
	Batcher(Tensor<T> &feat, Tensor<T> &lab, size_t bats = 1, bool copy = false) :
		m_feat(copy ? feat.copy() : feat),
		m_lab(copy ? lab.copy() : lab),
		m_featBatch(m_feat),
		m_labBatch(m_lab)
	{
		NNAssert(feat.size(0) == lab.size(0), "Incompatible features and labels!");
		batch(bats);
		reset();
	}
	
	Batcher &batch(size_t bats)
	{
		NNAssert(bats <= m_feat.size(0), "Invalid batch size!");
		m_featBatch.resizeDim(0, bats);
		m_labBatch.resizeDim(0, bats);
		return *this;
	}
	
	size_t batch() const
	{
		return m_featBatch.size(0);
	}
	
	size_t batches() const
	{
		return m_feat.size(0) / m_featBatch.size(0);
	}
	
	Batcher &reset()
	{
		m_offset = 0;
		for(size_t i = 0, end = m_feat.size(0); i < end; ++i)
		{
			size_t j = Random<size_t>::uniform(end);
			m_feat.select(0, i).swap(m_feat.select(0, j));
			m_lab.select(0, i).swap(m_lab.select(0, j));
		}
		return *this;
	}
	
	bool next(bool autoReset = false)
	{
		m_offset += m_featBatch.size(0);
		if(m_offset + m_featBatch.size(0) > m_feat.size(0))
		{
			if(autoReset)
			{
				reset();
			}
			else
			{
				return false;
			}
		}
		
		m_feat.sub(m_featBatch, { { m_offset, m_featBatch.size(0) }, {} });
		m_lab.sub(m_labBatch, { { m_offset, m_featBatch.size(0) }, {} });
		
		return true;
	}
	
	Tensor<T> &features()
	{
		return m_featBatch;
	}
	
	Tensor<T> &labels()
	{
		return m_labBatch;
	}
	
private:
	Tensor<T> m_feat;
	Tensor<T> m_lab;
	Tensor<T> m_featBatch;
	Tensor<T> m_labBatch;
	size_t m_offset;
};

}

#endif
