/*
 * CrossEntropyLoss.cpp
 *
 *  Created on: Feb 17, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/losses/CrossEntropyLoss.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/layers/Layer.hpp>
#include <Avocado/math/training.hpp>
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerLossFunction(CrossEntropyLoss());
	}

	bool CrossEntropyLoss::tryCombineWith(const Layer &layer) noexcept
	{
		if (layer.name() == "Softmax" or layer.name() == "Activation")
			m_is_combined_with_layer = true;
		return m_is_combined_with_layer;
	}

	Scalar CrossEntropyLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target) const
	{
		return math::calcLossFunction(context, LossType::CROSS_ENTROPY_LOSS, output, target);
	}
	void CrossEntropyLoss::getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const
	{
		math::calcLossGradient(context, LossType::CROSS_ENTROPY_LOSS, 1, 1, gradient, output, target, m_is_combined_with_layer);
	}

	std::string CrossEntropyLoss::name() const
	{
		return "CrossEntropyLoss";
	}
	CrossEntropyLoss* CrossEntropyLoss::clone() const
	{
		return new CrossEntropyLoss();
	}

} /* namespace avocado */

