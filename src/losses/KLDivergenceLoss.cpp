/*
 * KLDivergenceLoss.cpp
 *
 *  Created on: Feb 17, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/losses/KLDivergenceLoss.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/layers/Layer.hpp>
#include <Avocado/math/training.hpp>
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
		registerLossFunction(KLDivergenceLoss());
	}

	bool KLDivergenceLoss::tryCombineWith(const Layer &layer) noexcept
	{
		if (layer.name() == "Softmax")
			m_is_combined_with_layer = true;
		if (layer.name() == "Activation" and layer.getNonlinearity() == NonlinearityType::SIGMOID)
			m_is_combined_with_layer = true;
		return m_is_combined_with_layer;
	}

	Scalar KLDivergenceLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target) const
	{
		return math::calcLossFunction(context, LossType::KL_DIVERGECE_LOSS, output, target);
	}
	void KLDivergenceLoss::getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const
	{
		math::calcLossGradient(context, LossType::KL_DIVERGECE_LOSS, 1, 1, gradient, output, target, m_is_combined_with_layer);
	}

	std::string KLDivergenceLoss::name() const
	{
		return "KLDivergenceLoss";
	}
	KLDivergenceLoss* KLDivergenceLoss::clone() const
	{
		return new KLDivergenceLoss();
	}

} /* namespace avocado */

