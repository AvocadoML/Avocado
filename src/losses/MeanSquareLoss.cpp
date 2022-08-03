/*
 * MeanSquareLoss.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/losses/MeanSquareLoss.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/math/training.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
		registerLossFunction(MeanSquareLoss());
	}

	Scalar MeanSquareLoss::getLoss(const Context &context, const Tensor &output, const Tensor &target) const
	{
		const double scale = 1.0 / output.firstDim();
		const double result = scale * math::calcLossFunction(context, LossType::MEAN_SQUARE_LOSS, output, target).get<double>();
		return Scalar(result);
	}
	void MeanSquareLoss::getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const
	{
		math::calcLossGradient(context, LossType::MEAN_SQUARE_LOSS, 1, 1, gradient, output, target, false);
	}

	std::string MeanSquareLoss::name() const
	{
		return "MeanSquareLoss";
	}
	MeanSquareLoss* MeanSquareLoss::clone() const
	{
		return new MeanSquareLoss();
	}

} /* namespace avocado */

