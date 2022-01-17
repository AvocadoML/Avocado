/*
 * MeanSquareLoss.hpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LOSSES_MEANSQUARELOSS_HPP_
#define AVOCADO_LOSSES_MEANSQUARELOSS_HPP_

#include <Avocado/losses/LossFunction.hpp>

namespace avocado
{

	class MeanSquareLoss: public LossFunction
	{
		public:
			MeanSquareLoss() = default;

			Scalar getLoss(const Context &context, const Tensor &output, const Tensor &target) const;
			void getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const;

			std::string name() const;
			MeanSquareLoss* clone() const;
	};

} /* namespace avocado */

#endif /* AVOCADO_LOSSES_MEANSQUARELOSS_HPP_ */
