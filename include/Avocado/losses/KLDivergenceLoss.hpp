/*
 * KLDivergenceLoss.hpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LOSSES_KLDIVERGENCELOSS_HPP_
#define AVOCADO_LOSSES_KLDIVERGENCELOSS_HPP_

#include <Avocado/losses/LossFunction.hpp>

namespace avocado
{

	class KLDivergenceLoss: public LossFunction
	{
			bool m_is_combined_with_layer = false;
		public:
			KLDivergenceLoss() = default;

			bool tryCombineWith(const Layer &layer) noexcept;

			Scalar getLoss(const Context &context, const Tensor &output, const Tensor &target) const;
			void getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const;

			std::string name() const;
			KLDivergenceLoss* clone() const;
	};

} /* namespace avocado */

#endif /* AVOCADO_LOSSES_KLDIVERGENCELOSS_HPP_ */
