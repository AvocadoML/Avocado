/*
 * convolutions.cpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/math/convolutions.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/core/Shape.hpp>

namespace avocado
{
	namespace math
	{
		int getConvolutionPadding(const ConvConfig &config, int inputShape, const Shape &weightShape)
		{
			return 0;
		}
		Shape getConvolutionOutputShape(const ConvConfig &config, const Shape &inputShape, const Shape &weightShape)
		{
			return Shape();
		}

		void imToRow(const Context &context, const Tensor &input, Tensor &output, const ConvConfig &config, const Shape &weightShape,
				bool invertKernel)
		{
		}

		void convolutionForward(const Context &context, const ConvConfig &config, const Tensor &input, Tensor &output, const Tensor &weights,
				const Tensor &bias, const Tensor &add)
		{
		}
		void convolutionBackward(const Context &context, const ConvConfig &config, Tensor &gradientPrev, Tensor &gradientNext, const Tensor &input,
				const Tensor &output, const Tensor &weights)
		{
		}
		void convolutionUpdate(const Context &context, const ConvConfig &config, const Tensor &gradientNext, const Tensor &input,
				Tensor &weightUpdate, Tensor &biasUpdate)
		{
		}
	} /* namespace math */
} /* namespace aovocado */

