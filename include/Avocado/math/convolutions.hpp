/*
 * convolution.hpp
 *
 *  Created on: Oct 22, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_CONVOLUTIONS_HPP_
#define AVOCADO_MATH_CONVOLUTIONS_HPP_

#include <Avocado/math/activations.hpp>

#include <array>

namespace avocado
{
	class Context;
	class Shape;
	class Tensor;
	namespace backend
	{
		struct ConvolutionDescriptor;
	}
}

namespace avocado
{
	enum class ConvPadding
	{
		SAME,
		VALID
	};

	namespace math
	{
		enum class ConvAlgorithm
		{
			AUTO,
			EXPLICIT_GEMM,
			IMPLICIT_GEMM,
			WINOGRAD
		};

		struct ConvConfig
		{
				ConvAlgorithm algorithm = ConvAlgorithm::AUTO;
				NonlinearityType activation = NonlinearityType::LINEAR;
				int groups = 1;
//				Shape kernel;
//				Shape padding;
//				Shape stride;
//				Shape dilation;
				std::array<int, 3> kernel = { 0, 0, 0 };
				std::array<int, 3> padding = { 0, 0, 0 };
				std::array<int, 3> stride = { 1, 1, 1 };
				std::array<int, 3> dilation = { 1, 1, 1 };

				avocado::backend::ConvolutionDescriptor getDescriptor() const noexcept;
		};

		int getConvolutionPadding(const ConvConfig &config, int inputShape, const Shape &weightShape);
		Shape getConvolutionOutputShape(const ConvConfig &config, const Shape &inputShape, const Shape &weightShape);

		void imToRow(const Context &context, const Tensor &input, Tensor &output, const ConvConfig &config, const Shape &weightShape,
				bool invertKernel = false);

		/**
		 *  @brief Calculates output = activation( (input * weights) + bias + add)
		 */
		void convolutionForward(const Context &context, const ConvConfig &config, const Tensor &input, Tensor &output, const Tensor &weights,
				const Tensor &bias, const Tensor &add);
		void convolutionBackward(const Context &context, const ConvConfig &config, Tensor &gradientPrev, Tensor &gradientNext, const Tensor &input,
				const Tensor &output, const Tensor &weights);
		void convolutionUpdate(const Context &context, const ConvConfig &config, const Tensor &gradientNext, const Tensor &input,
				Tensor &weightUpdate, Tensor &biasUpdate);
	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_CONVOLUTIONS_HPP_ */
