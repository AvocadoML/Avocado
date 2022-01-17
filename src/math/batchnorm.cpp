/*
 * batchnorm.cpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/math/batchnorm.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>

namespace avocado
{
	namespace math
	{
		void affineForward(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &weight,
				const Tensor &bias, NonlinearityType activation)
		{
		}

		void batchNormInference(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &scale,
				const Tensor &bias, const Tensor &estimatedMean, const Tensor &estimatedVariance, double epsilon, NonlinearityType activation)
		{
		}
		void batchNormForward(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &scale,
				const Tensor &bias, Tensor &savedMean, Tensor &savedVariance, double epsilon, NonlinearityType activation)
		{
		}
		void batchNormBackward(const Context &context, NonlinearityType activation, Scalar alpha, Scalar beta, const Tensor &input,
				const Tensor &output, Tensor &gradientPrev, Tensor &gradientNext, const Tensor &scale, const Tensor &savedMean,
				const Tensor &savedVariance, double epsilon)
		{
		}
		void batchNormUpdate(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, const Tensor &gradientNext, Tensor &scaleUpdate,
				Tensor &biasUpdate, const Tensor &savedMean, const Tensor &savedVariance, double epsilon)
		{
		}

	} /* namespace math */
} /* namespace avocado */

