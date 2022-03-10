/*
 * batchnorm.hpp
 *
 *  Created on: Jan 2, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_BATCHNORM_HPP_
#define AVOCADO_MATH_BATCHNORM_HPP_

namespace avocado
{
	class Tensor;
	class Scalar;
	class Context;
	enum class NonlinearityType;
}

namespace avocado
{
	namespace math
	{
		void affineForward(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &weight,
				const Tensor &bias, NonlinearityType activation);

		void batchNormInference(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &scale,
				const Tensor &bias, const Tensor &estimatedMean, const Tensor &estimatedVariance, double epsilon, NonlinearityType activation);
		void batchNormForward(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &scale,
				const Tensor &bias, Tensor &savedMean, Tensor &savedVariance, double epsilon, NonlinearityType activation);
		void batchNormBackward(const Context &context, Scalar alpha, const Tensor &input, const Tensor &output, Scalar beta, Tensor &gradientIn,
				Tensor &gradientOut, const Tensor &scale, const Tensor &mean, const Tensor &variance, Scalar alpha2, Scalar beta2,
				Tensor &scaleUpdate, Tensor &biasUpdate, double epsilon, NonlinearityType activation);

	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_BATCHNORM_HPP_ */
