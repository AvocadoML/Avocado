/*
 * activations.hpp
 *
 *  Created on: Mar 17, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_ACTIVATIONS_HPP_
#define AVOCADO_MATH_ACTIVATIONS_HPP_

#include <string>

namespace avocado /* forward declarations */
{
	class Tensor;
	class Scalar;
	class Context;
}

namespace avocado
{
	enum class NonlinearityType
	{
		LINEAR,
		SIGMOID,
		TANH,
		RELU,
		SELU,
		ELU,
		EXPONENTIAL,
		SOFTPLUS,
		SOFTSIGN,
		SOFTMAX
	};

	std::string toString(NonlinearityType t);
	NonlinearityType nonlinearityFromString(const std::string &str);

	std::ostream& operator<<(std::ostream &stream, NonlinearityType t);
	std::string operator+(const std::string &lhs, NonlinearityType rhs);
	std::string operator+(NonlinearityType lhs, const std::string &rhs);

	enum class SoftmaxMode
	{
		PER_INSTANCE, /**< The softmax operation is computed per image (N) across the dimensions H,W,C. */
		PER_CHANNEL /**< The softmax operation is computed per image (N) and spatial location (H,W) across dimension C. */
	};

	std::string toString(SoftmaxMode t);
	SoftmaxMode softmaxModeFromString(const std::string &str);

	std::ostream& operator<<(std::ostream &stream, SoftmaxMode t);
	std::string operator+(const std::string &lhs, SoftmaxMode rhs);
	std::string operator+(SoftmaxMode lhs, const std::string &rhs);

	namespace math
	{
		void activationForwardInPlace(const Context &context, NonlinearityType activation, Tensor &output);
		void activationBackwardInPlace(const Context &context, NonlinearityType activation, const Tensor &output, Tensor &gradientOut);

		void activationForward(const Context &context, NonlinearityType activation, Scalar alpha, const Tensor &input, Scalar beta, Tensor &output);
		void activationBackward(const Context &context, NonlinearityType activation, Scalar alpha, const Tensor &output, const Tensor &gradientOut,
				Scalar beta, Tensor &gradientIn);

		void softmaxForward(const Context &context, SoftmaxMode mode, Scalar alpha, const Tensor &input, Scalar beta, Tensor &output);
		void softmaxBackward(const Context &context, SoftmaxMode mode, Scalar alpha, const Tensor &output, const Tensor &gradientOut, Scalar beta,
				Tensor &gradientIn);
	}

} /* namespace avocado */

#endif /* AVOCADO_MATH_ACTIVATIONS_HPP_ */
