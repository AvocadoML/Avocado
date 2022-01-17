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
		float nonlinearityForward(float input, NonlinearityType activation);
		float nonlinearityBackward(float gradient, float output, NonlinearityType activation);
		double nonlinearityForward(double input, NonlinearityType activation);
		double nonlinearityBackward(double gradient, double output, NonlinearityType activation);

		void activationForward(const Context &context, NonlinearityType activation, Scalar alpha1, Scalar alpha2, const Scalar beta,
				const Tensor &input, Tensor &output);
		void activationBackward(const Context &context, NonlinearityType activation, Scalar alpha, Scalar beta, Tensor &gradientPrev,
				const Tensor &gradientNext, const Tensor &output);

		void softmaxForward(const Context &context, SoftmaxMode mode, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output);
		void softmaxBackward(const Context &context, SoftmaxMode mode, Scalar alpha, Scalar beta, Tensor &gradientPrev, const Tensor &gradientNext,
				const Tensor &output);

//		void nonlinearityForwardInPlace(const Context &context, Tensor &output, NonlinearityType activation);
//		void nonlinearityBackwardInPlace(const Context &context, Tensor &gradient, const Tensor &output, NonlinearityType activation);
//		void nonlinearityForward(const Context &context, const Tensor &input, Tensor &output, NonlinearityType activation);
//		void nonlinearityBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &output,
//				NonlinearityType activation);
//
//		void softmaxForwardInPlace(const Context &context, Tensor &output);
//		void softmaxBackwardInPlace(const Context &context, Tensor &gradient, const Tensor &output);
//		void softmaxForward(const Context &context, const Tensor &input, Tensor &output);
//		void softmaxBackward(const Context &context, Tensor &gradient_prev, const Tensor &gradient_next, const Tensor &output);
	}

} /* namespace avocado */

#endif /* AVOCADO_MATH_ACTIVATIONS_HPP_ */
