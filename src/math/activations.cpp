/*
 * activations.cpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/math/activations.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>

namespace avocado
{

	std::string toString(NonlinearityType t)
	{
		switch (t)
		{
			default:
			case NonlinearityType::LINEAR:
				return "linear";
			case NonlinearityType::SIGMOID:
				return "sigmoid";
			case NonlinearityType::TANH:
				return "tanh";
			case NonlinearityType::RELU:
				return "relu";
			case NonlinearityType::SELU:
				return "selu";
			case NonlinearityType::ELU:
				return "elu";
			case NonlinearityType::EXPONENTIAL:
				return "exponential";
			case NonlinearityType::SOFTPLUS:
				return "softplus";
			case NonlinearityType::SOFTSIGN:
				return "softsign";
			case NonlinearityType::SOFTMAX:
				return "softmax";
		}
	}
	NonlinearityType nonlinearityFromString(const std::string &str)
	{
		if (str == "linear")
			return NonlinearityType::LINEAR;
		if (str == "sigmoid")
			return NonlinearityType::SIGMOID;
		if (str == "tanh")
			return NonlinearityType::TANH;
		if (str == "relu")
			return NonlinearityType::RELU;
		if (str == "selu")
			return NonlinearityType::SELU;
		if (str == "elu")
			return NonlinearityType::ELU;
		if (str == "exponential")
			return NonlinearityType::EXPONENTIAL;
		if (str == "softplus")
			return NonlinearityType::SOFTPLUS;
		if (str == "softsign")
			return NonlinearityType::SOFTSIGN;
		if (str == "softmax")
			return NonlinearityType::SOFTMAX;
		throw LogicError(METHOD_NAME, "unknown nonlinearity '" + str + "'");
	}

	std::ostream& operator<<(std::ostream &stream, NonlinearityType t)
	{
		stream << toString(t);
		return stream;
	}
	std::string operator+(const std::string &lhs, NonlinearityType rhs)
	{
		return lhs + toString(rhs);
	}
	std::string operator+(NonlinearityType lhs, const std::string &rhs)
	{
		return toString(lhs) + rhs;
	}

	std::string toString(SoftmaxMode t)
	{
		switch (t)
		{
			default:
			case SoftmaxMode::PER_CHANNEL:
				return "per_channel";
			case SoftmaxMode::PER_INSTANCE:
				return "per_instance";
		}
	}
	SoftmaxMode softmaxModeFromString(const std::string &str)
	{
		if (str == "per_channel")
			return SoftmaxMode::PER_CHANNEL;
		if (str == "per_instance")
			return SoftmaxMode::PER_INSTANCE;
		throw LogicError(METHOD_NAME, "unknown softmax mode '" + str + "'");
	}

	std::ostream& operator<<(std::ostream &stream, SoftmaxMode t)
	{
		stream << toString(t);
		return stream;
	}
	std::string operator+(const std::string &lhs, SoftmaxMode rhs)
	{
		return lhs + toString(rhs);
	}
	std::string operator+(SoftmaxMode lhs, const std::string &rhs)
	{
		return toString(lhs) + rhs;
	}

	namespace math
	{
		void activationForward(const Context &context, NonlinearityType activation, Scalar alpha1, Scalar alpha2, const Scalar beta,
				const Tensor &input, Tensor &output)
		{
		}
		void activationBackward(const Context &context, NonlinearityType activation, Scalar alpha, Scalar beta, Tensor &gradientPrev,
				const Tensor &gradientNext, const Tensor &output)
		{
		}

		void softmaxForward(const Context &context, SoftmaxMode mode, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output)
		{
		}
		void softmaxBackward(const Context &context, SoftmaxMode mode, Scalar alpha, Scalar beta, Tensor &gradientPrev, const Tensor &gradientNext,
				const Tensor &output)
		{
		}

	} /* namespace math */
} /* namespace avocado */

