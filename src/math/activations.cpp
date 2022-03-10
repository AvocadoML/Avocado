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
#include <Avocado/backend/backend_libraries.hpp>

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
		void activationForwardInPlace(const Context &context, NonlinearityType activation, Tensor &output)
		{
			if (not same_device(context, output))
				throw DeviceMismatch(METHOD_NAME, "");

			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);
			backend::avTensorDescriptor_t yDesc = output.getDescriptor();
			backend::avMemoryDescriptor_t yMem = output.getMemory();
			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuActivationForward(context, act, nullptr, yDesc, yMem, nullptr, yDesc, yMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaActivationForward(context, act, nullptr, yDesc, yMem, nullptr, yDesc, yMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclActivationForward(context, act, nullptr, yDesc, yMem, nullptr, yDesc, yMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void activationBackwardInPlace(const Context &context, NonlinearityType activation, const Tensor &output, Tensor &gradientOut)
		{
			if (not same_device(context, output, gradientOut))
				throw DeviceMismatch(METHOD_NAME, "");

			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);

			backend::avTensorDescriptor_t yDesc = output.getDescriptor();
			backend::avTensorDescriptor_t dyDesc = gradientOut.getDescriptor();

			backend::avMemoryDescriptor_t yMem = output.getMemory();
			backend::avMemoryDescriptor_t dyMem = gradientOut.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuActivationBackward(context, act, nullptr, yDesc, yMem, dyDesc, dyMem, nullptr, dyDesc,
							dyMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaActivationBackward(context, act, nullptr, yDesc, yMem, dyDesc, dyMem, nullptr, dyDesc,
							dyMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclActivationBackward(context, act, nullptr, yDesc, yMem, dyDesc, dyMem, nullptr, dyDesc,
//							dyMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void activationForward(const Context &context, NonlinearityType activation, Scalar alpha, const Tensor &input, Scalar beta, Tensor &output)
		{
			if (not same_device(context, input, output))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(output.dtype());
			beta.toScalingTypeFor(output.dtype());
			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);

			backend::avTensorDescriptor_t xDesc = input.getDescriptor();
			backend::avTensorDescriptor_t yDesc = output.getDescriptor();

			backend::avMemoryDescriptor_t xMem = input.getMemory();
			backend::avMemoryDescriptor_t yMem = output.getMemory();
			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuActivationForward(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaActivationForward(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclActivationForward(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void activationBackward(const Context &context, NonlinearityType activation, Scalar alpha, const Tensor &output, const Tensor &gradientOut,
				Scalar beta, Tensor &gradientIn)
		{
			if (not same_device(context, output, gradientOut, gradientIn))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(output.dtype());
			beta.toScalingTypeFor(output.dtype());
			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);

			backend::avTensorDescriptor_t yDesc = output.getDescriptor();
			backend::avTensorDescriptor_t dxDesc = gradientIn.getDescriptor();
			backend::avTensorDescriptor_t dyDesc = gradientOut.getDescriptor();

			backend::avMemoryDescriptor_t yMem = output.getMemory();
			backend::avMemoryDescriptor_t dxMem = gradientIn.getMemory();
			backend::avMemoryDescriptor_t dyMem = gradientOut.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuActivationBackward(context, act, alpha.data(), yDesc, yMem, dyDesc, dyMem, beta.data(),
							dxDesc, dxMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaActivationBackward(context, act, alpha.data(), yDesc, yMem, dyDesc, dyMem, beta.data(),
							dxDesc, dxMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclActivationBackward(context, act, alpha.data(), yDesc, yMem, dyDesc, dyMem,
//							beta.data(), dxDesc, dxMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void softmaxForward(const Context &context, SoftmaxMode mode, Scalar alpha, const Tensor &input, Scalar beta, Tensor &output)
		{
			if (not same_device(context, input, output))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(output.dtype());
			beta.toScalingTypeFor(output.dtype());
			backend::avSoftmaxMode_t _mode = static_cast<backend::avSoftmaxMode_t>(mode);

			backend::avTensorDescriptor_t xDesc = input.getDescriptor();
			backend::avTensorDescriptor_t yDesc = output.getDescriptor();

			backend::avMemoryDescriptor_t xMem = input.getMemory();
			backend::avMemoryDescriptor_t yMem = output.getMemory();
			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuSoftmaxForward(context, _mode, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaSoftmaxForward(context, _mode, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclSoftmaxForward(context, _mode, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void softmaxBackward(const Context &context, SoftmaxMode mode, Scalar alpha, const Tensor &output, const Tensor &gradientOut, Scalar beta,
				Tensor &gradientIn)
		{
			if (not same_device(context, output, gradientOut, gradientIn))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(output.dtype());
			beta.toScalingTypeFor(output.dtype());
			backend::avSoftmaxMode_t _mode = static_cast<backend::avSoftmaxMode_t>(mode);

			backend::avTensorDescriptor_t yDesc = output.getDescriptor();
			backend::avTensorDescriptor_t dxDesc = gradientIn.getDescriptor();
			backend::avTensorDescriptor_t dyDesc = gradientOut.getDescriptor();

			backend::avMemoryDescriptor_t yMem = output.getMemory();
			backend::avMemoryDescriptor_t dxMem = gradientIn.getMemory();
			backend::avMemoryDescriptor_t dyMem = gradientOut.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuSoftmaxBackward(context, _mode, alpha.data(), yDesc, yMem, dyDesc, dyMem, beta.data(),
							dxDesc, dxMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaSoftmaxBackward(context, _mode, alpha.data(), yDesc, yMem, dyDesc, dyMem, beta.data(),
							dxDesc, dxMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclSoftmaxBackward(context, _mode, alpha.data(), yDesc, yMem, dyDesc, dyMem, beta.data(),
//							dxDesc, dxMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
	} /* namespace math */
} /* namespace avocado */

