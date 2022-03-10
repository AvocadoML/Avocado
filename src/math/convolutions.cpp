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

#include <array>

namespace
{
	using namespace avocado;
	using namespace avocado::math;
	using namespace avocado::backend;

	void im_2_row(const Context &context, const ConvConfig &config, const avTensorDescriptor_t filterDesc, const avTensorDescriptor_t srcDesc,
			const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t rowDesc, avMemoryDescriptor_t rowMem)
	{
	}

	void convolution_implicit_gemm_forward(const Context &context, const ConvConfig &config, Scalar alpha1, const Tensor &input, const Tensor &filter,
			const Tensor &bias, Scalar alpha2, const Tensor &ext, Scalar beta, const Tensor &output, NonlinearityType activation)
	{
	}

	void convolution_winograd_fused_forward(const Context &context, const ConvConfig &config, Scalar alpha1, const Tensor &input,
			const Tensor &filter, const Tensor &bias, Scalar alpha2, const Tensor &ext, Scalar beta, const Tensor &output,
			NonlinearityType activation)
	{
	}

	std::array<int64_t, 5> get_winograd_transform_workspace_wize(const ConvConfig &config, const Tensor &input, const Tensor &filter)
	{
//		if (not same_device(input, filter))
//			throw DeviceMismatch(METHOD_NAME, "");
//
//		std::array<int64_t, 5> result;
//
//		backend::avTensorDescriptor_t xDesc = input.getDescriptor();
//		backend::avTensorDescriptor_t wDesc = filter.getDescriptor();

//		switch (input.device().type())
		{
//			case DeviceType::CPU:
//			{
//				backend::avStatus_t status = backend::cpuGetWinogradTransformWorkspaceSize(config, xDesc, wDesc, result.data());
//				CHECK_CPU_STATUS(status)
//				break;
//			}
//			case DeviceType::CUDA:
//			{
//				backend::avStatus_t status = backend::cudaActivationForward(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem);
//				CHECK_CUDA_STATUS(status)
//				break;
//			}
//			case DeviceType::OPENCL:
//			{
//				backend::avStatus_t status = backend::openclActivationForward(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem);
//				CHECK_OPENCL_STATUS(status)
//				break;
//			}
		}
//		return result;
	}

	void winograd_weight_transform(const Context &context, const ConvConfig &config, const Tensor &filter, Tensor &matrices)
	{
	}

	void winograd_input_transform(const Context &context, const ConvConfig &config, const Tensor &filter, const Tensor &input, Tensor &matrices)
	{
	}

	void winograd_output_transform(const Context &context, const ConvConfig &config, const Tensor &filter, Scalar alpha1, const Tensor &matrices,
			Tensor &output, const Tensor &bias, Scalar alpha2, const Tensor &ext, Scalar beta, NonlinearityType activation)
	{
	}

	void winograd_gradient_transform(const Context &context, const ConvConfig &config, const Tensor &filter, const Tensor &gradientNext,
			Tensor &matrices)
	{
	}

	void winograd_update_transform(const Context &context, const ConvConfig &config, Scalar alpha, const Tensor &matricesDesc, Scalar beta,
			Tensor &dwDesc)
	{
	}
}

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

