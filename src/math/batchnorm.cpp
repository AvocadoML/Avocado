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
#include <Avocado/backend/backend_libraries.hpp>

namespace avocado
{
	namespace math
	{
		void affineForward(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &weight,
				const Tensor &bias, NonlinearityType activation)
		{
			if (not same_device(context, input, output, weight, bias))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(output.dtype());
			beta.toScalingTypeFor(output.dtype());
			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);

			backend::avTensorDescriptor_t xDesc = input.getDescriptor();
			backend::avTensorDescriptor_t yDesc = output.getDescriptor();
			backend::avTensorDescriptor_t wDesc = weight.getDescriptor();
			backend::avTensorDescriptor_t bDesc = bias.getDescriptor();

			backend::avMemoryDescriptor_t xMem = input.getMemory();
			backend::avMemoryDescriptor_t yMem = output.getMemory();
			backend::avMemoryDescriptor_t wMem = weight.getMemory();
			backend::avMemoryDescriptor_t bMem = bias.getMemory();
			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuAffineForward(context, act, wDesc, wMem, bDesc, bMem, alpha.data(), xDesc, xMem,
							beta.data(), yDesc, yMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaAffineForward(context, act, wDesc, wMem, bDesc, bMem, alpha.data(), xDesc, xMem,
							beta.data(), yDesc, yMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclAffineForward(context, act, wDesc, wMem, bDesc, bMem, alpha.data(), xDesc, xMem,
//							beta.data(), yDesc, yMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void batchNormInference(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &scale,
				const Tensor &bias, const Tensor &estimatedMean, const Tensor &estimatedVariance, double epsilon, NonlinearityType activation)
		{
			if (not same_device(context, input, output, scale, bias, estimatedMean, estimatedVariance))
				throw DeviceMismatch(METHOD_NAME, "");
			if (not same_shape(scale, bias, estimatedMean, estimatedVariance))
				throw ShapeMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(output.dtype());
			beta.toScalingTypeFor(output.dtype());
			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);

			backend::avTensorDescriptor_t xDesc = input.getDescriptor();
			backend::avTensorDescriptor_t yDesc = output.getDescriptor();
			backend::avTensorDescriptor_t scaleDesc = scale.getDescriptor();

			backend::avMemoryDescriptor_t xMem = input.getMemory();
			backend::avMemoryDescriptor_t yMem = output.getMemory();
			backend::avMemoryDescriptor_t scaleMem = scale.getMemory();
			backend::avMemoryDescriptor_t biasMem = bias.getMemory();
			backend::avMemoryDescriptor_t meanMem = estimatedMean.getMemory();
			backend::avMemoryDescriptor_t varMem = estimatedVariance.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuBatchNormInference(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem,
							scaleDesc, scaleMem, biasMem, meanMem, varMem, epsilon);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaBatchNormInference(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem,
							scaleDesc, scaleMem, biasMem, meanMem, varMem, epsilon);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclBatchNormInference(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem,
//							scaleDesc, scaleMem, biasMem, meanMem, varMem, epsilon);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void batchNormForward(const Context &context, Scalar alpha, Scalar beta, const Tensor &input, Tensor &output, const Tensor &scale,
				const Tensor &bias, Tensor &savedMean, Tensor &savedVariance, double epsilon, NonlinearityType activation)
		{
			if (not same_device(context, input, output, scale, bias, savedMean, savedVariance))
				throw DeviceMismatch(METHOD_NAME, "");
			if (not same_shape(scale, bias, savedMean, savedVariance))
				throw ShapeMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(output.dtype());
			beta.toScalingTypeFor(output.dtype());
			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);

			backend::avTensorDescriptor_t xDesc = input.getDescriptor();
			backend::avTensorDescriptor_t yDesc = output.getDescriptor();
			backend::avTensorDescriptor_t scaleDesc = scale.getDescriptor();

			backend::avMemoryDescriptor_t xMem = input.getMemory();
			backend::avMemoryDescriptor_t yMem = output.getMemory();
			backend::avMemoryDescriptor_t scaleMem = scale.getMemory();
			backend::avMemoryDescriptor_t biasMem = bias.getMemory();
			backend::avMemoryDescriptor_t meanMem = savedMean.getMemory();
			backend::avMemoryDescriptor_t varMem = savedVariance.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuBatchNormForward(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem,
							scaleDesc, scaleMem, biasMem, meanMem, varMem, epsilon);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaBatchNormForward(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem,
							scaleDesc, scaleMem, biasMem, meanMem, varMem, epsilon);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclBatchNormForward(context, act, alpha.data(), xDesc, xMem, beta.data(), yDesc, yMem,
//							scaleDesc, scaleMem, biasMem, meanMem, varMem, epsilon);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void batchNormBackward(const Context &context, Scalar alpha, const Tensor &input, const Tensor &output, Scalar beta, Tensor &gradientIn,
				Tensor &gradientOut, const Tensor &scale, const Tensor &mean, const Tensor &variance, Scalar alpha2, Scalar beta2,
				Tensor &scaleUpdate, Tensor &biasUpdate, double epsilon, NonlinearityType activation)
		{
			if (not same_device(context, input, output, scale, mean, variance, scaleUpdate, biasUpdate))
				throw DeviceMismatch(METHOD_NAME, "");
			if (not same_shape(scale, mean, variance, scaleUpdate, biasUpdate))
				throw ShapeMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(output.dtype());
			beta.toScalingTypeFor(output.dtype());
			alpha2.toScalingTypeFor(output.dtype());
			beta2.toScalingTypeFor(output.dtype());
			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);

			backend::avTensorDescriptor_t xDesc = input.getDescriptor();
			backend::avTensorDescriptor_t dxDesc = gradientIn.getDescriptor();
			backend::avTensorDescriptor_t yDesc = output.getDescriptor();
			backend::avTensorDescriptor_t dyDesc = gradientOut.getDescriptor();
			backend::avTensorDescriptor_t scaleDesc = scale.getDescriptor();

			backend::avMemoryDescriptor_t xMem = input.getMemory();
			backend::avMemoryDescriptor_t dxMem = gradientIn.getMemory();
			backend::avMemoryDescriptor_t yMem = output.getMemory();
			backend::avMemoryDescriptor_t dyMem = gradientOut.getMemory();
			backend::avMemoryDescriptor_t scaleMem = scale.getMemory();
			backend::avMemoryDescriptor_t meanMem = mean.getMemory();
			backend::avMemoryDescriptor_t varMem = variance.getMemory();
			backend::avMemoryDescriptor_t scaleUpdateMem = scaleUpdate.getMemory();
			backend::avMemoryDescriptor_t biasUpdateMem = biasUpdate.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuBatchNormBackward(context, act, alpha.data(), xDesc, xMem, yDesc, yMem, beta.data(),
							dxDesc, dxMem, dyDesc, dyMem, scaleDesc, scaleMem, meanMem, varMem, alpha2.data(), beta2.data(), scaleUpdateMem,
							biasUpdateMem, epsilon);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaBatchNormBackward(context, act, alpha.data(), xDesc, xMem, yDesc, yMem, beta.data(),
							dxDesc, dxMem, dyDesc, dyMem, scaleDesc, scaleMem, meanMem, varMem, alpha2.data(), beta2.data(), scaleUpdateMem,
							biasUpdateMem, epsilon);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclBatchNormBackward(context, act, alpha.data(), xDesc, xMem, yDesc, yMem, beta.data(),
//							dxDesc, dxMem, dyDesc, dyMem, scaleDesc, scaleMem, meanMem, varMem, alpha2.data(), beta2.data(), scaleUpdateMem,
//							biasUpdateMem, epsilon);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

	} /* namespace math */
} /* namespace avocado */

