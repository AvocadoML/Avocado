/*
 * training.cpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/math/training.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/core/Context.hpp>

#include <Avocado/backend/backend_libraries.hpp>

namespace avocado
{
	namespace math
	{

		Scalar calcMetricFunction(const Context &context, MetricType metricType, const Tensor &output, const Tensor &target)
		{
			if (not same_device(context, output, target))
				throw DeviceMismatch(METHOD_NAME, "");
			if (not same_shape(output, target))
				throw ShapeMismatch(METHOD_NAME, "");
			if (not same_type(output, target))
				throw DataTypeMismatch(METHOD_NAME, "");

			backend::avMetricType_t metric = static_cast<backend::avMetricType_t>(metricType);

			backend::avTensorDescriptor_t outputDesc = output.getDescriptor();
			backend::avMemoryDescriptor_t outputMem = output.getMemory();

			backend::avTensorDescriptor_t targetDesc = target.getDescriptor();
			backend::avMemoryDescriptor_t targetMem = target.getMemory();

			Scalar result(output.dtype());

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuMetricFunction(context, metric, outputDesc, outputMem, targetDesc, targetMem,
							result.data());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaMetricFunction(context, metric, outputDesc, outputMem, targetDesc, targetMem,
							result.data());
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclMetricFunction(context, metric, outputDesc, outputMem, targetDesc, targetMem, result.data());
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
			return result;
		}

		Scalar calcLossFunction(const Context &context, LossType lossType, const Tensor &output, const Tensor &target)
		{
			if (not same_device(context, output, target))
				throw DeviceMismatch(METHOD_NAME, "");
			if (not same_shape(output, target))
				throw ShapeMismatch(METHOD_NAME, "");
			if (not same_type(output, target))
				throw DataTypeMismatch(METHOD_NAME, "");

			backend::avLossType_t loss = static_cast<backend::avLossType_t>(lossType);

			backend::avTensorDescriptor_t outputDesc = output.getDescriptor();
			backend::avMemoryDescriptor_t outputMem = output.getMemory();

			backend::avTensorDescriptor_t targetDesc = target.getDescriptor();
			backend::avMemoryDescriptor_t targetMem = target.getMemory();

			Scalar result(output.dtype());

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuLossFunction(context, loss, outputDesc, outputMem, targetDesc, targetMem, result.data());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaLossFunction(context, loss, outputDesc, outputMem, targetDesc, targetMem,
							result.data());
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclLossFunction(context, loss, outputDesc, outputMem, targetDesc, targetMem, result.data());
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
			return result;
		}
		void calcLossGradient(const Context &context, LossType lossType, Scalar alpha, Scalar beta, Tensor &gradient, const Tensor &output,
				const Tensor &target, bool isFused)
		{
			if (not same_device(context, output, target))
				throw DeviceMismatch(METHOD_NAME, "");
			if (not same_shape(output, target))
				throw ShapeMismatch(METHOD_NAME, "");
			if (not same_type(output, target))
				throw DataTypeMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(gradient.dtype());
			beta.toScalingTypeFor(gradient.dtype());

			backend::avLossType_t loss = static_cast<backend::avLossType_t>(lossType);

			backend::avTensorDescriptor_t outputDesc = output.getDescriptor();
			backend::avMemoryDescriptor_t outputMem = output.getMemory();

			backend::avTensorDescriptor_t targetDesc = target.getDescriptor();
			backend::avMemoryDescriptor_t targetMem = target.getMemory();

			backend::avTensorDescriptor_t gradientDesc = target.getDescriptor();
			backend::avMemoryDescriptor_t gradientMem = target.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuLossGradient(context, loss, alpha.data(), outputDesc, outputMem, targetDesc, targetMem,
							beta.data(), gradientDesc, gradientMem, isFused);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaLossGradient(context, loss, alpha.data(), outputDesc, outputMem, targetDesc, targetMem,
							beta.data(), gradientDesc, gradientMem, isFused);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclLossGradient(context, loss, alpha.data(), outputDesc, outputMem, targetDesc,
//							targetMem, beta.data(), gradientDesc, gradientMem, isFused);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void calcOptimizerLearn(const Context &context, const OptimizerConfig &config, Scalar alpha, Scalar beta, Tensor &weight,
				const Tensor &update, Tensor &workspace)
		{
			if (not same_device(context, weight, update))
				throw DeviceMismatch(METHOD_NAME, "");
			if (not same_shape(weight, update))
				throw ShapeMismatch(METHOD_NAME, "");
			if (not same_type(weight, update))
				throw DataTypeMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(weight.dtype());
			beta.toScalingTypeFor(weight.dtype());

			backend::avTensorDescriptor_t wDesc = weight.getDescriptor();
			backend::avMemoryDescriptor_t wMem = weight.getMemory();

			backend::avTensorDescriptor_t dwDesc = update.getDescriptor();
			backend::avMemoryDescriptor_t dwMem = update.getMemory();

			backend::avMemoryDescriptor_t workspaceMem = workspace.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuOptimizerLearn(context, config, alpha.data(), dwDesc, dwMem, beta.data(), wDesc, wMem, workspaceMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaOptimizerLearn(context, config, alpha.data(), dwDesc, dwMem, beta.data(), wDesc, wMem, workspaceMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclOptimizerLearn(context, config, alpha.data(), dwDesc, dwMem, beta.data(), wDesc, wMem, workspaceMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		Scalar applyRegularizerL2(const Context &context, Tensor &gradient, const Tensor &weight, Tensor &update, Scalar scale, Scalar offset,
				bool calcLoss)
		{
			if (not same_device(context, weight))
				throw DeviceMismatch(METHOD_NAME, context.device(), weight.device());

			scale.toScalingTypeFor(weight.dtype());
			offset.toScalingTypeFor(weight.dtype());

			backend::avTensorDescriptor_t wDesc = weight.getDescriptor();
			backend::avMemoryDescriptor_t wMem = weight.getMemory();

			backend::avTensorDescriptor_t dwDesc = update.getDescriptor();
			backend::avMemoryDescriptor_t dwMem = update.getMemory();

			Scalar result;

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuRegularizerL2(context, dwDesc, dwMem, wDesc, wMem, scale.data(), offset.data(),
							calcLoss ? result.data() : nullptr);
					CHECK_CPU_STATUS(status);
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaRegularizerL2(context, dwDesc, dwMem, wDesc, wMem, scale.data(), offset.data(),
							calcLoss ? result.data() : nullptr);
					CHECK_CUDA_STATUS(status);
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclRegularizerL2(context, dwDesc, dwMem, wDesc, wMem, scale.data(), offset.data(),
//							calcLoss ? result.data() : nullptr);
//					CHECK_OPENCL_STATUS(status);
					break;
				}
			}
			return result;
		}
	}
}

