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
		}

		Scalar calcLossFunction(const Context &context, LossType lossType, const Tensor &output, const Tensor &target)
		{
		}
		void calcLossGradient(const Context &context, LossType lossType, const Scalar alpha, const Scalar beta, Tensor &gradient,
				const Tensor &output, const Tensor &target, bool isFused)
		{
		}

		void calcOptimizerLearn(const Context &context, const OptimizerConfig &config, const Scalar alpha, const Scalar beta, Tensor &weight,
				const Tensor &update, Tensor &workspace1, Tensor &workspace2)
		{
		}

		void applyRegularizerL2(const Context &context, Tensor &gradient, const Tensor &weight, Tensor &update, const Scalar coefficient,
				const Scalar offset, Scalar loss)
		{
			if (not same_device(context, weight))
				throw DeviceMismatch(METHOD_NAME, context.device(), weight.device());

//			if (m_coefficient == 0.0f)
//				return;
//			backend::TensorDescriptor desc_gradient = weight.getUpdate().getDescriptor();
//			backend::TensorDescriptor desc_param = weight.getParam().getDescriptor();
//			backend::ScalarDescriptor desc_coefficient = Scalar(m_coefficient).getDescriptor();
//			backend::ScalarDescriptor desc_offset = Scalar(m_offset).getDescriptor();
//
//			switch (context.device().type())
//			{
//				case DeviceType::CPU:
//				{
//					backend::avStatus_t status = backend::cpuRegularizerL2(context.getDescriptor(), &desc_gradient, &desc_param, &desc_coefficient,
//							&desc_offset);
//					CHECK_CPU_STATUS(status);
//					break;
//				}
//				case DeviceType::CUDA:
//				{
//					backend::avStatus_t status = backend::cudaRegularizerL2(context.getDescriptor(), &desc_gradient, &desc_param, &desc_coefficient,
//							&desc_offset);
//					CHECK_CUDA_STATUS(status);
//					break;
//				}
//				case DeviceType::OPENCL:
//				{
//					backend::avStatus_t status = backend::openclRegularizerL2(context.getDescriptor(), &desc_gradient, &desc_param, &desc_coefficient,
//							&desc_offset);
//					CHECK_OPENCL_STATUS(status);
//					break;
//				}
//			}
		}
	}
}

