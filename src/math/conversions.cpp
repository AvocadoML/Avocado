/*
 * conversions.cpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/math/conversions.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/error_handling.hpp>
#include <Avocado/backend/backend_libraries.hpp>

namespace avocado
{
	namespace math
	{
		void changeType(void *dst, DataType dstType, const void *src, DataType srcType, size_t elements)
		{
			backend::avStatus_t status = backend::cpuChangeTypeHost(backend::cpuGetDefaultContext(), dst, static_cast<backend::avDataType_t>(dstType),
					src, static_cast<backend::avDataType_t>(srcType), elements);
			CHECK_CPU_STATUS(status);
		}

		void changeType(const Context &context, Tensor &dst, const Tensor &src)
		{
			if (not same_device(context, dst, src))
				throw DeviceMismatch(METHOD_NAME, "");

			backend::avMemoryDescriptor_t srcMem = src.getMemory();
			backend::avMemoryDescriptor_t dstMem = dst.getMemory();

			backend::avDataType_t srcType = static_cast<backend::avDataType_t>(src.dtype());
			backend::avDataType_t dstType = static_cast<backend::avDataType_t>(dst.dtype());

			const backend::av_int64 elements = dst.volume();
			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuChangeType(context, dstMem, dstType, srcMem, srcType, elements);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaChangeType(context, dstMem, dstType, srcMem, srcType, elements);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclChangeType(context, dstMem, dstType, srcMem, srcType, elements);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void changeType(const Context &context, const Tensor &tensor, DataType dstType)
		{
			if (not same_device(context, tensor))
				throw DeviceMismatch(METHOD_NAME, "");
			backend::avMemoryDescriptor_t srcMem = tensor.getMemory();

			backend::avDataType_t srcDtype = static_cast<backend::avDataType_t>(tensor.dtype());
			backend::avDataType_t dstDtype = static_cast<backend::avDataType_t>(dstType);

			const backend::av_int64 elements = tensor.volume();
			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuChangeType(context, srcMem, dstDtype, srcMem, srcDtype, elements);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaChangeType(context, srcMem, dstDtype, srcMem, srcDtype, elements);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclChangeType(context, srcMem, dstDtype, srcMem, srcDtype, elements);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
	} /* namespace math */
} /* namespace avocado */

