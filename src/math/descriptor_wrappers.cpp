/*
 * descriptor_wrappers.cpp
 *
 *  Created on: Jan 17, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/math/descriptor_wrappers.hpp>
#include <Avocado/backend/backend_libraries.hpp>
#include <Avocado/core/error_handling.hpp>
#include <Avocado/core/Device.hpp>
#include <Avocado/core/Shape.hpp>

#include <memory>
#include <algorithm>

namespace
{
	using namespace avocado;
	Device get_device(backend::av_int64 desc)
	{
		const backend::av_int64 device_type_mask = 0xFF00000000000000ull;
		const backend::av_int64 device_index_mask = 0x0000FFFF00000000ull;
		DeviceType type = static_cast<DeviceType>((desc & device_type_mask) >> 56ull);
		int index = static_cast<int>((desc & device_index_mask) >> 32ull);
		switch (type)
		{
			case DeviceType::CPU:
				return Device::cpu();
			case DeviceType::CUDA:
				return Device::cuda(index);
			case DeviceType::OPENCL:
				return Device::opencl(index);
			default:
				throw LogicError(METHOD_NAME, "invalid descriptor");
		}
	}
}

namespace avocado
{
	namespace internal
	{
		using namespace avocado::backend;

		/*
		 * Memory descriptor wrapper.
		 */
		MemoryDescWrapper::MemoryDescWrapper(Device device, size_t sizeInBytes)
		{
			switch (device.type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuCreateMemoryDescriptor(&m_descriptor, sizeInBytes);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaCreateMemoryDescriptor(&m_descriptor, device.index(), sizeInBytes);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclCreateMemoryDescriptor(&m_descriptor, device.index(), sizeInBytes);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		MemoryDescWrapper::MemoryDescWrapper(const MemoryDescWrapper &desc, size_t sizeInBytes, size_t offsetInBytes)
		{
			switch (desc.device().type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuCreateMemoryView(&m_descriptor, desc, sizeInBytes, offsetInBytes);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaCreateMemoryView(&m_descriptor, desc, sizeInBytes, offsetInBytes);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclCreateMemoryView(&m_descriptor, desc, sizeInBytes, offsetInBytes);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		MemoryDescWrapper::MemoryDescWrapper(MemoryDescWrapper &&other) :
				m_descriptor(other.m_descriptor)
		{
			other.m_descriptor = AVOCADO_NULL_DESCRIPTOR;
		}
		MemoryDescWrapper& MemoryDescWrapper::operator=(MemoryDescWrapper &&other)
		{
			std::swap(this->m_descriptor, other.m_descriptor);
			return *this;
		}
		MemoryDescWrapper::~MemoryDescWrapper()
		{
			if (m_descriptor == AVOCADO_NULL_DESCRIPTOR)
				return;
			avStatus_t status = AVOCADO_STATUS_SUCCESS;
			switch (device().type())
			{
				case DeviceType::CPU:
					status = cpuDestroyMemoryDescriptor(m_descriptor);
					break;
				case DeviceType::CUDA:
					status = cudaDestroyMemoryDescriptor(m_descriptor);
					break;
				case DeviceType::OPENCL:
//					status = openclDestroyMemoryDescriptor(m_descriptor);
					break;
			}
			if (status == AVOCADO_STATUS_FREE_FAILED)
			{
				std::cout << "free failed\n";
				exit(-1);
			}
		}
		Device MemoryDescWrapper::device() const noexcept
		{
			return get_device(m_descriptor);
		}

		/*
		 * Tensor descriptor wrapper.
		 */
		TensorDescWrapper::TensorDescWrapper(Device device)
		{
			switch (device.type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuCreateTensorDescriptor(&m_descriptor);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaCreateTensorDescriptor(&m_descriptor);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclCreateTensorDescriptor(&m_descriptor);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		TensorDescWrapper::TensorDescWrapper(TensorDescWrapper &&other) :
				m_descriptor(other.m_descriptor)
		{
			other.m_descriptor = AVOCADO_NULL_DESCRIPTOR;
		}
		TensorDescWrapper& TensorDescWrapper::operator=(TensorDescWrapper &&other)
		{
			std::swap(this->m_descriptor, other.m_descriptor);
			return *this;
		}
		TensorDescWrapper::~TensorDescWrapper()
		{
			if (m_descriptor == AVOCADO_NULL_DESCRIPTOR)
				return;
			avStatus_t status = AVOCADO_STATUS_SUCCESS;
			switch (device().type())
			{
				case DeviceType::CPU:
					status = cpuDestroyTensorDescriptor(m_descriptor);
					break;
				case DeviceType::CUDA:
					status = cudaDestroyTensorDescriptor(m_descriptor);
					break;
				case DeviceType::OPENCL:
//					status = openclDestroyTensorDescriptor(m_descriptor);
					break;
			}
			if (status == AVOCADO_STATUS_FREE_FAILED)
			{
				std::cout << "free failed\n";
				exit(-1);
			}
		}
		Device TensorDescWrapper::device() const noexcept
		{
			return get_device(m_descriptor);
		}
		void TensorDescWrapper::set(const Shape &shape, DataType dtype)
		{
			switch (device().type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuSetTensorDescriptor(m_descriptor, static_cast<avDataType_t>(dtype), shape.length(), shape.data());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaSetTensorDescriptor(m_descriptor, static_cast<avDataType_t>(dtype), shape.length(), shape.data());
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclSetTensorDescriptor(m_descriptor, static_cast<avDataType_t>(dtype), shape.length(), shape.data());
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		/*
		 * Convolution descriptor wrapper.
		 */
		ConvolutionDescWrapper::ConvolutionDescWrapper(Device device)
		{
			switch (device.type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuCreateConvolutionDescriptor(&m_descriptor);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaCreateConvolutionDescriptor(&m_descriptor);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclCreateConvolutionDescriptor(&m_descriptor);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		ConvolutionDescWrapper::ConvolutionDescWrapper(ConvolutionDescWrapper &&other) :
				m_descriptor(other.m_descriptor)
		{
			other.m_descriptor = AVOCADO_NULL_DESCRIPTOR;
		}
		ConvolutionDescWrapper& ConvolutionDescWrapper::operator=(ConvolutionDescWrapper &&other)
		{
			std::swap(this->m_descriptor, other.m_descriptor);
			return *this;
		}
		ConvolutionDescWrapper::~ConvolutionDescWrapper()
		{
			if (m_descriptor == AVOCADO_NULL_DESCRIPTOR)
				return;
			avStatus_t status = AVOCADO_STATUS_SUCCESS;
			switch (device().type())
			{
				case DeviceType::CPU:
					status = cpuDestroyConvolutionDescriptor(m_descriptor);
					break;
				case DeviceType::CUDA:
					status = cudaDestroyConvolutionDescriptor(m_descriptor);
					break;
				case DeviceType::OPENCL:
//					status = openclDestroyConvolutionDescriptor(m_descriptor);
					break;
			}
			if (status == AVOCADO_STATUS_FREE_FAILED)
			{
				std::cout << "free failed\n";
				exit(-1);
			}
		}
		Device ConvolutionDescWrapper::device() const noexcept
		{
			return get_device(m_descriptor);
		}
		void ConvolutionDescWrapper::set(ConvMode mode, int nbDims, const std::array<int, 3> &padding, const std::array<int, 3> &strides,
				const std::array<int, 3> &dilation, int groups, const std::array<uint8_t, 16> &paddingValue)
		{
			switch (device().type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuSetConvolutionDescriptor(m_descriptor, static_cast<avocado::backend::avConvolutionMode_t>(mode), nbDims,
							padding.data(), strides.data(), dilation.data(), groups, paddingValue.data());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaSetConvolutionDescriptor(m_descriptor, static_cast<avocado::backend::avConvolutionMode_t>(mode), nbDims,
							padding.data(), strides.data(), dilation.data(), groups, paddingValue.data());
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclSetTensorDescriptor(m_descriptor, static_cast<avocado::backend::avConvolutionMode_t>(mode), nbDims,
//					padding.data(), strides.data(), dilation.data(), groups, paddingValue.data());
//					CHECK_OPENCL_STATUS(status);
					break;
				}
			}
		}

		/*
		 * Convolution descriptor wrapper.
		 */
		OptimizerDescWrapper::OptimizerDescWrapper(Device device)
		{
			switch (device.type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuCreateOptimizerDescriptor(&m_descriptor);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaCreateOptimizerDescriptor(&m_descriptor);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclCreateOptimizerDescriptor(&m_descriptor);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		OptimizerDescWrapper::OptimizerDescWrapper(OptimizerDescWrapper &&other) :
				m_descriptor(other.m_descriptor)
		{
			other.m_descriptor = AVOCADO_NULL_DESCRIPTOR;
		}
		OptimizerDescWrapper& OptimizerDescWrapper::operator=(OptimizerDescWrapper &&other)
		{
			std::swap(this->m_descriptor, other.m_descriptor);
			return *this;
		}
		OptimizerDescWrapper::~OptimizerDescWrapper()
		{
			if (m_descriptor == AVOCADO_NULL_DESCRIPTOR)
				return;
			avStatus_t status = AVOCADO_STATUS_SUCCESS;
			switch (device().type())
			{
				case DeviceType::CPU:
					status = cpuDestroyOptimizerDescriptor(m_descriptor);
					break;
				case DeviceType::CUDA:
					status = cudaDestroyOptimizerDescriptor(m_descriptor);
					break;
				case DeviceType::OPENCL:
//					status = openclDestroyOptimizerDescriptor(m_descriptor);
					break;
			}
			if (status == AVOCADO_STATUS_FREE_FAILED)
			{
				std::cout << "free failed\n";
				exit(-1);
			}
		}
		Device OptimizerDescWrapper::device() const noexcept
		{
			return get_device(m_descriptor);
		}
		void OptimizerDescWrapper::set(OptimizerType type, int64_t steps, double learningRate, const std::array<double, 4> &coefficients,
				const std::array<bool, 4> &flags)
		{
			switch (device().type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuSetOptimizerDescriptor(m_descriptor, static_cast<avocado::backend::avOptimizerType_t>(type), steps,
							learningRate, coefficients.data(), flags.data());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaSetOptimizerDescriptor(m_descriptor, static_cast<avocado::backend::avOptimizerType_t>(type), steps,
							learningRate, coefficients.data(), flags.data());
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclSetOptimizerDescriptor(m_descriptor, static_cast<avocado::backend::avOptimizerType_t>(type), learningRate,
//					coefficients.data(), flags.data());
//					CHECK_OPENCL_STATUS(status);
					break;
				}
			}
		}

		void set_memory(avContextDescriptor_t context, avMemoryDescriptor_t mem, size_t dstOffset, size_t count, const void *pattern,
				size_t patternSizeInBytes)
		{
			switch (get_device(mem).type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuSetMemory(context, mem, dstOffset, count, pattern, patternSizeInBytes);
					CHECK_CPU_STATUS(status);
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaSetMemory(context, mem, dstOffset, count, pattern, patternSizeInBytes);
					CHECK_CUDA_STATUS(status);
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclSetMemory(context, mem, dstOffset, count, pattern, patternSizeInBytes);
//					CHECK_OPENCL_STATUS(status);
					break;
				}
			}
		}
		void copy_memory(avContextDescriptor_t context, avMemoryDescriptor_t dstMem, size_t dstOffset, const avMemoryDescriptor_t srcMem,
				size_t srcOffset, size_t count)
		{
			if (count == 0)
				return;
			switch (get_device(srcMem).type())
			{
				case DeviceType::CPU: // source device is CPU
				{
					switch (get_device(dstMem).type())
					{
						case DeviceType::CPU: // from CPU to CPU
						{
							avStatus_t status = cpuCopyMemory(context, dstMem, dstOffset, srcMem, srcOffset, count);
							CHECK_CPU_STATUS(status);
							break;
						}
						case DeviceType::CUDA: // from CPU to CUDA
						{
							avStatus_t status = cudaCopyMemoryFromHost(context, dstMem, dstOffset,
									reinterpret_cast<const int8_t*>(cpuGetMemoryPointer(srcMem)) + srcOffset, count);
							CHECK_CUDA_STATUS(status);
							break;
						}
						case DeviceType::OPENCL: // from CPU to OPENCL
						{
//							avStatus_t status = openclCopyMemoryFromHost(context, dstMem, dstOffset,
//									reinterpret_cast<const int8_t*>(cpuGetMemoryPointer(srcMem)) + srcOffset, count);
//							CHECK_OPENCL_STATUS(status);
							break;
						}
					}
					break;
				}
				case DeviceType::CUDA: // source device is CUDA
				{
					switch (get_device(dstMem).type())
					{
						case DeviceType::CPU: // from CUDA to CPU
						{
							avStatus_t status = cudaCopyMemoryToHost(context, reinterpret_cast<int8_t*>(cpuGetMemoryPointer(dstMem)) + dstOffset,
									srcMem, srcOffset, count);
							CHECK_CUDA_STATUS(status);
							break;
						}
						case DeviceType::CUDA: // from CUDA to CUDA
						{
							avStatus_t status = cudaCopyMemory(context, dstMem, dstOffset, srcMem, srcOffset, count);
							CHECK_CUDA_STATUS(status);
							break;
						}
						case DeviceType::OPENCL: // from CUDA to OPENCL
						{
							std::unique_ptr<int8_t[]> buffer = std::make_unique<int8_t[]>(count);
							avStatus_t status = cudaCopyMemoryToHost(context, buffer.get(), srcMem, srcOffset, count);
							CHECK_CUDA_STATUS(status);

//							status = openclCopyMemoryFromHost(context, dstMem, dstOffset, buffer.get(), count);
//							CHECK_OPENCL_STATUS(status);
							break;
						}
					}
					break;
				}
				case DeviceType::OPENCL: // source device is OPENCL
				{
//					switch (get_device(dstMem).type())
//					{
//						case DeviceType::CPU: // from OPENCL to CPU
//						{
//							avStatus_t status = openclCopyMemoryToHost(context, reinterpret_cast<int8_t*>(cpuGetMemoryPointer(dstMem)) + dstOffset,
//									srcMem, srcOffset, count);
//							CHECK_CUDA_STATUS(status);
//							break;
//						}
//						case DeviceType::CUDA: // from OPENCL to CUDA
//						{
//							std::unique_ptr<int8_t[]> buffer = std::make_unique<int8_t[]>(count);
//							avStatus_t status = openclCopyMemoryToHost(context, buffer.get(), srcMem, srcOffset, count);
//							CHECK_OPENCL_STATUS(status);
//
//							status = openclCopyMemoryFromHost(context, dstMem, dstOffset, buffer.get(), count);
//							CHECK_OPENCL_STATUS(status);
//							break;
//						}
//						case DeviceType::OPENCL: // from OPENCL to OPENCL
//						{
//							avStatus_t status = openclCopyMemory(context, dstMem, dstOffset, srcMem, srcOffset, count);
//							CHECK_OPENCL_STATUS(status);
//							break;
//						}
//					}
					break;
				}
			}
		}
		void change_type(avContextDescriptor_t context, avMemoryDescriptor_t dstMem, DataType dstType, const avMemoryDescriptor_t srcMem,
				DataType srcType, size_t elements)
		{
			switch (get_device(dstMem).type())
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuChangeType(context, dstMem, static_cast<avDataType_t>(dstType), srcMem, static_cast<avDataType_t>(srcType),
							elements);
					CHECK_CPU_STATUS(status);
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaChangeType(context, dstMem, static_cast<avDataType_t>(dstType), srcMem,
							static_cast<avDataType_t>(srcType), elements);
					CHECK_CUDA_STATUS(status);
					break;
				}
				case DeviceType::OPENCL:
				{
//					avStatus_t status = openclChangeType(context, dstMem, static_cast<avDataType_t>(dstType), srcMem,
//							static_cast<avDataType_t>(srcType), elements);
//					CHECK_OPENCL_STATUS(status);
					break;
				}
			}
		}

	} /* namespace internal */
} /* namespace avocado */

