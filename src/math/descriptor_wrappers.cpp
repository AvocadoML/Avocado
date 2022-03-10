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
	DeviceType get_device_type(backend::av_int64 desc) noexcept
	{
		const backend::av_int64 device_type_mask = 0xFF00000000000000ull;
		return static_cast<DeviceType>((desc & device_type_mask) >> 56ull);
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
					avStatus_t status = openclCreateMemoryDescriptor(&m_descriptor, device.index(), sizeInBytes);
					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		MemoryDescWrapper::MemoryDescWrapper(const MemoryDescWrapper &desc, size_t sizeInBytes, size_t offsetInBytes)
		{
			switch (get_device_type(desc.m_descriptor))
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
					avStatus_t status = openclCreateMemoryView(&m_descriptor, desc, sizeInBytes, offsetInBytes);
					CHECK_OPENCL_STATUS(status)
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
			avStatus_t status = AVOCADO_STATUS_SUCCESS;
			switch (get_device_type(m_descriptor))
			{
				case DeviceType::CPU:
					status = cpuDestroyMemoryDescriptor(m_descriptor);
					break;
				case DeviceType::CUDA:
					status = cudaDestroyMemoryDescriptor(m_descriptor);
					break;
				case DeviceType::OPENCL:
					status = openclDestroyMemoryDescriptor(m_descriptor);
					break;
			}
			if (status == AVOCADO_STATUS_FREE_FAILED)
			{
				std::cout << "free failed\n";
				exit(-1);
			}
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
					avStatus_t status = openclCreateTensorDescriptor(&m_descriptor);
					CHECK_OPENCL_STATUS(status)
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
			avStatus_t status = AVOCADO_STATUS_SUCCESS;
			switch (get_device_type(m_descriptor))
			{
				case DeviceType::CPU:
					status = cpuDestroyTensorDescriptor(m_descriptor);
					break;
				case DeviceType::CUDA:
					status = cudaDestroyTensorDescriptor(m_descriptor);
					break;
				case DeviceType::OPENCL:
					status = openclDestroyTensorDescriptor(m_descriptor);
					break;
			}
			if (status == AVOCADO_STATUS_FREE_FAILED)
			{
				std::cout << "free failed\n";
				exit(-1);
			}
		}
		void TensorDescWrapper::set(const Shape &shape, DataType dtype)
		{
			switch (get_device_type(m_descriptor))
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
					avStatus_t status = openclSetTensorDescriptor(m_descriptor, static_cast<avDataType_t>(dtype), shape.length(), shape.data());
					CHECK_OPENCL_STATUS(status)
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
					avStatus_t status = openclCreateConvolutionDescriptor(&m_descriptor);
					CHECK_OPENCL_STATUS(status)
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
			avStatus_t status = AVOCADO_STATUS_SUCCESS;
			switch (get_device_type(m_descriptor))
			{
				case DeviceType::CPU:
					status = cpuDestroyConvolutionDescriptor(m_descriptor);
					break;
				case DeviceType::CUDA:
					status = cudaDestroyConvolutionDescriptor(m_descriptor);
					break;
				case DeviceType::OPENCL:
					status = openclDestroyConvolutionDescriptor(m_descriptor);
					break;
			}
			if (status == AVOCADO_STATUS_FREE_FAILED)
			{
				std::cout << "free failed\n";
				exit(-1);
			}
		}
		void ConvolutionDescWrapper::set(ConvMode mode, int nbDims, const std::array<int, 3> &padding, const std::array<int, 3> &strides,
				const std::array<int, 3> &dilation, int groups, const std::array<uint8_t, 16> &paddingValue)
		{
			switch (get_device_type(m_descriptor))
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
					avStatus_t status = openclCreateOptimizerDescriptor(&m_descriptor);
					CHECK_OPENCL_STATUS(status)
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
			avStatus_t status = AVOCADO_STATUS_SUCCESS;
			switch (get_device_type(m_descriptor))
			{
				case DeviceType::CPU:
					status = cpuDestroyOptimizerDescriptor(m_descriptor);
					break;
				case DeviceType::CUDA:
					status = cudaDestroyOptimizerDescriptor(m_descriptor);
					break;
				case DeviceType::OPENCL:
					status = openclDestroyOptimizerDescriptor(m_descriptor);
					break;
			}
			if (status == AVOCADO_STATUS_FREE_FAILED)
			{
				std::cout << "free failed\n";
				exit(-1);
			}
		}
		void OptimizerDescWrapper::set(OptimizerType type, double learningRate, const std::array<double, 4> &coefficients,
				const std::array<bool, 4> &flags)
		{
			switch (get_device_type(m_descriptor))
			{
				case DeviceType::CPU:
				{
					avStatus_t status = cpuSetOptimizerDescriptor(m_descriptor, static_cast<avocado::backend::avOptimizerType_t>(type), learningRate,
							coefficients.data(), flags.data());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					avStatus_t status = cudaSetOptimizerDescriptor(m_descriptor, static_cast<avocado::backend::avOptimizerType_t>(type), learningRate,
							coefficients.data(), flags.data());
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
			switch (get_device_type(mem))
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
					avStatus_t status = openclSetMemory(context, mem, dstOffset, count, pattern, patternSizeInBytes);
					CHECK_OPENCL_STATUS(status);
					break;
				}
			}
		}
		void copy_memory(avContextDescriptor_t context, avMemoryDescriptor_t dstMem, size_t dstOffset, const avMemoryDescriptor_t srcMem,
				size_t srcOffset, size_t count)
		{
			if (count == 0)
				return;
			switch (get_device_type(srcMem))
			{
				case DeviceType::CPU: // source device is CPU
				{
					switch (get_device_type(dstMem))
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
							avStatus_t status = openclCopyMemoryFromHost(context, dstMem, dstOffset,
									reinterpret_cast<const int8_t*>(cpuGetMemoryPointer(srcMem)) + srcOffset, count);
							CHECK_OPENCL_STATUS(status);
							break;
						}
					}
					break;
				}
				case DeviceType::CUDA: // source device is CUDA
				{
					switch (get_device_type(dstMem))
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

							status = openclCopyMemoryFromHost(context, dstMem, dstOffset, buffer.get(), count);
							CHECK_OPENCL_STATUS(status);
							break;
						}
					}
					break;
				}
				case DeviceType::OPENCL: // source device is OPENCL
				{
					switch (get_device_type(dstMem))
					{
						case DeviceType::CPU: // from OPENCL to CPU
						{
							avStatus_t status = openclCopyMemoryToHost(context, reinterpret_cast<int8_t*>(cpuGetMemoryPointer(dstMem)) + dstOffset,
									srcMem, srcOffset, count);
							CHECK_CUDA_STATUS(status);
							break;
						}
						case DeviceType::CUDA: // from OPENCL to CUDA
						{
							std::unique_ptr<int8_t[]> buffer = std::make_unique<int8_t[]>(count);
							avStatus_t status = openclCopyMemoryToHost(context, buffer.get(), srcMem, srcOffset, count);
							CHECK_OPENCL_STATUS(status);

							status = openclCopyMemoryFromHost(context, dstMem, dstOffset, buffer.get(), count);
							CHECK_OPENCL_STATUS(status);
							break;
						}
						case DeviceType::OPENCL: // from OPENCL to OPENCL
						{
							avStatus_t status = openclCopyMemory(context, dstMem, dstOffset, srcMem, srcOffset, count);
							CHECK_OPENCL_STATUS(status);
							break;
						}
					}
					break;
				}
			}
		}
		void change_type(avContextDescriptor_t context, avMemoryDescriptor_t dstMem, DataType dstType, const avMemoryDescriptor_t srcMem,
				DataType srcType, size_t elements)
		{
			switch (get_device_type(dstMem))
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
					avStatus_t status = openclChangeType(context, dstMem, static_cast<avDataType_t>(dstType), srcMem,
							static_cast<avDataType_t>(srcType), elements);
					CHECK_OPENCL_STATUS(status);
					break;
				}
			}
		}

	} /* namespace internal */
} /* namespace avocado */

