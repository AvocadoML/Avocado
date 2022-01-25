/*
 * wrappers.cpp
 *
 *  Created on: Jan 21, 2022
 *      Author: Maciej Kozarzewski
 */

#include "wrappers.hpp"
#if USE_CPU
#  include <CpuBackend/cpu_backend.h>
#elif USE_CUDA
#  include <CudaBackend/cuda_backend.h>
#elif USE_OPENCL
#  include <OpenCLBackend/opencl_backend.h>
#endif
#include <ReferenceBackend/reference_backend.h>
#include "../backend_descriptors.hpp"

#include <algorithm>

namespace
{
	size_t shape_volume(const std::vector<int> &shape) noexcept
	{
		if (shape.size() == 0)
			return 0;
		size_t result = 1;
		for (size_t i = 0; i < shape.size(); i++)
			result *= shape[i];
		return result;
	}
}

namespace avocado
{
	namespace backend
	{
		ContextWrapper::ContextWrapper(avDeviceIndex_t device)
		{
#if USE_CPU
			cpuCreateContextDescriptor(&m_desc);
#elif USE_CUDA
			cudaCreateContextDescriptor(&m_desc, device);
#elif USE_OPENCL
			openclCreateContextDescriptor(&m_desc, device);
#endif
			refCreateContextDescriptor(&m_ref_desc);
		}
		ContextWrapper::ContextWrapper(ContextWrapper &&other) noexcept :
				m_desc(other.m_desc),
				m_ref_desc(other.m_ref_desc)
		{
			other.m_desc = AVOCADO_INVALID_DESCRIPTOR;
			other.m_ref_desc = AVOCADO_INVALID_DESCRIPTOR;
		}
		ContextWrapper& ContextWrapper::operator=(ContextWrapper &&other) noexcept
		{
			std::swap(this->m_desc, other.m_desc);
			std::swap(this->m_ref_desc, other.m_ref_desc);
			return *this;
		}
		ContextWrapper::~ContextWrapper()
		{
			if (m_desc != AVOCADO_INVALID_DESCRIPTOR)
			{
#if USE_CPU
				cpuDestroyContextDescriptor(m_desc);
#elif USE_CUDA
				cudaDestroyContextDescriptor(m_desc);
#elif USE_OPENCL
				openclDestroyContextDescriptor(m_desc);
#endif
			}
			if (m_ref_desc != AVOCADO_INVALID_DESCRIPTOR)
				refDestroyContextDescriptor(m_ref_desc);
		}

		TensorWrapper::TensorWrapper(std::vector<int> shape, avDataType_t dtype, avDeviceIndex_t device) :
				m_device_index(device)
		{
#if USE_CPU
			auto size_in_bytes = shape_volume(shape) * cpu::dataTypeSize(dtype);
			cpuCreateTensorDescriptor(&m_tensor_descriptor);
			cpuSetTensorDescriptor(m_tensor_descriptor, dtype, shape.size(), shape.data());
			cpuCreateMemoryDescriptor(&m_memory_descriptor, size_in_bytes);
#elif USE_CUDA
			auto size_in_bytes = shape_volume(shape) * cuda::dataTypeSize(dtype);
			cudaCreateTensorDescriptor(&m_tensor_descriptor);
			cudaSetTensorDescriptor(m_tensor_descriptor, dtype, shape.size(), shape.data());
			cudaCreateMemoryDescriptor(&m_memory_descriptor, device, size_in_bytes);
#elif USE_OPENCL
			auto size_in_bytes = shape_volume(shape) * opencl::dataTypeSize(dtype);
			openclCreateTensorDescriptor(&m_tensor_descriptor);
			openclSetTensorDescriptor(m_tensor_descriptor, dtype, shape.size(), shape.data());
			openclCreateMemoryDescriptor(&m_memory_descriptor, device, size_in_bytes);
#endif
			refCreateTensorDescriptor(&m_ref_tensor_descriptor);
			refSetTensorDescriptor(m_ref_tensor_descriptor, dtype, shape.size(), shape.data());
			refCreateMemoryDescriptor(&m_ref_memory_descriptor, size_in_bytes);

			zeroall();
		}
		TensorWrapper::TensorWrapper(TensorWrapper &&other) noexcept :
				m_device_index(other.m_device_index),
				m_tensor_descriptor(other.m_tensor_descriptor),
				m_memory_descriptor(other.m_memory_descriptor),
				m_ref_tensor_descriptor(other.m_ref_tensor_descriptor),
				m_ref_memory_descriptor(other.m_ref_memory_descriptor)
		{
			other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
			other.m_tensor_descriptor = AVOCADO_INVALID_DESCRIPTOR;
			other.m_memory_descriptor = AVOCADO_INVALID_DESCRIPTOR;
			other.m_ref_tensor_descriptor = AVOCADO_INVALID_DESCRIPTOR;
			other.m_ref_memory_descriptor = AVOCADO_INVALID_DESCRIPTOR;
		}
		TensorWrapper& TensorWrapper::operator=(TensorWrapper &&other) noexcept
		{
			std::swap(this->m_device_index, other.m_device_index);
			std::swap(this->m_tensor_descriptor, other.m_tensor_descriptor);
			std::swap(this->m_memory_descriptor, other.m_memory_descriptor);
			std::swap(this->m_ref_tensor_descriptor, other.m_ref_tensor_descriptor);
			std::swap(this->m_ref_memory_descriptor, other.m_ref_memory_descriptor);
			return *this;
		}
		TensorWrapper::~TensorWrapper() noexcept
		{
#if USE_CPU
			if (m_tensor_descriptor != AVOCADO_INVALID_DESCRIPTOR)
				cpuDestroyTensorDescriptor(m_tensor_descriptor);
			if (m_memory_descriptor != AVOCADO_INVALID_DESCRIPTOR)
				cpuDestroyMemoryDescriptor(m_memory_descriptor);
#elif USE_CUDA
			if (m_tensor_descriptor != AVOCADO_INVALID_DESCRIPTOR)
				cudaDestroyTensorDescriptor(m_tensor_descriptor);
			if (m_memory_descriptor != AVOCADO_INVALID_DESCRIPTOR)
				cudaDestroyMemoryDescriptor(m_memory_descriptor);
#elif USE_OPENCL
			if (m_tensor_descriptor != AVOCADO_INVALID_DESCRIPTOR)
				openclDestroyTensorDescriptor(m_tensor_descriptor);
			if (m_memory_descriptor != AVOCADO_INVALID_DESCRIPTOR)
				openclDestroyMemoryDescriptor(m_memory_descriptor);
#endif
			if (m_ref_tensor_descriptor != AVOCADO_INVALID_DESCRIPTOR)
				refDestroyTensorDescriptor(m_ref_tensor_descriptor);
			if (m_ref_memory_descriptor != AVOCADO_INVALID_DESCRIPTOR)
				refDestroyMemoryDescriptor(m_ref_memory_descriptor);
		}

		avDataType_t TensorWrapper::dtype() const noexcept
		{
#if USE_CPU
			return cpu::getTensor(m_tensor_descriptor).dtype();
#elif USE_CUDA
			return cuda::getTensor(m_tensor_descriptor).dtype();
#elif USE_OPENCL
			return opencl::getTensor(m_tensor_descriptor).dtype();
#endif
		}
		size_t TensorWrapper::sizeInBytes() const noexcept
		{
#if USE_CPU
			return cpu::getTensor(m_tensor_descriptor).sizeInBytes();
#elif USE_CUDA
			return cuda::getTensor(m_tensor_descriptor).sizeInBytes();
#elif USE_OPENCL
			return opencl::getTensor(m_tensor_descriptor).sizeInBytes();
#endif
		}

		int TensorWrapper::numberOfDimensions() const noexcept
		{
#if USE_CPU
			return cpu::getTensor(m_tensor_descriptor).nbDims();
#elif USE_CUDA
			return cuda::getTensor(m_tensor_descriptor).nbDims();
#elif USE_OPENCL
			return opencl::getTensor(m_tensor_descriptor).nbDims();
#endif
		}
		int TensorWrapper::dimension(int idx) const noexcept
		{
#if USE_CPU
			return cpu::getTensor(m_tensor_descriptor).dimension(idx);
#elif USE_CUDA
			return cuda::getTensor(m_tensor_descriptor).dimension(idx);
#elif USE_OPENCL
			return opencl::getTensor(m_tensor_descriptor).dimension(idx);
#endif
		}
		int TensorWrapper::firstDim() const noexcept
		{
#if USE_CPU
			return cpu::getTensor(m_tensor_descriptor).firstDim();
#elif USE_CUDA
			return cuda::getTensor(m_tensor_descriptor).firstDim();
#elif USE_OPENCL
			return opencl::getTensor(m_tensor_descriptor).firstDim();
#endif
		}
		int TensorWrapper::lastDim() const noexcept
		{
#if USE_CPU
			return cpu::getTensor(m_tensor_descriptor).lastDim();
#elif USE_CUDA
			return cuda::getTensor(m_tensor_descriptor).lastDim();
#elif USE_OPENCL
			return opencl::getTensor(m_tensor_descriptor).lastDim();
#endif
		}
		int TensorWrapper::volume() const noexcept
		{
#if USE_CPU
			return cpu::getTensor(m_tensor_descriptor).volume();
#elif USE_CUDA
			return cuda::getTensor(m_tensor_descriptor).volume();
#elif USE_OPENCL
			return opencl::getTensor(m_tensor_descriptor).volume();
#endif
		}

		void TensorWrapper::synchronize() const
		{
			switch (m_sync)
			{
				case -1: // reference is more recent, copy to device
				{
#if USE_CPU
					std::memcpy(cpuGetMemoryPointer(m_memory_descriptor), refGetMemoryPointer(m_ref_memory_descriptor), sizeInBytes());
#elif USE_CUDA
					cudaCopyMemoryFromHost(cudaGetDefaultContext(m_device_index), m_memory_descriptor, dst_offset, refGetMemoryPointer(m_ref_memory_descriptor), sizeInBytes());
#elif USE_OPENCL
					openclCopyMemoryFromHost(openclGetDefaultContext(m_device_index), m_memory_descriptor, dst_offset, refGetMemoryPointer(m_ref_memory_descriptor), sizeInBytes());
#endif
					break;
				}
				case 0: // in sync
					break;
				case 1: // device is more recent, copy to reference
				{
#if USE_CPU
					std::memcpy(refGetMemoryPointer(m_ref_memory_descriptor), cpuGetMemoryPointer(m_memory_descriptor), sizeInBytes());
#elif USE_CUDA
					cudaCopyMemoryToHost(cudaGetDefaultContext(m_device_index), m_memory_descriptor, dst_offset, refGetMemoryPointer(m_ref_memory_descriptor), sizeInBytes());
#elif USE_OPENCL
					openclCopyMemoryToHost(openclGetDefaultContext(m_device_index), m_memory_descriptor, dst_offset, refGetMemoryPointer(m_ref_memory_descriptor), sizeInBytes());
#endif
					break;
				}
			}
			m_sync = 0;
		}
		void TensorWrapper::zeroall()
		{
			set_pattern(nullptr, 0);
		}
		void TensorWrapper::copyToHost(void *dst) const
		{
			copy_data_to_cpu(dst, 0, sizeInBytes());
		}
		void TensorWrapper::copyFromHost(const void *src)
		{
			copy_data_from_cpu(0, src, sizeInBytes());
		}

		size_t TensorWrapper::get_index(std::initializer_list<int> idx) const
		{
#if USE_CPU
			return cpu::getTensor(m_tensor_descriptor).getIndex(idx);
#elif USE_CUDA
			return cuda::getTensor(m_tensor_descriptor).getIndex(idx);
#elif USE_OPENCL
			return opencl::getTensor(m_tensor_descriptor).getIndex(idx);
#endif
		}
		void TensorWrapper::copy_data_to_cpu(void *dst, size_t src_offset, size_t count) const
		{
			synchronize();
			std::memcpy(dst, reinterpret_cast<const int8_t*>(refGetMemoryPointer(m_ref_memory_descriptor)) + src_offset, count);
		}
		void TensorWrapper::copy_data_from_cpu(size_t dst_offset, const void *src, size_t count)
		{
			std::memcpy(refGetMemoryPointer(m_ref_memory_descriptor), src, sizeInBytes());
#if USE_CPU
			std::memcpy(reinterpret_cast<int8_t*>(cpuGetMemoryPointer(m_memory_descriptor)) + dst_offset, src, count);
#elif USE_CUDA
			cudaCopyMemoryFromHost(cudaGetDefaultContext(m_device_index), m_memory_descriptor, dst_offset, src, count);
#elif USE_OPENCL
			openclCopyMemoryFromHost(openclGetDefaultContext(m_device_index), m_memory_descriptor, dst_offset, src, count);
#endif
		}
		void TensorWrapper::set_pattern(const void *pattern, size_t patternSize)
		{
			refSetMemory(0, m_ref_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#if USE_CPU
			cpuSetMemory(cpuGetDefaultContext(), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#elif USE_CUDA
			cudaSetMemory(cpuGetDefaultContext(m_device_index), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#elif USE_OPENCL
			openclSetMemory(cpuGetDefaultContext(m_device_index), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#endif
		}

	} /* namespace backend */
} /* namespace avocado */

