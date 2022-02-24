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
		ContextWrapper::ContextWrapper(avDeviceIndex_t device, bool isDefault, bool isSynchronized) :
				m_device_index(device), m_is_default(isDefault), m_is_synchronized(isDefault or isSynchronized)
		{
			if (m_is_default)
			{
#if USE_CPU
				m_desc = cpuGetDefaultContext();
#elif USE_CUDA
				m_desc = cudaGetDefaultContext(m_device_index);
#elif USE_OPENCL
				m_desc = openclGetDefaultContext(m_device_index);
#endif
			}
			else
			{
#if USE_CPU
				cpuCreateContextDescriptor(&m_desc);
#elif USE_CUDA
				cudaCreateContextDescriptor(&m_desc, device);
#elif USE_OPENCL
				openclCreateContextDescriptor(&m_desc, device);
#endif
			}
			refCreateContextDescriptor(&m_ref_desc);
		}
		ContextWrapper::ContextWrapper(ContextWrapper &&other) noexcept :
		m_desc(other.m_desc), m_ref_desc(other.m_ref_desc), m_device_index(other.m_device_index), m_is_default(other.m_is_default), m_is_synchronized(other.m_is_synchronized)
		{
			other.m_desc = AVOCADO_INVALID_DESCRIPTOR;
			other.m_ref_desc = AVOCADO_INVALID_DESCRIPTOR;
			other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
		}
		ContextWrapper& ContextWrapper::operator=(ContextWrapper &&other) noexcept
		{
			std::swap(this->m_desc, other.m_desc);
			std::swap(this->m_ref_desc, other.m_ref_desc);
			std::swap(this->m_device_index, other.m_device_index);
			std::swap(this->m_is_default, other.m_is_default);
			std::swap(this->m_is_synchronized, other.m_is_synchronized);
			return *this;
		}
		ContextWrapper::~ContextWrapper()
		{
			if (m_desc != AVOCADO_INVALID_DESCRIPTOR and not m_is_default)
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
		void ContextWrapper::synchronize() const
		{
#if USE_CPU
			cpuSynchronizeWithContext(m_desc);
#elif USE_CUDA
			cudaSynchronizeWithContext(m_desc);
#elif USE_OPENCL
			openclSynchronizeWithContext(m_desc);
#endif
		}

		TensorWrapper::TensorWrapper(std::vector<int> shape, avDataType_t dtype, avDeviceIndex_t device) :
				m_device_index(device)
		{
#if USE_CPU
			size_t size_in_bytes = shape_volume(shape) * cpu::dataTypeSize(dtype);
			cpuCreateTensorDescriptor(&m_tensor_descriptor);
			cpuSetTensorDescriptor(m_tensor_descriptor, dtype, shape.size(), shape.data());
			cpuCreateMemoryDescriptor(&m_memory_descriptor, size_in_bytes);
#elif USE_CUDA
			size_t size_in_bytes = shape_volume(shape) * cuda::dataTypeSize(dtype);
			cudaCreateTensorDescriptor(&m_tensor_descriptor);
			cudaSetTensorDescriptor(m_tensor_descriptor, dtype, shape.size(), shape.data());
			cudaCreateMemoryDescriptor(&m_memory_descriptor, device, size_in_bytes);
#elif USE_OPENCL
			size_t size_in_bytes = shape_volume(shape) * opencl::dataTypeSize(dtype);
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
					cudaCopyMemoryFromHost(cudaGetDefaultContext(m_device_index), m_memory_descriptor, 0, refGetMemoryPointer(m_ref_memory_descriptor),
							sizeInBytes());
#elif USE_OPENCL
					openclCopyMemoryFromHost(openclGetDefaultContext(m_device_index), m_memory_descriptor, 0, refGetMemoryPointer(m_ref_memory_descriptor), sizeInBytes());
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
					cudaCopyMemoryToHost(cudaGetDefaultContext(m_device_index), refGetMemoryPointer(m_ref_memory_descriptor), m_memory_descriptor, 0,
							sizeInBytes());
#elif USE_OPENCL
					openclCopyMemoryToHost(openclGetDefaultContext(m_device_index), refGetMemoryPointer(m_ref_memory_descriptor),m_memory_descriptor, 0, sizeInBytes());
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
			if (refGetMemoryPointer(m_ref_memory_descriptor) == nullptr)
				return;
			std::memcpy(dst, reinterpret_cast<const int8_t*>(refGetMemoryPointer(m_ref_memory_descriptor)) + src_offset, count);
		}
		void TensorWrapper::copy_data_from_cpu(size_t dst_offset, const void *src, size_t count)
		{
			if (refGetMemoryPointer(m_ref_memory_descriptor) == nullptr)
				return;
			std::memcpy(reinterpret_cast<int8_t*>(refGetMemoryPointer(m_ref_memory_descriptor)) + dst_offset, src, count);
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
			if (refGetMemoryPointer(m_ref_memory_descriptor) == nullptr)
				return;
			refSetMemory(0, m_ref_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#if USE_CPU
			cpuSetMemory(cpuGetDefaultContext(), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#elif USE_CUDA
			cudaSetMemory(cudaGetDefaultContext(m_device_index), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#elif USE_OPENCL
			openclSetMemory(openclGetDefaultContext(m_device_index), m_memory_descriptor, 0, sizeInBytes(), pattern, patternSize);
#endif
		}

		OptimizerWrapper::OptimizerWrapper(avDeviceIndex_t device)
		{
#if USE_CPU
			cpuCreateOptimizerDescriptor(&m_desc);
#elif USE_CUDA
			cudaCreateOptimizerDescriptor(&m_desc);
#elif USE_OPENCL
			openclCreateOptimizerDescriptor(&m_desc);
#endif
			refCreateOptimizerDescriptor(&m_ref_desc);
		}
		OptimizerWrapper::OptimizerWrapper(OptimizerWrapper &&other) noexcept :
		m_desc(other.m_desc),
		m_ref_desc(other.m_ref_desc)
		{
			other.m_desc = AVOCADO_INVALID_DESCRIPTOR;
			other.m_ref_desc = AVOCADO_INVALID_DESCRIPTOR;
		}
		OptimizerWrapper& OptimizerWrapper::operator=(OptimizerWrapper &&other) noexcept
		{
			std::swap(this->m_desc, other.m_desc);
			std::swap(this->m_ref_desc, other.m_ref_desc);
			return *this;
		}
		OptimizerWrapper::~OptimizerWrapper()
		{
			if (m_desc != AVOCADO_INVALID_DESCRIPTOR)
			{
#if USE_CPU
				cpuDestroyOptimizerDescriptor(m_desc);
#elif USE_CUDA
				cudaDestroyOptimizerDescriptor(m_desc);
#elif USE_OPENCL
				openclDestroyOptimizerDescriptor(m_desc);
#endif
			}
			if (m_ref_desc != AVOCADO_INVALID_DESCRIPTOR)
				refDestroyOptimizerDescriptor(m_ref_desc);
		}
		void OptimizerWrapper::set(avOptimizerType_t type, double learningRate, const std::array<double, 4> &coefficients, const std::array<bool, 4> &flags)
		{
#if USE_CPU
			cpuSetOptimizerDescriptor(m_desc, type, learningRate, coefficients.data(), flags.data());
#elif USE_CUDA
			cudaSetOptimizerDescriptor(m_desc, type, learningRate, coefficients.data(), flags.data());
#elif USE_OPENCL
			openclSetOptimizerDescriptor(m_desc, type, learningRate, coefficients.data(), flags.data());
#endif
			refSetOptimizerDescriptor(m_ref_desc, type, learningRate, coefficients.data(), flags.data());
		}
		size_t OptimizerWrapper::getWorkspaceSize(const TensorWrapper &weights)
		{
			avSize_t result;
			refGetOptimizerWorkspaceSize(m_ref_desc, weights.getRefDescriptor(), &result);
			return result;
		}

		ConvolutionWrapper::ConvolutionWrapper(avDeviceIndex_t device, int nbDims) :
				nbDims(nbDims)
		{
#if USE_CPU
			cpuCreateConvolutionDescriptor(&m_desc);
#elif USE_CUDA
			cudaCreateConvolutionDescriptor(&m_desc);
#elif USE_OPENCL
			openclCreateConvolutionDescriptor(&m_desc);
#endif
			refCreateConvolutionDescriptor(&m_ref_desc);
		}
		ConvolutionWrapper::ConvolutionWrapper(ConvolutionWrapper &&other) noexcept :
		nbDims(other.nbDims),
		m_desc(other.m_desc),
		m_ref_desc(other.m_ref_desc)
		{
			other.m_desc = AVOCADO_INVALID_DESCRIPTOR;
			other.m_ref_desc = AVOCADO_INVALID_DESCRIPTOR;
		}
		ConvolutionWrapper& ConvolutionWrapper::operator=(ConvolutionWrapper &&other) noexcept
		{
			std::swap(this->nbDims, other.nbDims);
			std::swap(this->m_desc, other.m_desc);
			std::swap(this->m_ref_desc, other.m_ref_desc);
			return *this;
		}
		ConvolutionWrapper::~ConvolutionWrapper()
		{
			if (m_desc != AVOCADO_INVALID_DESCRIPTOR)
			{
#if USE_CPU
				cpuDestroyConvolutionDescriptor(m_desc);
#elif USE_CUDA
				cudaDestroyConvolutionDescriptor(m_desc);
#elif USE_OPENCL
				openclDestroyConvolutionDescriptor(m_desc);
#endif
			}
			if (m_ref_desc != AVOCADO_INVALID_DESCRIPTOR)
				refDestroyConvolutionDescriptor(m_ref_desc);
		}
		void ConvolutionWrapper::set(avConvolutionMode_t mode, const std::array<int, 3> &padding, const std::array<int, 3> &strides,
				const std::array<int, 3> &dilation, int groups, const void *paddingValue)
		{
#if USE_CPU
			cpuSetConvolutionDescriptor(m_desc, mode, nbDims, padding.data(), strides.data(), dilation.data(), groups, paddingValue);
#elif USE_CUDA
			cudaSetConvolutionDescriptor(m_desc, mode, nbDims, padding.data(), strides.data(), dilation.data(), groups, paddingValue);
#elif USE_OPENCL
			openclSetConvolutionDescriptor(m_desc, mode, nbDims, padding.data(), strides.data(), dilation.data(), groups, paddingValue);
#endif
			refSetConvolutionDescriptor(m_ref_desc, mode, nbDims, padding.data(), strides.data(), dilation.data(), groups, paddingValue);
		}
		std::vector<int> ConvolutionWrapper::getOutputShape(const TensorWrapper &input, const TensorWrapper &weights)
		{
#if USE_CPU
			cpu::TensorDescriptor tmp = cpu::getConvolution(m_desc).getOutputShape(cpu::getTensor(input.getDescriptor()),
					cpu::getTensor(weights.getDescriptor()));
#elif USE_CUDA
			cuda::TensorDescriptor tmp = cuda::getConvolution(m_desc).getOutputShape(cuda::getTensor(input.getDescriptor()),
					cuda::getTensor(weights.getDescriptor()));
#elif USE_OPENCL
			opencl::TensorDescriptor tmp = opencl::getConvolution(m_desc).getOutputShape(opencl::getTensor(input.getDescriptor()),
					opencl::getTensor(weights.getDescriptor()));
#endif
			int size;
			tmp.get(nullptr, &size, nullptr);
			std::vector<int> result(size);
			tmp.get(nullptr, nullptr, result.data());
			return result;
		}
	} /* namespace backend */
} /* namespace avocado */

