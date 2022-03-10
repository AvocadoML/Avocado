/*
 * descriptor_wrappers.hpp
 *
 *  Created on: Jan 17, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_DESCRIPTOR_WRAPPERS_HPP_
#define AVOCADO_MATH_DESCRIPTOR_WRAPPERS_HPP_

#include <Avocado/backend/backend_defs.h>
#include <Avocado/core/Device.hpp>
#include <cstdint>
#include <cstddef>
#include <array>

namespace avocado
{
	class Device;
	class Shape;
	enum class DataType;
	enum class ConvMode;
	enum class OptimizerType;
}

namespace avocado
{
	namespace internal
	{
		class MemoryDescWrapper
		{
				backend::avMemoryDescriptor_t m_descriptor = backend::AVOCADO_NULL_DESCRIPTOR;
			public:
				MemoryDescWrapper() = default;
				MemoryDescWrapper(Device device, size_t sizeInBytes);
				MemoryDescWrapper(const MemoryDescWrapper &desc, size_t sizeInBytes, size_t offsetInBytes);
				MemoryDescWrapper(const MemoryDescWrapper &other) = delete;
				MemoryDescWrapper(MemoryDescWrapper &&other);
				MemoryDescWrapper& operator=(const MemoryDescWrapper &other) = delete;
				MemoryDescWrapper& operator=(MemoryDescWrapper &&other);
				~MemoryDescWrapper();
				Device device() const noexcept;
				operator backend::avMemoryDescriptor_t() const noexcept
				{
					return m_descriptor;
				}
		};

		class TensorDescWrapper
		{
				backend::avTensorDescriptor_t m_descriptor = backend::AVOCADO_NULL_DESCRIPTOR;
			public:
				TensorDescWrapper(Device device = Device::cpu());
				TensorDescWrapper(const TensorDescWrapper &other) = delete;
				TensorDescWrapper(TensorDescWrapper &&other);
				TensorDescWrapper& operator=(const TensorDescWrapper &other) = delete;
				TensorDescWrapper& operator=(TensorDescWrapper &&other);
				~TensorDescWrapper();
				Device device() const noexcept;
				void set(const Shape &shape, DataType dtype);
				operator backend::avTensorDescriptor_t() const noexcept
				{
					return m_descriptor;
				}
		};

		class ConvolutionDescWrapper
		{
				backend::avConvolutionDescriptor_t m_descriptor = backend::AVOCADO_NULL_DESCRIPTOR;
			public:
				ConvolutionDescWrapper(Device device = Device::cpu());
				ConvolutionDescWrapper(const ConvolutionDescWrapper &other) = delete;
				ConvolutionDescWrapper(ConvolutionDescWrapper &&other);
				ConvolutionDescWrapper& operator=(const ConvolutionDescWrapper &other) = delete;
				ConvolutionDescWrapper& operator=(ConvolutionDescWrapper &&other);
				~ConvolutionDescWrapper();
				Device device() const noexcept;
				void set(ConvMode mode, int nbDims, const std::array<int, 3> &padding, const std::array<int, 3> &strides,
						const std::array<int, 3> &dilation, int groups, const std::array<uint8_t, 16> &paddingValue);
				operator backend::avConvolutionDescriptor_t() const noexcept
				{
					return m_descriptor;
				}
		};

		class OptimizerDescWrapper
		{
				backend::avOptimizerDescriptor_t m_descriptor = backend::AVOCADO_NULL_DESCRIPTOR;
			public:
				OptimizerDescWrapper(Device device = Device::cpu());
				OptimizerDescWrapper(const OptimizerDescWrapper &other) = delete;
				OptimizerDescWrapper(OptimizerDescWrapper &&other);
				OptimizerDescWrapper& operator=(const OptimizerDescWrapper &other);
				OptimizerDescWrapper& operator=(OptimizerDescWrapper &&other);
				~OptimizerDescWrapper();
				Device device() const noexcept;
				void set(OptimizerType type, int64_t steps, double learningRate, const std::array<double, 4> &coefficients,
						const std::array<bool, 4> &flags);
				operator backend::avOptimizerDescriptor_t() const noexcept
				{
					return m_descriptor;
				}
		};

		void set_memory(backend::avContextDescriptor_t context, backend::avMemoryDescriptor_t mem, size_t dstOffset, size_t count,
				const void *pattern, size_t patternSizeInBytes);
		void copy_memory(backend::avContextDescriptor_t context, backend::avMemoryDescriptor_t dstMem, size_t dstOffset,
				const backend::avMemoryDescriptor_t srcMem, size_t srcOffset, size_t count);
		void change_type(backend::avContextDescriptor_t context, backend::avMemoryDescriptor_t dstMem, DataType dstType,
				const backend::avMemoryDescriptor_t srcMem, DataType srcType, size_t elements);

	} /* namespace internal */
} /* namespace avocado */

#endif /* AVOCADO_MATH_DESCRIPTOR_WRAPPERS_HPP_ */
