/*
 * descriptor_wrappers.hpp
 *
 *  Created on: Jan 17, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_DESCRIPTOR_WRAPPERS_HPP_
#define AVOCADO_MATH_DESCRIPTOR_WRAPPERS_HPP_

#include <Avocado/backend/backend_defs.h>
#include <stddef.h>

namespace avocado
{
	class Device;
	class Shape;
	enum class DataType
	;
}

namespace avocado
{
	namespace internal
	{
		class MemoryDescWrapper
		{
				backend::avMemoryDescriptor_t m_descriptor = backend::AVOCADO_INVALID_DESCRIPTOR;
			public:
				MemoryDescWrapper() = default;
				MemoryDescWrapper(Device device, size_t sizeInBytes);
				MemoryDescWrapper(const MemoryDescWrapper &desc, size_t sizeInBytes, size_t offsetInBytes);
				MemoryDescWrapper(const MemoryDescWrapper &other) = delete;
				MemoryDescWrapper(MemoryDescWrapper &&other);
				MemoryDescWrapper& operator=(const MemoryDescWrapper &other) = delete;
				MemoryDescWrapper& operator=(MemoryDescWrapper &&other);
				~MemoryDescWrapper();
				operator backend::avMemoryDescriptor_t() const noexcept
				{
					return m_descriptor;
				}
		};

		class TensorDescWrapper
		{
				backend::avTensorDescriptor_t m_descriptor = backend::AVOCADO_INVALID_DESCRIPTOR;
			public:
				TensorDescWrapper() = default;
				TensorDescWrapper(Device device);
				TensorDescWrapper(const TensorDescWrapper &other) = delete;
				TensorDescWrapper(TensorDescWrapper &&other);
				TensorDescWrapper& operator=(const TensorDescWrapper &other) = delete;
				TensorDescWrapper& operator=(TensorDescWrapper &&other);
				~TensorDescWrapper();
				void set(const Shape &shape, DataType dtype);
				operator backend::avTensorDescriptor_t() const noexcept
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
