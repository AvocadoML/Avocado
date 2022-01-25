/*
 * wrappers.hpp
 *
 *  Created on: Jan 21, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_BACKEND_TESTING_WRAPPERS_HPP_
#define AVOCADO_BACKEND_TESTING_WRAPPERS_HPP_

#include "../backend_defs.h"

#include <initializer_list>
#include <vector>
#include <stddef.h>
#include <memory>

namespace avocado
{
	namespace backend
	{
		class ContextWrapper
		{
			private:
				avContextDescriptor_t m_desc;
				avContextDescriptor_t m_ref_desc;
			public:
				ContextWrapper(avDeviceIndex_t device);
				ContextWrapper(const ContextWrapper &other) = delete;
				ContextWrapper(ContextWrapper &&other) noexcept;
				ContextWrapper& operator=(const ContextWrapper &other) = delete;
				ContextWrapper& operator=(ContextWrapper &&other) noexcept;
				~ContextWrapper();
				avTensorDescriptor_t getDescriptor() const noexcept
				{
					return m_desc;
				}
				avTensorDescriptor_t getRefDescriptor() const noexcept
				{
					return m_ref_desc;
				}
		};

		class TensorWrapper
		{
			private:
				avDeviceIndex_t m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				avTensorDescriptor_t m_tensor_descriptor = AVOCADO_INVALID_DESCRIPTOR;
				avMemoryDescriptor_t m_memory_descriptor = AVOCADO_INVALID_DESCRIPTOR;

				avTensorDescriptor_t m_ref_tensor_descriptor = AVOCADO_INVALID_DESCRIPTOR;
				avMemoryDescriptor_t m_ref_memory_descriptor = AVOCADO_INVALID_DESCRIPTOR;
				mutable int m_sync = 0;
			public:
				TensorWrapper() = default;
				TensorWrapper(std::vector<int> shape, avDataType_t dtype, avDeviceIndex_t device);
				TensorWrapper(const TensorWrapper &other) = delete;
				TensorWrapper(TensorWrapper &&TensorWrapper) noexcept;
				TensorWrapper& operator=(const TensorWrapper &other) = delete;
				TensorWrapper& operator=(TensorWrapper &&other) noexcept;
				~TensorWrapper() noexcept;

				avDataType_t dtype() const noexcept;
				size_t sizeInBytes() const noexcept;

				int numberOfDimensions() const noexcept;
				int dimension(int idx) const noexcept;
				int firstDim() const noexcept;
				int lastDim() const noexcept;
				int volume() const noexcept;

				void synchronize() const;
				void zeroall();
				template<typename T>
				void setall(T value)
				{
					set_pattern(&value, sizeof(T));
				}
				void copyToHost(void *dst) const;
				void copyFromHost(const void *src);

				template<typename T>
				T get(std::initializer_list<int> idx) const
				{
					T result;
					copy_data_to_cpu(&result, sizeof(T) * get_index(idx), sizeof(T));
					return result;
				}
				template<typename T>
				void set(T value, std::initializer_list<int> idx)
				{
					copy_data_from_cpu(sizeof(T) * get_index(idx), &value, sizeof(T));
				}

				avTensorDescriptor_t getDescriptor() const noexcept
				{
					return m_tensor_descriptor;
				}
				avMemoryDescriptor_t getMemory() const noexcept
				{
					m_sync = 1;
					return m_memory_descriptor;
				}

				avTensorDescriptor_t getRefDescriptor() const noexcept
				{
					return m_ref_tensor_descriptor;
				}
				avMemoryDescriptor_t getRefMemory() const noexcept
				{
					m_sync = -1;
					return m_ref_memory_descriptor;
				}
			private:
				size_t get_index(std::initializer_list<int> idx) const;
				void copy_data_to_cpu(void *dst, size_t src_offset, size_t count) const;
				void copy_data_from_cpu(size_t dst_offset, const void *src, size_t count);
				void set_pattern(const void *pattern, size_t patternSize);
		};


	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_TESTING_WRAPPERS_HPP_ */
