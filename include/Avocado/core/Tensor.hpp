/*
 * Tensor.hpp
 *
 *  Created on: Aug 17, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_CORE_TENSOR_HPP_
#define AVOCADO_CORE_TENSOR_HPP_

#include <Avocado/backend/backend_defs.h>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/Device.hpp>
#include <Avocado/core/DataType.hpp>
#include <Avocado/core/error_handling.hpp>
#include <Avocado/math/descriptor_wrappers.hpp>

#include <cstring>
#include <assert.h>
#include <variant>
#include <memory>

namespace avocado /* forward declarations */
{
	class Json;
	class SerializedObject;
	class Scalar;
	class Context;
}

namespace avocado
{
	class Tensor
	{
		private:
			Shape m_shape;
			int32_t m_stride[Shape::max_dimension];
			DataType m_dtype = DataType::UNKNOWN;
			Device m_device = Device::cpu();

			internal::TensorDescWrapper m_tensor_descriptor;
			internal::MemoryDescWrapper m_memory_descriptor;
			const Tensor *m_owning_tensor_pointer = nullptr;
			size_t m_memory_offset = 0;
			bool m_is_page_locked = false;
		public:
			Tensor() = default;
			Tensor(const Shape &shape, DataType dtype, Device device);
			Tensor(const Shape &shape, const std::string &dtype, Device device);
			Tensor(const Json &json, const SerializedObject &binarData);

			Tensor(const Tensor &other);
			Tensor(Tensor &&other) noexcept;

			~Tensor() noexcept;

			Tensor& operator=(const Tensor &other);
			Tensor& operator=(Tensor &&other) noexcept;

			std::string info(bool full = false) const;

			Device device() const noexcept;
			DataType dtype() const noexcept;
			size_t sizeInBytes() const noexcept;

			bool isOwning() const noexcept;
			bool isView() const noexcept;
			bool isEmpty() const noexcept;

			int numberOfDimensions() const noexcept;
			int dimension(int idx) const noexcept;
			int firstDim() const noexcept;
			int lastDim() const noexcept;
			int volume() const noexcept;
			const Shape& shape() const noexcept;

			void moveTo(Device newDevice);
			void reshape(const Shape &newShape);

			void convertTo(DataType newType);
			void zeroall();
			void setall(const Scalar &value);
			void copyTo(void *dst, size_t elements) const;
			void copyFrom(const void *src, size_t elements);
			void copyFrom(const Tensor &other);
			void copyFrom(const Tensor &other, size_t elements);

			bool isPageLocked() const;
			void pageLock();
			void pageUnlock();

			Tensor view() const;
			Tensor view(const Shape &shape, size_t offsetInElements = 0) const;

			void* data();
			const void* data() const;
			template<typename T>
			T get(std::initializer_list<int> idx) const
			{
				T result = 0;
				copy_data_to_cpu(&result, sizeOf(dtype()) * get_index(idx.begin(), idx.size()), sizeof(T));
				return result;
			}
			template<typename T>
			void set(T value, std::initializer_list<int> idx)
			{
				copy_data_from_cpu(sizeOf(dtype()) * get_index(idx.begin(), idx.size()), &value, sizeof(T));
			}

			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);

			backend::avTensorDescriptor_t getDescriptor() const noexcept;
			backend::avMemoryDescriptor_t getMemory() const;
		private:
			size_t get_index(const int *ptr, size_t size) const;
			void create_stride() noexcept;
			void copy_data_to_cpu(void *dst, size_t src_offset, size_t count) const;
			void copy_data_from_cpu(size_t dst_offset, const void *src, size_t count);
	};

	template<typename T>
	Tensor toTensor(std::initializer_list<T> data)
	{
		Tensor result( { static_cast<int>(data.size()) }, typeOf<T>(), Device::cpu());
		result.copyFrom(data.begin(), data.size());
		return result;
	}
	template<typename T>
	Tensor toTensor(std::initializer_list<std::initializer_list<T>> data)
	{
		Shape shape( { static_cast<int>(data.size()), static_cast<int>((data.begin()[0]).size()) });
		std::unique_ptr<T[]> tmp = std::make_unique<T[]>(shape.volume());
		for (int i = 0; i < shape[0]; i++)
		{
			assert(shape.lastDim() == static_cast<int>((data.begin()[i]).size()));
			std::memcpy(tmp.get() + i * shape.lastDim(), (data.begin()[i]).begin(), sizeof(T) * shape.lastDim());
		}

		Tensor result(shape, typeOf<T>(), Device::cpu());
		result.copyFrom(tmp.get(), result.volume());
		return result;
	}
	template<typename T>
	std::unique_ptr<T[]> toArray(const Tensor &t)
	{
		std::unique_ptr<T[]> result = std::make_unique<T[]>(t.volume());
		t.copyTo(result.get(), t.volume());
		return result;
	}
	template<typename T>
	void fromArray(Tensor &dst, const std::unique_ptr<T[]> &src)
	{
		dst.copyFrom(src.get(), dst.volume());
	}

	template<class T, class U>
	bool same_device(const T &lhs, const U &rhs)
	{
		return lhs.device() == rhs.device();
	}
	template<class T, class U, class ... ARGS>
	bool same_device(const T &lhs, const U &rhs, const ARGS &... args)
	{
		if (lhs.device() == rhs.device())
			return same_device(lhs, args...);
		else
			return false;
	}

	template<class T, class U>
	bool same_type(const T &lhs, const U &rhs)
	{
		return lhs.dtype() == rhs.dtype();
	}
	template<class T, class U, class ... ARGS>
	bool same_type(const T &lhs, const U &rhs, const ARGS &... args)
	{
		if (lhs.dtype() == rhs.dtype())
			return same_type(lhs, args...);
		else
			return false;
	}

	template<class T, class U>
	bool same_shape(const T &lhs, const U &rhs)
	{
		return lhs.shape() == rhs.shape();
	}
	template<class T, class U, class ... ARGS>
	bool same_shape(const T &lhs, const U &rhs, const ARGS &... args)
	{
		if (lhs.shape() == rhs.shape())
			return same_shape(lhs, args...);
		else
			return false;
	}
}
/* namespace avocado */

#endif /* AVOCADO_CORE_TENSOR_HPP_ */
