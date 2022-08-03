/*
 * Tensor.cpp
 *
 *  Created on: Aug 18, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>
#include <Avocado/backend/backend_libraries.hpp>

#include <cassert>
#include <memory>
#include <algorithm>

namespace avocado
{
	Tensor::Tensor() :
			Tensor(Shape(), DataType::UNKNOWN, Device::cpu())
	{

	}
	Tensor::Tensor(const Shape &shape, DataType dtype, Device device) :
			m_shape(shape),
			m_dtype(dtype),
			m_device(device),
			m_tensor_descriptor(device),
			m_memory_descriptor(device, sizeInBytes())
	{
		create_stride();
		m_tensor_descriptor.set(shape, dtype);
		internal::set_memory(get_default_context(device), m_memory_descriptor, 0, sizeInBytes(), nullptr, 0);
	}
	Tensor::Tensor(const Shape &shape, const std::string &dtype, Device device) :
			Tensor(shape, typeFromString(dtype), device)
	{
	}
	Tensor::Tensor(const Json &json, const SerializedObject &binary_data) :
			Tensor(json["shape"], typeFromString(json["dtype"]), Device::cpu())
	{
		if (not isEmpty())
			binary_data.load(this->data(), static_cast<size_t>(json["binary_offset"]), this->sizeInBytes());
	}

	Tensor::Tensor(const Tensor &other) :
			m_shape(other.m_shape),
			m_dtype(other.m_dtype),
			m_device(other.m_device),
			m_tensor_descriptor(other.m_device),
			m_owning_tensor_pointer(other.m_owning_tensor_pointer),
			m_memory_offset(other.m_memory_offset)
	{
		create_stride();
		m_tensor_descriptor.set(m_shape, m_dtype);
		if (other.isOwning())
		{
			m_memory_descriptor = internal::MemoryDescWrapper(m_device, sizeInBytes());
			internal::copy_memory(get_default_context(m_device), m_memory_descriptor, 0, other.m_memory_descriptor, 0, sizeInBytes());
		}
		else
			m_memory_descriptor = internal::MemoryDescWrapper(other.m_memory_descriptor, sizeInBytes(), 0);
	}
	Tensor::Tensor(Tensor &&other) noexcept :
			m_shape(other.m_shape),
			m_dtype(other.m_dtype),
			m_device(other.m_device),
			m_tensor_descriptor(std::move(other.m_tensor_descriptor)),
			m_memory_descriptor(std::move(other.m_memory_descriptor)),
			m_owning_tensor_pointer(other.m_owning_tensor_pointer),
			m_memory_offset(other.m_memory_offset)
	{
		create_stride();
		other.m_owning_tensor_pointer = nullptr;
	}
	Tensor::~Tensor() noexcept
	{
		if (isPageLocked()) // TODO not sure if this is necessary
			pageUnlock();
	}
	Tensor& Tensor::operator=(const Tensor &other)
	{
		if (this != &other)
		{
			if (other.isOwning()) // make a full copy
			{
				if (this->isOwning())
				{
					if (this->sizeInBytes() != other.sizeInBytes() or this->device() != other.device()) // reallocate if different size or device
						m_memory_descriptor = internal::MemoryDescWrapper(m_device, other.sizeInBytes());
				}
				else
					m_memory_descriptor = internal::MemoryDescWrapper(m_device, other.sizeInBytes());
				internal::copy_memory(get_default_context(m_device), m_memory_descriptor, 0, other.m_memory_descriptor, 0, other.sizeInBytes());
			}
			else
				m_memory_descriptor = internal::MemoryDescWrapper(other.m_memory_descriptor, sizeInBytes(), 0);
			this->m_shape = other.m_shape;
			create_stride();
			this->m_dtype = other.m_dtype;
			this->m_device = other.m_device;
			this->m_tensor_descriptor.set(this->m_shape, this->m_dtype);
			this->m_owning_tensor_pointer = other.m_owning_tensor_pointer;
			this->m_memory_offset = other.m_memory_offset;
		}
		return *this;
	}
	Tensor& Tensor::operator=(Tensor &&other) noexcept
	{
		if (this != &other)
		{
			std::swap(this->m_shape, other.m_shape);
			std::swap(this->m_stride, other.m_stride);
			std::swap(this->m_dtype, other.m_dtype);
			std::swap(this->m_device, other.m_device);
			std::swap(this->m_tensor_descriptor, other.m_tensor_descriptor);
			std::swap(this->m_memory_descriptor, other.m_memory_descriptor);
			std::swap(this->m_owning_tensor_pointer, other.m_owning_tensor_pointer);
			std::swap(this->m_memory_offset, other.m_memory_offset);
		}
		return *this;
	}

	std::string Tensor::info(bool full) const
	{
		if (full)
		{
			std::string result;
			result += std::string("device     : ") + device() + '\n';
			result += std::string("data type  : ") + dtype() + '\n';
			result += std::string("size       : ") + std::to_string(sizeInBytes()) + '\n';
			result += std::string("volume     : ") + std::to_string(volume()) + '\n';
			result += std::string("is owning  : ") + std::to_string(isOwning()) + '\n';
			result += std::string("is view    : ") + std::to_string(isView()) + '\n';
			result += std::string("dim        : ") + std::to_string(numberOfDimensions()) + '\n';
			result += std::string("is locked  : ") + std::to_string(isPageLocked()) + '\n';
//			result += "data       : " + std::to_string(data()) + '\n';
			return result;
		}
		else
		{
			if (isOwning())
				return std::string("Tensor<") + dtype() + ">" + m_shape.toString() + " on " + device();
			else
				return std::string("TensorView<") + dtype() + ">" + m_shape.toString() + " on " + device();
		}
	}

	Device Tensor::device() const noexcept
	{
		return m_device;
	}
	DataType Tensor::dtype() const noexcept
	{
		return m_dtype;
	}
	size_t Tensor::sizeInBytes() const noexcept
	{
		return sizeOf(dtype()) * volume();
	}

	bool Tensor::isOwning() const noexcept
	{
		return m_owning_tensor_pointer == nullptr;
	}
	bool Tensor::isView() const noexcept
	{
		return not isOwning();
	}
	bool Tensor::isEmpty() const noexcept
	{
		return numberOfDimensions() == 0;
	}

	int Tensor::numberOfDimensions() const noexcept
	{
		return m_shape.length();
	}
	int Tensor::firstDim() const noexcept
	{
		return m_shape.firstDim();
	}
	int Tensor::lastDim() const noexcept
	{
		return m_shape.lastDim();
	}
	int Tensor::dimension(int idx) const noexcept
	{
		return m_shape[idx];
	}
	const Shape& Tensor::shape() const noexcept
	{
		return m_shape;
	}
	int Tensor::volume() const noexcept
	{
		return m_shape.volume();
	}

	void Tensor::moveTo(Device newDevice)
	{
		if (isView())
			throw LogicError(METHOD_NAME, "Tensor view cannot be moved to another device");
		if (device() == newDevice)
			return;

		internal::MemoryDescWrapper newMemDesc(newDevice, sizeInBytes());
		internal::copy_memory(get_default_context(m_device), newMemDesc, 0, m_memory_descriptor, 0, sizeInBytes());
		std::swap(m_memory_descriptor, newMemDesc);

		m_tensor_descriptor = internal::TensorDescWrapper(newDevice);
		m_tensor_descriptor.set(m_shape, m_dtype);
	}
	void Tensor::reshape(const Shape &newShape)
	{
		if (this->m_shape.volume() != newShape.volume())
			throw ShapeMismatch(METHOD_NAME, "");

		this->m_shape = newShape;
		create_stride();
	}

	void Tensor::convertTo(DataType newType)
	{
		if (isView())
			throw LogicError(METHOD_NAME, "Tensor view cannot be converted to another type");
		if (dtype() == newType)
			return;

		if (sizeOf(m_dtype) != sizeOf(newType)) // no reallocation needed
			internal::change_type(get_default_context(m_device), m_memory_descriptor, newType, m_memory_descriptor, m_dtype, volume());
		else
		{
			internal::MemoryDescWrapper newMemDesc(m_device, sizeInBytes());
			internal::change_type(get_default_context(m_device), newMemDesc, newType, m_memory_descriptor, m_dtype, volume());
			std::swap(m_memory_descriptor, newMemDesc);
		}
		m_dtype = newType;
		m_tensor_descriptor.set(m_shape, m_dtype);
	}
	void Tensor::zeroall()
	{
		internal::set_memory(get_default_context(m_device), m_memory_descriptor, 0, sizeInBytes(), nullptr, 0);
	}
	void Tensor::setall(const Scalar &value)
	{
		if (value.dtype() != this->dtype())
			throw DataTypeMismatch(METHOD_NAME, this->dtype(), value.dtype());
		internal::set_memory(get_default_context(m_device), m_memory_descriptor, 0, sizeInBytes(), value.data(), value.sizeInBytes());
	}
	void Tensor::copyToHost(void *dst, size_t elements) const
	{
		copy_data_to_cpu(dst, 0, sizeOf(dtype()) * elements);
	}
	void Tensor::copyFromHost(const void *src, size_t elements)
	{
		copy_data_from_cpu(0, src, sizeOf(dtype()) * elements);
	}
	void Tensor::copyFrom(const Tensor &other)
	{
		this->copyFrom(other, this->volume());
	}
	void Tensor::copyFrom(const Tensor &other, size_t elements)
	{
		if (elements > static_cast<size_t>(std::min(this->volume(), other.volume())))
			throw IllegalArgument(METHOD_NAME, "elements", "must be lower than tensor size", elements);
		if (elements == 0)
			return; // no elements copied
		if (this->m_shape != other.m_shape)
			throw ShapeMismatch(METHOD_NAME, this->m_shape, other.m_shape);
		if (this->dtype() != other.dtype())
			throw DataTypeMismatch(METHOD_NAME, this->dtype(), other.dtype());

		internal::copy_memory(get_default_context(m_device), m_memory_descriptor, 0, other.m_memory_descriptor, 0, sizeOf(m_dtype) * elements);
	}

	bool Tensor::isPageLocked() const
	{
		return m_is_page_locked;
	}
	void Tensor::pageLock()
	{
		if (isView())
			throw LogicError(METHOD_NAME, "tensor view cannot be page locked");
		if (isPageLocked())
			throw LogicError(METHOD_NAME, "tensor already is page locked");
		if (device().isCPU())
		{
			backend::avStatus_t status = backend::cudaPageLock(data(), sizeInBytes());
			CHECK_CUDA_STATUS(status);
			m_is_page_locked = true;
		}
	}
	void Tensor::pageUnlock()
	{
		if (isView())
			throw LogicError(METHOD_NAME, "tensor view cannot be page unlocked");
		if (!isPageLocked())
			throw LogicError(METHOD_NAME, "tensor is not page locked");
		if (device().isCPU())
		{
			backend::avStatus_t status = backend::cudaPageUnlock(data());
			CHECK_CUDA_STATUS(status);
			m_is_page_locked = false;
		}
	}

	Tensor Tensor::view()
	{
		return view(shape(), 0);
	}
	Tensor Tensor::view(const Shape &shape, size_t offsetInElements)
	{
		if (this->isView())
			offsetInElements += m_memory_offset;
		if (offsetInElements + shape.volume() > static_cast<size_t>(this->volume()))
			throw ShapeMismatch(METHOD_NAME, "view would extend beyond the original tensor");

		Tensor result;
		result.m_shape = shape;
		result.create_stride();
		result.m_dtype = this->m_dtype;
		result.m_device = this->m_device;

		result.m_tensor_descriptor = internal::TensorDescWrapper(result.m_device);
		result.m_tensor_descriptor.set(result.m_shape, result.m_dtype);
		if (this->isOwning())
			result.m_owning_tensor_pointer = this;
		else
			result.m_owning_tensor_pointer = this->m_owning_tensor_pointer;
		result.m_memory_descriptor = internal::MemoryDescWrapper(result.m_owning_tensor_pointer->m_memory_descriptor, result.sizeInBytes(),
				offsetInElements);
		result.m_memory_offset = offsetInElements;
		result.m_is_page_locked = this->m_is_page_locked;
		return result;
	}

	backend::avTensorDescriptor_t Tensor::getDescriptor() const noexcept
	{
		return m_tensor_descriptor;
	}
	backend::avMemoryDescriptor_t Tensor::getMemory() const
	{
		return m_memory_descriptor;
	}

	void* Tensor::data()
	{
		if (m_device.isCPU())
			return backend::cpuGetMemoryPointer(m_memory_descriptor);
		else
			throw LogicError(METHOD_NAME, "tensor is not on CPU");
	}
	const void* Tensor::data() const
	{
		if (m_device.isCPU())
			return backend::cpuGetMemoryPointer(m_memory_descriptor);
		else
			throw LogicError(METHOD_NAME, "tensor is not on CPU");
	}

	Json Tensor::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["shape"] = m_shape.toJson();
		result["dtype"] = toString(dtype());
		result["binary_offset"] = binary_data.size();

		if (!isEmpty())
		{
			if (device().isCPU())
				binary_data.save(data(), sizeInBytes());
			else
			{
				std::unique_ptr<int8_t[]> buffer_on_cpu = std::make_unique<int8_t[]>(sizeInBytes());
				copyToHost(buffer_on_cpu.get(), sizeInBytes());
				binary_data.save(buffer_on_cpu.get(), sizeInBytes());
			}
		}

		return result;
	}
	void Tensor::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		if (shape() != Shape(json["shape"]))
			throw ShapeMismatch(METHOD_NAME, shape(), Shape(json["shape"]));
		if (dtype() != typeFromString(json["dtype"]))
			throw DataTypeMismatch(METHOD_NAME, dtype(), typeFromString(json["dtype"]));

		if (!isEmpty())
		{
			if (device().isCPU())
				binary_data.load(data(), static_cast<size_t>(json["binary_offset"]), sizeInBytes());
			else
			{
				std::unique_ptr<int8_t[]> buffer_on_cpu = std::make_unique<int8_t[]>(sizeInBytes());
				binary_data.load(buffer_on_cpu.get(), static_cast<size_t>(json["binary_offset"]), sizeInBytes());
				copyFromHost(buffer_on_cpu.get(), sizeInBytes());
			}
		}
	}

	size_t Tensor::get_index(const int *ptr, size_t size) const
	{
		if (static_cast<int>(size) != numberOfDimensions())
			throw ShapeMismatch(METHOD_NAME, numberOfDimensions(), static_cast<int>(size));

		assert(ptr != nullptr);
		size_t result = 0;
		for (int i = 0; i < numberOfDimensions(); i++)
		{
#ifndef NDEBUG
			if (ptr[i] < 0 || ptr[i] > m_shape[i])
				throw IndexOutOfBounds(METHOD_NAME, std::string("index:") + std::to_string(i), ptr[i], m_shape[i]);
#endif
			result += m_stride[i] * static_cast<uint32_t>(ptr[i]);
		}
		return result;
	}
	void Tensor::create_stride() noexcept
	{
		uint32_t tmp = 1;
		for (int i = Shape::max_dimension - 1; i >= m_shape.length(); i--)
			m_stride[i] = 0;
		for (int i = m_shape.length() - 1; i >= 0; i--)
		{
			m_stride[i] = tmp;
			tmp *= static_cast<uint32_t>(m_shape[i]);
		}
	}
	void Tensor::copy_data_to_cpu(void *dst, size_t src_offset, size_t count) const
	{
		switch (m_device.type())
		{
			case DeviceType::CPU: // from CPU to CPU
				std::memcpy(dst, reinterpret_cast<uint8_t*>(backend::cpuGetMemoryPointer(m_memory_descriptor)) + src_offset, count);
				break;
			case DeviceType::CUDA: // from CPU to CUDA
			{
				backend::avStatus_t status = backend::cudaCopyMemoryToHost(get_default_context(m_device), dst, m_memory_descriptor, src_offset,
						count);
				CHECK_CUDA_STATUS(status);
				break;
			}
			case DeviceType::OPENCL: // from CPU to OPENCL
			{
//				backend::avStatus_t status = backend::openclCopyMemoryToHost(get_default_context(m_device), dst, m_memory_descriptor, src_offset,
//						count);
//				CHECK_OPENCL_STATUS(status);
				break;
			}
		}
	}
	void Tensor::copy_data_from_cpu(size_t dst_offset, const void *src, size_t count)
	{
		switch (m_device.type())
		{
			case DeviceType::CPU: // from CPU to CPU
				std::memcpy(reinterpret_cast<uint8_t*>(backend::cpuGetMemoryPointer(m_memory_descriptor)) + dst_offset, src, count);
				break;
			case DeviceType::CUDA: // from CPU to CUDA
			{
				backend::avStatus_t status = backend::cudaCopyMemoryFromHost(get_default_context(m_device), m_memory_descriptor, dst_offset, src,
						count);
				CHECK_CUDA_STATUS(status);
				break;
			}
			case DeviceType::OPENCL: // from CPU to OPENCL
			{
//				backend::avStatus_t status = backend::openclCopyMemoryFromHost(get_default_context(m_device), m_memory_descriptor, dst_offset, src,
//						count);
//				CHECK_OPENCL_STATUS(status);
				break;
			}
		}
	}

} /* namespace avocado */

