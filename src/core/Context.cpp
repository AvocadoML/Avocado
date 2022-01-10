/*
 * Context.cpp
 *
 *  Created on: Jun 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <avocado/core/Context.hpp>
#include <avocado/backend/backend_libraries.hpp>

#include <algorithm>

namespace avocado
{
	Context::Context(Device device) :
			m_device(device)
	{
		switch (device.type())
		{
			case DeviceType::CPU:
				backend::cpuCreateContextDescriptor(&m_data);
				break;
			case DeviceType::CUDA:
				backend::cudaCreateContextDescriptor(&m_data, device.index());
				break;
			case DeviceType::OPENCL:
				backend::openclCreateContextDescriptor(&m_data, device.index());
				break;
		}
	}
	Context::Context(Context &&other) :
			m_data(other.m_data),
			m_device(other.m_device)
	{
		other.m_data = backend::AVOCADO_INVALID_DESCRIPTOR;
	}
	Context& Context::operator=(Context &&other)
	{
		std::swap(this->m_data, other.m_data);
		std::swap(this->m_device, other.m_device);
		return *this;
	}
	Context::~Context()
	{
		switch (m_device.type())
		{
			case DeviceType::CPU:
				backend::cpuDestroyContextDescriptor(m_data);
				break;
			case DeviceType::CUDA:
				backend::cudaDestroyContextDescriptor(m_data);
				break;
			case DeviceType::OPENCL:
				backend::openclDestroyContextDescriptor(m_data);
				break;
		}
	}

	Device Context::device() const noexcept
	{
		return m_device;
	}
	backend::avContextDescriptor_t Context::getDescriptor() const noexcept
	{
		return m_data;
	}
	Context::operator backend::avContextDescriptor_t() const noexcept
	{
		return m_data;
	}

	backend::avContextDescriptor_t get_default_context(Device device) noexcept
	{
		switch (device.type())
		{
			case DeviceType::CPU:
				return backend::cpuGetDefaultContext();
			case DeviceType::CUDA:
				return backend::cudaGetDefaultContext(device.index());
			case DeviceType::OPENCL:
				return backend::openclGetDefaultContext(device.index());
			default:
				return backend::AVOCADO_INVALID_DESCRIPTOR;
		}
	}

} /* namespace avocado */

