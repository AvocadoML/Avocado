/*
 * Context.cpp
 *
 *  Created on: Jun 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/core/Context.hpp>
#include <Avocado/backend/backend_libraries.hpp>
#include <Avocado/core/error_handling.hpp>

#include <algorithm>
#include <iostream>

namespace avocado
{
	Context::Context(Device device) :
			m_device(device)
	{
		switch (device.type())
		{
			case DeviceType::CPU:
			{
				backend::avStatus_t status = backend::cpuCreateContextDescriptor(&m_data);
				CHECK_CPU_STATUS(status)
				break;
			}
			case DeviceType::CUDA:
			{
				backend::avStatus_t status = backend::cudaCreateContextDescriptor(&m_data, device.index());
				CHECK_CUDA_STATUS(status)
				break;
			}
			case DeviceType::OPENCL:
			{
//				backend::avStatus_t status = backend::openclCreateContextDescriptor(&m_data, device.index());
//				CHECK_OPENCL_STATUS(status)
				break;
			}
		}
	}
	Context::Context(Context &&other) :
			m_data(other.m_data),
			m_device(other.m_device)
	{
		other.m_data = backend::AVOCADO_NULL_DESCRIPTOR;
	}
	Context& Context::operator=(Context &&other)
	{
		std::swap(this->m_data, other.m_data);
		std::swap(this->m_device, other.m_device);
		return *this;
	}
	Context::~Context()
	{
		backend::avStatus_t status;
		switch (m_device.type())
		{
			case DeviceType::CPU:
				status = backend::cpuDestroyContextDescriptor(m_data);
				break;
			case DeviceType::CUDA:
				status = backend::cudaDestroyContextDescriptor(m_data);
				break;
			case DeviceType::OPENCL:
//				status = backend::openclDestroyContextDescriptor(m_data);
				break;
		}
		if (status == backend::AVOCADO_STATUS_FREE_FAILED)
		{
			std::cout << "free failed\n";
			exit(-1);
		}
	}

	Device Context::device() const noexcept
	{
		return m_device;
	}
	void Context::synchronize() const
	{
		switch (m_device.type())
		{
			case DeviceType::CPU:
			{
				backend::avStatus_t status = backend::cpuSynchronizeWithContext(m_data);
				CHECK_CPU_STATUS(status)
				break;
			}
			case DeviceType::CUDA:
			{
				backend::avStatus_t status = backend::cudaSynchronizeWithContext(m_data);
				CHECK_CUDA_STATUS(status)
				break;
			}
			case DeviceType::OPENCL:
			{
//				backend::avStatus_t status = backend::openclSynchronizeWithContext(m_data);
//				CHECK_OPENCL_STATUS(status)
				break;
			}
		}
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
//			case DeviceType::OPENCL:
//				return backend::openclGetDefaultContext(device.index());
			default:
				return backend::AVOCADO_NULL_DESCRIPTOR;
		}
	}

} /* namespace avocado */

