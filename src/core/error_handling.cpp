/*
 * error_handling.cpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/core/error_handling.hpp>
#include <Avocado/core/Device.hpp>
#include <Avocado/backend/backend_libraries.hpp>

namespace
{
	const char* get_status_name(int status)
	{
		switch (status)
		{
			case avocado::backend::AVOCADO_STATUS_SUCCESS:
				return "SUCCESS";
			case avocado::backend::AVOCADO_STATUS_ALLOC_FAILED:
				return "ALLOC_FAILED";
			case avocado::backend::AVOCADO_STATUS_FREE_FAILED:
				return "FREE_FAILED";
			case avocado::backend::AVOCADO_STATUS_BAD_PARAM:
				return "BAD_PARAM";
			case avocado::backend::AVOCADO_STATUS_ARCH_MISMATCH:
				return "ARCH_MISMATCH";
			case avocado::backend::AVOCADO_STATUS_INTERNAL_ERROR:
				return "INTERNAL_ERROR";
			case avocado::backend::AVOCADO_STATUS_NOT_SUPPORTED:
				return "NOT_SUPPORTED";
			case avocado::backend::AVOCADO_STATUS_UNSUPPORTED_DATATYPE:
				return "UNSUPPORTED_DATATYPE";
			case avocado::backend::AVOCADO_STATUS_EXECUTION_FAILED:
				return "EXECUTION_FAILED";
			case avocado::backend::AVOCADO_STATUS_INSUFFICIENT_DRIVER:
				return "INSUFFICIENT_DRIVER";
			case avocado::backend::AVOCADO_STATUS_DEVICE_TYPE_MISMATCH:
				return "DEVICE_MISMATCH";
			default:
				return "UNKNOWN";
		}
	}
}

namespace avocado
{

//runtime errors
	RuntimeError::RuntimeError(const char *function) :
			runtime_error(function)
	{
	}
	RuntimeError::RuntimeError(const char *function, const std::string &comment) :
			runtime_error(std::string(function) + comment)
	{
	}

	//range errors
	IndexOutOfBounds::IndexOutOfBounds(const char *function, const std::string &index_name, int index_value, int range) :
			out_of_range(
					std::string(function) + " : '" + index_name + "' = " + std::to_string(index_value) + " out of range [0, " + std::to_string(range)
							+ ")")
	{
	}
	OutOfRange::OutOfRange(const char *function, int value, int range) :
			out_of_range(std::string(function) + " : " + std::to_string(value) + " out of range [0, " + std::to_string(range) + ")")
	{
	}

	//not-supported errors
	NotImplemented::NotImplemented(const char *function) :
			logic_error(function)
	{
	}
	NotImplemented::NotImplemented(const char *function, const std::string &comment) :
			logic_error(std::string(function) + " : " + comment)
	{
	}

	LogicError::LogicError(const char *function) :
			std::logic_error(function)
	{
	}
	LogicError::LogicError(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}

	UninitializedObject::UninitializedObject(const char *function) :
			std::logic_error(function)
	{
	}
	UninitializedObject::UninitializedObject(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}

	//illegal argument
	IllegalArgument::IllegalArgument(const char *function, const std::string &comment) :
			invalid_argument(std::string(function) + " : " + comment)
	{
	}
	IllegalArgument::IllegalArgument(const char *function, const char *arg_name, const std::string &comment, int arg_value) :
			IllegalArgument(function, arg_name, comment, std::to_string(arg_value))
	{
	}
	IllegalArgument::IllegalArgument(const char *function, const char *arg_name, const std::string &comment, const std::string &arg_value) :
			invalid_argument(std::string(function) + " : '" + arg_name + "' " + comment + ", got " + arg_value)
	{
	}

	DeviceMismatch::DeviceMismatch(const char *function) :
			std::logic_error(function)
	{
	}
	DeviceMismatch::DeviceMismatch(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}
	DeviceMismatch::DeviceMismatch(const char *function, Device expected, Device got) :
			logic_error(std::string(function) + " : expected device " + expected + ", got " + got)
	{
	}
	IllegalDevice::IllegalDevice(const char *function, Device d) :
			std::invalid_argument(std::string(function) + " : " + d.toString())
	{
	}
	CpuRuntimeError::CpuRuntimeError(const char *function, int error) :
			std::runtime_error(std::string(function) + " : " + get_status_name(error))
	{
	}
	CpuRuntimeError::CpuRuntimeError(const char *function, const std::string &comment) :
			std::runtime_error(std::string(function) + " : " + comment)
	{
	}
	CudaRuntimeError::CudaRuntimeError(const char *function, int error) :
			std::runtime_error(std::string(function) + " : " + get_status_name(error))
	{
	}
	CudaRuntimeError::CudaRuntimeError(const char *function, const std::string &comment) :
			std::runtime_error(std::string(function) + " : " + comment)
	{
	}
	OpenCLRuntimeError::OpenCLRuntimeError(const char *function, int error) :
			std::runtime_error(std::string(function) + " : " + get_status_name(error))
	{
	}
	OpenCLRuntimeError::OpenCLRuntimeError(const char *function, const std::string &comment) :
			std::runtime_error(std::string(function) + " : " + comment)
	{
	}

} /* namespace avocado */

