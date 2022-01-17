/*
 * error_handling.hpp
 *
 *  Created on: May 7, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_CORE_ERROR_HANDLING_HPP_
#define AVOCADO_CORE_ERROR_HANDLING_HPP_

#include <stdexcept>
#include <string>

namespace avocado
{
	class Device;
}

namespace avocado
{
#ifdef __GNUC__
#  define METHOD_NAME __PRETTY_FUNCTION__
#else
#  define METHOD_NAME __FUNCTION__
#endif

	//runtime errors
	class RuntimeError: public std::runtime_error
	{
		public:
			RuntimeError(const char *function);
			RuntimeError(const char *function, const std::string &comment);
	};

	//range errors
	class IndexOutOfBounds: public std::out_of_range
	{
		public:
			IndexOutOfBounds(const char *function, const std::string &index_name, int index_value, int range);
	};
	class OutOfRange: public std::out_of_range
	{
		public:
			OutOfRange(const char *function, int value, int range);
	};

	//not-supported errors
	class NotImplemented: public std::logic_error
	{
		public:
			NotImplemented(const char *function);
			NotImplemented(const char *function, const std::string &comment);
	};

	class LogicError: public std::logic_error
	{
		public:
			LogicError(const char *function);
			LogicError(const char *function, const std::string &comment);
	};

	class UninitializedObject: public std::logic_error
	{
		public:
			UninitializedObject(const char *function);
			UninitializedObject(const char *function, const std::string &comment);
	};

	//illegal argument
	class IllegalArgument: public std::invalid_argument
	{
		public:
			IllegalArgument(const char *function, const std::string &comment);
			IllegalArgument(const char *function, const char *arg_name, const std::string &comment, int arg_value);
			IllegalArgument(const char *function, const char *arg_name, const std::string &comment, const std::string &arg_value);
	};

	class DeviceMismatch: public std::logic_error
	{
		public:
			DeviceMismatch(const char *function);
			DeviceMismatch(const char *function, const std::string &comment);
			DeviceMismatch(const char *function, Device expected, Device got);
	};
	class IllegalDevice: public std::invalid_argument
	{
		public:
			IllegalDevice(const char *function, Device d);
	};
	class CudaRuntimeError: public std::runtime_error
	{
		public:
			CudaRuntimeError(const char *function, int error);
			CudaRuntimeError(const char *function, const std::string &comment);
	};
	class OpenCLRuntimeError: public std::runtime_error
	{
		public:
			OpenCLRuntimeError(const char *function, int error);
			OpenCLRuntimeError(const char *function, const std::string &comment);
	};
	class InsufficientComputeCapability: public std::logic_error
	{
	};

#define CHECK_CPU_STATUS(status) if(status != backend::AVOCADO_STATUS_SUCCESS) throw RuntimeError(METHOD_NAME, "");
#define CHECK_CUDA_STATUS(status) if(status != backend::AVOCADO_STATUS_SUCCESS) throw CudaRuntimeError(METHOD_NAME, status);
#define CHECK_OPENCL_STATUS(status) if(status != backend::AVOCADO_STATUS_SUCCESS) throw OpenCLRuntimeError(METHOD_NAME, status);
} /* namespace avocado */

#endif /* AVOCADO_CORE_ERROR_HANDLING_HPP_ */
