/*
 * Device.cpp
 *
 *  Created on: May 12, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/core/Device.hpp>
#include <Avocado/core/DataType.hpp>
#include <Avocado/core/error_handling.hpp>
#include <Avocado/backend/backend_libraries.hpp>

#include <cstring>
#include <vector>
#include <iostream>

namespace
{
	using namespace avocado;
	using namespace avocado::backend;

	template<typename T>
	T get_device_property(avDeviceType_t type, avDeviceIndex_t index, avDeviceProperty_t prop, T defaultValue)
	{
		T result = defaultValue;
		switch (type)
		{
			case AVOCADO_DEVICE_CPU:
				cpuGetDeviceProperty(prop, &result);
				break;
			case AVOCADO_DEVICE_CUDA:
				cudaGetDeviceProperty(index, prop, &result);
				break;
			case AVOCADO_DEVICE_OPENCL:
				openclGetDeviceProperty(index, prop, &result);
				break;
		}
		return result;
	}
	CpuSimd get_cpu_simd_level()
	{
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_AVX512_VL_BW_DQ, false))
			return CpuSimd::AVX512VL_BW_DQ;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_AVX512_F, false))
			return CpuSimd::AVX512F;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_AVX2, false))
			return CpuSimd::AVX2;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_AVX, false))
			return CpuSimd::AVX;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_SSE42, false))
			return CpuSimd::SSE42;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_SSE41, false))
			return CpuSimd::SSE41;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_SSSE3, false))
			return CpuSimd::SSSE3;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_SSE3, false))
			return CpuSimd::SSE3;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_SSE2, false))
			return CpuSimd::SSE2;
		if (get_device_property(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_SSE, false))
			return CpuSimd::SSE;
		return CpuSimd::NONE;
	}

	struct CpuFeatures
	{
			char name[256];
			int64_t memory;
			int cores;
			CpuSimd simd_level;
			bool supports_fp16;
			CpuFeatures()
			{
				cpuGetDeviceProperty(AVOCADO_DEVICE_NAME, name);
				memory = get_device_property<int64_t>(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_MEMORY, 0);
				cores = get_device_property<int32_t>(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_PROCESSOR_COUNT, 0);
				simd_level = get_cpu_simd_level();
				supports_fp16 = get_device_property<bool>(AVOCADO_DEVICE_CPU, 0, AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, false);
			}
	};

	struct CudaFeatures
	{
			char name[256];
			int64_t global_memory;
			int sm_count;
			int major;
			int minor;
			bool supports_dp4a;
			bool supports_fp16;
			bool supports_bfloat16;
			bool supports_fp64;
			bool has_tensor_cores;
			CudaFeatures(int deviceIndex)
			{
				cudaGetDeviceProperty(deviceIndex, AVOCADO_DEVICE_NAME, name);
				global_memory = get_device_property<int64_t>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_MEMORY, 0);
				sm_count = get_device_property<int32_t>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_PROCESSOR_COUNT, 0);
				major = get_device_property<int32_t>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_ARCH_MAJOR, 0);
				minor = get_device_property<int32_t>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_ARCH_MINOR, 0);
				supports_dp4a = get_device_property<bool>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_SUPPORTS_DP4A, false);
				supports_fp16 = get_device_property<bool>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, false);
				supports_bfloat16 = get_device_property<bool>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_SUPPORTS_BFLOAT16, false);
				supports_fp64 = get_device_property<bool>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION, false);
				has_tensor_cores = get_device_property<bool>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_SUPPORTS_TENSOR_CORES, false);
			}
	};

	struct OpenCLFeatures
	{
			char name[256];
			int64_t global_memory;
			int compute_units;
			int cl_version_major;
			int cl_version_minor;
			bool supports_fp16;
			bool supports_fp32;
			bool supports_fp64;
			OpenCLFeatures(int deviceIndex)
			{
				cudaGetDeviceProperty(deviceIndex, AVOCADO_DEVICE_NAME, name);
				global_memory = get_device_property<int64_t>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_MEMORY, 0);
				compute_units = get_device_property<int32_t>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_PROCESSOR_COUNT, 0);
				cl_version_major = get_device_property<int32_t>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_ARCH_MAJOR, 0);
				cl_version_minor = get_device_property<int32_t>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_ARCH_MINOR, 0);
				supports_fp16 = get_device_property<bool>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, false);
				supports_fp32 = get_device_property<bool>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION, false);
				supports_fp64 = get_device_property<bool>(AVOCADO_DEVICE_CUDA, deviceIndex, AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION, false);
			}
	};

	const CpuFeatures& cpu_features()
	{
		static CpuFeatures result;
		return result;
	}
	const CudaFeatures& cuda_features(int index)
	{
		static std::vector<CudaFeatures> result = []()
		{
			std::vector<CudaFeatures> result;
			for (int i = 0; i < Device::numberOfCudaDevices(); i++)
				result.push_back(CudaFeatures(i));
			return result;
		}();
		return result[index];
	}
	const OpenCLFeatures& opencl_features(int index)
	{
		static std::vector<OpenCLFeatures> result = []()
		{
			std::vector<OpenCLFeatures> result;
			for (int i = 0; i < Device::numberOfOpenCLDevices(); i++)
				result.push_back(OpenCLFeatures(i));
			return result;
		}();
		return result[index];
	}

	const char* get_simd_name(CpuSimd s)
	{
		switch (s)
		{
			default:
			case CpuSimd::NONE:
				return "NONE";
			case CpuSimd::SSE:
				return "SSE";
			case CpuSimd::SSE2:
				return "SSE2";
			case CpuSimd::SSE3:
				return "SSE3";
			case CpuSimd::SSSE3:
				return "SSSE3";
			case CpuSimd::SSE41:
				return "SSE4.1";
			case CpuSimd::SSE42:
				return "SSE4.2";
			case CpuSimd::AVX:
				return "AVX";
			case CpuSimd::AVX2:
				return "AVX2";
			case CpuSimd::AVX512F:
				return "AVX512F";
			case CpuSimd::AVX512VL_BW_DQ:
				return "AVX512VL_BW_DQ";
		}
	}
}

namespace avocado
{

	Device::Device(DeviceType type, int index) :
			m_type(type),
			m_index(index)
	{
	}

//device creation
	Device Device::cpu() noexcept
	{
		return Device(DeviceType::CPU, 0);
	}
	Device Device::cuda(int index)
	{
		if (index < 0 || index >= Device::numberOfCudaDevices())
			throw IllegalDevice(METHOD_NAME, { DeviceType::CUDA, index });
		return Device(DeviceType::CUDA, index);
	}
	Device Device::opencl(int index)
	{
		if (index < 0 || index >= Device::numberOfOpenCLDevices())
			throw IllegalDevice(METHOD_NAME, { DeviceType::OPENCL, index });
		return Device(DeviceType::OPENCL, index);
	}
	Device Device::fromString(const std::string &str)
	{
		if (str == "CPU" || str == "cpu")
			return Device::cpu();
		if (str.substr(0, 5) == "CUDA:" || str.substr(0, 5) == "cuda:")
			return Device::cuda(std::atoi(str.data() + 5));
		if (str.substr(0, 7) == "OPENCL:" || str.substr(0, 7) == "opencl:")
			return Device::opencl(std::atoi(str.data() + 7));
		throw LogicError(METHOD_NAME, "Illegal device '" + str + "'");
	}

	int Device::index() const noexcept
	{
		return m_index;
	}
	DeviceType Device::type() const noexcept
	{
		return m_type;
	}
	bool Device::isCPU() const noexcept
	{
		return m_type == DeviceType::CPU;
	}
	bool Device::isCUDA() const noexcept
	{
		return m_type == DeviceType::CUDA;
	}
	bool Device::isOPENCL() const noexcept
	{
		return m_type == DeviceType::OPENCL;
	}

//flags
	bool Device::supportsType(DataType t) const noexcept
	{
		switch (t)
		{
			case DataType::UINT8:
			case DataType::INT8:
			case DataType::INT16:
			case DataType::INT32:
			case DataType::INT64:
				return true;
			case DataType::BFLOAT16:
			{
				if (this->isCUDA())
					return cuda_features(index()).supports_bfloat16;
				else
					return true;
			}
			case DataType::FLOAT16:
			{
				switch (this->type())
				{
					default:
					case DeviceType::CPU:
						return cpu_features().supports_fp16;
					case DeviceType::CUDA:
						return cuda_features(index()).supports_fp16;
					case DeviceType::OPENCL:
						return opencl_features(index()).supports_fp16;
				}
				break;
			}
			case DataType::FLOAT32:
			case DataType::COMPLEX32:
			{
				if (this->isOPENCL())
					return false;
				else
					return true;
			}
			case DataType::FLOAT64:
			case DataType::COMPLEX64:
			{
				switch (this->type())
				{
					default:
					case DeviceType::CPU:
						return true;
					case DeviceType::CUDA:
						return cuda_features(index()).supports_fp64;
					case DeviceType::OPENCL:
						return opencl_features(index()).supports_fp64;
				}
				break;
			}
			default:
				return false;
		}
	}
	bool Device::hasDP4A() const noexcept
	{
		switch (m_type)
		{
			default:
			case DeviceType::CPU:
				return false;
			case DeviceType::CUDA:
				return cuda_features(index()).supports_dp4a;
			case DeviceType::OPENCL:
				return false;
		}
	}
	bool Device::hasTensorCores() const noexcept
	{
		return false; //TODO later add detecting of tensor cores on GPUs
	}

//common features
	std::string Device::name() const
	{
		switch (m_type)
		{
			default:
			case DeviceType::CPU:
				return cpu_features().name;
			case DeviceType::CUDA:
				return cuda_features(index()).name;
			case DeviceType::OPENCL:
				return opencl_features(index()).name;
		}
	}
	int Device::memory() const noexcept
	{
		switch (m_type)
		{
			default:
			case DeviceType::CPU:
				return cpu_features().memory >> 20;
			case DeviceType::CUDA:
				return cuda_features(index()).global_memory >> 20;
			case DeviceType::OPENCL:
				return opencl_features(index()).global_memory >> 20;
		}
	}
	int Device::cores() const noexcept
	{
		switch (m_type)
		{
			default:
			case DeviceType::CPU:
				return cpu_features().cores;
			case DeviceType::CUDA:
				return cuda_features(index()).sm_count;
			case DeviceType::OPENCL:
				return opencl_features(index()).compute_units;
		}
	}

//CPU features
	void Device::setNumberOfThreads(int t) const noexcept
	{
		if (this->isCPU())
			backend::cpuSetNumberOfThreads(t);
	}
	int Device::getNumberOfThreads() const noexcept
	{
		if (this->isCPU())
			return backend::cpuGetNumberOfThreads();
		else
			return 0;
	}
	CpuSimd Device::simd() const noexcept
	{
		if (this->isCPU())
			return cpu_features().simd_level;
		else
			return CpuSimd::NONE;
	}

//CUDA features
	int Device::computeCapability() const noexcept
	{
		if (this->isCUDA())
			return cuda_features(index()).major * 10 + cuda_features(index()).minor;
		else
			return 0;
	}

	//OpenCL features
	int Device::openclVersion() const noexcept
	{
		if (this->isOPENCL())
			return opencl_features(index()).cl_version_major * 100 + opencl_features(index()).cl_version_minor * 10;
		else
			return 0;
	}

	std::string Device::toString() const
	{
		switch (m_type)
		{
			case DeviceType::CPU:
				return "CPU";
			case DeviceType::CUDA:
				return "CUDA:" + std::to_string(m_index);
			case DeviceType::OPENCL:
				return "OPENCL:" + std::to_string(m_index);
		}
		return "Illegal device";
	}
	std::string Device::info() const
	{
		std::string result = name() + " : " + std::to_string(cores()) + " x ";

		switch (m_type)
		{
			case DeviceType::CPU:
				result += get_simd_name(simd());
				break;
			case DeviceType::CUDA:
				result += "SM " + std::to_string(computeCapability() / 10) + "." + std::to_string(computeCapability() % 10);
				break;
			case DeviceType::OPENCL:
				result += "CL " + std::to_string(openclVersion() / 100) + "." + std::to_string((openclVersion() % 100) / 10);
				break;
		}

		return result + " with " + std::to_string(memory()) + "MB of memory";;
	}
	std::string Device::hardwareInfo()
	{
		std::string result = Device::cpu().toString() + " = " + Device::cpu().info() + '\n';
		for (int i = 0; i < Device::numberOfCudaDevices(); i++)
			result += Device::cuda(i).toString() + " = " + Device::cuda(i).info() + '\n';
		for (int i = 0; i < Device::numberOfOpenCLDevices(); i++)
			result += Device::opencl(i).toString() + " = " + Device::opencl(i).info() + '\n';
		return result;
	}

	int Device::numberOfCudaDevices() noexcept
	{
		static const int number_of_cuda_devices = backend::cudaGetNumberOfDevices();
		return number_of_cuda_devices;
	}
	int Device::numberOfOpenCLDevices() noexcept
	{
		static const int number_of_opencl_devices = openclGetNumberOfDevices();
		return number_of_opencl_devices;
	}
	bool Device::isCopyPossible(Device from, Device to) noexcept
	{
		if (from.isCPU() or to.isCPU())
			return true; // copying from/to CPU is always possible
		if (from.isCUDA() and to.isCUDA())
		{
			static const std::vector<bool> copying_table = []()
			{
				std::vector<bool> result;
				for (int i = 0; i < numberOfCudaDevices(); i++)
					for (int j = 0; j < numberOfCudaDevices(); j++)
					{
						bool tmp = false;
						backend::cudaIsCopyPossible(i, j, &tmp);
						result.push_back(tmp);
					}
				return result;
			}();
			return copying_table[from.index() * numberOfCudaDevices() + to.index()];
		}
		if (from.isOPENCL() and to.isOPENCL())
			return true;
		return false;
	}

	bool operator==(const Device &lhs, const Device &rhs)
	{
		return lhs.m_type == rhs.m_type && lhs.m_index == rhs.m_index;
	}
	bool operator!=(const Device &lhs, const Device &rhs)
	{
		return !(lhs == rhs);
	}

	std::ostream& operator<<(std::ostream &stream, const Device &d)
	{
		stream << d.toString();
		return stream;
	}
	std::string operator+(const std::string &str, const Device &d)
	{
		return str + d.toString();
	}
	std::string operator+(const Device &d, const std::string &str)
	{
		return d.toString() + str;
	}

} /* namespace avocado */
