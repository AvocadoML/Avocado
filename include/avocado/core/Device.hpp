/*
 * Device.hpp
 *
 *  Created on: May 12, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef CORE_DEVICE_HPP_
#define CORE_DEVICE_HPP_

#include <string>
#include <stdexcept>

namespace avocado /* forward declarations */
{
	enum class DataType;
}

namespace avocado
{
	enum class DeviceType
	{
		CPU,
		CUDA,
		OPENCL
	};

	enum class CpuSimd
	{
		NONE,
		SSE,
		SSE2,
		SSE3,
		SSSE3,
		SSE41,
		SSE42,
		AVX,
		AVX2,
		AVX512F,
		AVX512VL_BW_DQ
	};

	class Device
	{
		private:
			DeviceType m_type;
			int m_index;
			Device(DeviceType type, int index);
		public:
			// device creation
			static Device cpu() noexcept;
			static Device cuda(int index);
			static Device opencl(int index);
			static Device fromString(const std::string &str);

			DeviceType type() const noexcept;
			int index() const noexcept;
			bool isCPU() const noexcept;
			bool isCUDA() const noexcept;
			bool isOPENCL() const noexcept;

			//flags
			bool supportsType(DataType t) const noexcept;
			bool hasDP4A() const noexcept;
			bool hasTensorCores() const noexcept;

			//common features
			std::string name() const;
			int memory() const noexcept;
			int cores() const noexcept;

			//CPU features
			void setNumberOfThreads(int t) const noexcept;
			int getNumberOfThreads() const noexcept;
			CpuSimd simd() const noexcept;

			//CUDA features
			int computeCapability() const noexcept;

			//OpenCL features
			int openclVersion() const noexcept;

			std::string toString() const;
			std::string info() const;

			static int numberOfCudaDevices() noexcept;
			static int numberOfOpenCLDevices() noexcept;
			static bool isCopyPossible(Device from, Device to) noexcept;
			static std::string hardwareInfo();

			friend bool operator==(const Device &lhs, const Device &rhs);
			friend bool operator!=(const Device &lhs, const Device &rhs);
	};

	std::ostream& operator<<(std::ostream &stream, const Device &d);
	std::string operator+(const std::string &str, const Device &d);
	std::string operator+(const Device &d, const std::string &str);

} /* namespace avocado */

#endif /* CORE_DEVICE_HPP_ */
