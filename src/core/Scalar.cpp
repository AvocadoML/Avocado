/*
 * Scalar.cpp
 *
 *  Created on: May 10, 2020
 *      Author: Maciej Kozarzewski
 */

#include <avocado/core/Scalar.hpp>
#include <avocado/backend/backend_defs.h>

#include <cassert>

namespace
{
	using namespace avocado;

	template<typename T>
	std::string real_to_string(const uint8_t scalar[])
	{
		T tmp;
		std::memcpy(&tmp, scalar, sizeof(tmp));
		return std::to_string(tmp);
	}
	template<typename T>
	std::string complex_to_string(const uint8_t scalar[])
	{
		std::complex<T> tmp;
		std::memcpy(&tmp, scalar, sizeof(tmp));
		return std::to_string(tmp.real()) + " + " + std::to_string(tmp.imag()) + "i";
	}
	std::string scalar_to_string(const uint8_t scalar[], DataType dtype)
	{
		switch (dtype)
		{
			case DataType::UINT8:
				return real_to_string<uint8_t>(scalar);
			case DataType::INT8:
				return real_to_string<int8_t>(scalar);
			case DataType::INT16:
				return real_to_string<int16_t>(scalar);
			case DataType::INT32:
				return real_to_string<int32_t>(scalar);
			case DataType::INT64:
				return real_to_string<int64_t>(scalar);
			case DataType::FLOAT16:
				return ""; // TODO
//			{
//				float16 tmp;
//				std::memcpy(&tmp, scalar, sizeof(tmp));
//				return tmp.toString();
//			}
			case DataType::BFLOAT16:
				return ""; //TODO
			case DataType::FLOAT32:
				return real_to_string<float>(scalar);
			case DataType::FLOAT64:
				return real_to_string<double>(scalar);
			case DataType::COMPLEX32:
				return complex_to_string<float>(scalar);
			case DataType::COMPLEX64:
				return complex_to_string<double>(scalar);
			default:
				return "";
		}
	}

}

namespace avocado
{
	Scalar::Scalar(DataType dtype) :
			m_dtype(dtype)
	{
	}
	std::string Scalar::toString() const
	{
		return std::string("Scalar<") + m_dtype + ">{" + scalar_to_string(m_data, m_dtype) + "}";
	}
	const void* Scalar::data() const noexcept
	{
		return static_cast<const void*>(m_data);
	}
	void* Scalar::data() noexcept
	{
		return static_cast<void*>(m_data);
	}
	size_t Scalar::sizeInBytes() const noexcept
	{
		return sizeOf(m_dtype);
	}
	DataType Scalar::dtype() const noexcept
	{
		return m_dtype;
	}
	Scalar Scalar::asType(DataType newType) const
	{
		Scalar result(newType);
		math::changeType(result.data(), newType, this->data(), this->dtype(), 1);
		return result;
	}

	std::ostream& operator<<(std::ostream &stream, const Scalar &s)
	{
		stream << s.toString();
		return stream;
	}
	std::string operator+(const std::string &str, const Scalar &scalar)
	{
		return str + scalar.toString();
	}
	std::string operator+(const Scalar &scalar, const std::string &str)
	{
		return scalar.toString() + str;
	}

} /* namespace avocado */

