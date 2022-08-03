/*
 * Scalar.cpp
 *
 *  Created on: May 10, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/core/Scalar.hpp>

#include <cassert>

namespace
{
	using namespace avocado;

	template<typename T>
	std::string real_to_string(T value)
	{
		return std::to_string(value);
	}
	template<typename T>
	std::string complex_to_string(std::complex<T> value)
	{
		if (value.imag() >= 0)
			return std::to_string(value.real()) + " + " + std::to_string(value.imag()) + "i";
		else
			return std::to_string(value.real()) + " - " + std::to_string(-value.imag()) + "i";
	}
	std::string scalar_to_string(const Scalar &scalar)
	{
		switch (scalar.dtype())
		{
			case DataType::UINT8:
				return real_to_string(scalar.get<uint8_t>());
			case DataType::INT8:
				return real_to_string(scalar.get<int8_t>());
			case DataType::INT16:
				return real_to_string(scalar.get<int16_t>());
			case DataType::INT32:
				return real_to_string(scalar.get<int32_t>());
			case DataType::INT64:
				return real_to_string(scalar.get<int64_t>());
			case DataType::FLOAT16:
				return real_to_string(scalar.get<float>());
			case DataType::BFLOAT16:
				return real_to_string(scalar.get<float>());
			case DataType::FLOAT32:
				return real_to_string(scalar.get<float>());
			case DataType::FLOAT64:
				return real_to_string(scalar.get<double>());
			case DataType::COMPLEX32:
				return complex_to_string(scalar.get<std::complex<float>>());
			case DataType::COMPLEX64:
				return complex_to_string(scalar.get<std::complex<double>>());
			default:
				return "";
		}
	}

}

namespace avocado
{
	Scalar::Scalar(DataType dtype, const std::array<uint8_t, 16> &rawBytes) :
			m_data(rawBytes),
			m_dtype(dtype)
	{
	}
	Scalar::Scalar(DataType dtype) :
			m_dtype(dtype)
	{
		std::fill(m_data.begin(), m_data.end(), 0);
	}
	std::string Scalar::toString() const
	{
		return std::string("Scalar<") + m_dtype + ">{" + scalar_to_string(*this) + "}";
	}
	const void* Scalar::data() const noexcept
	{
		return static_cast<const void*>(m_data.data());
	}
	void* Scalar::data() noexcept
	{
		return static_cast<void*>(m_data.data());
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
		if (newType == m_dtype)
			return Scalar(*this);
		else
		{
			Scalar result(newType);
			math::changeType(result.data(), newType, this->data(), this->dtype(), 1);
			return result;
		}
	}
	void Scalar::toScalingTypeFor(DataType type)
	{
		if (type == m_dtype)
			return;
		switch (type)
		{
			case DataType::UINT8:
			case DataType::INT8:
			case DataType::INT16:
			case DataType::INT32:
			case DataType::INT64:
			case DataType::FLOAT16:
			case DataType::BFLOAT16:
			case DataType::FLOAT32:
				*this = asType(DataType::FLOAT32);
				break;
			case DataType::FLOAT64:
				*this = asType(DataType::FLOAT64);
				break;
			case DataType::COMPLEX32:
				*this = asType(DataType::COMPLEX32);
				break;
			case DataType::COMPLEX64:
				*this = asType(DataType::COMPLEX64);
				break;
			case DataType::UNKNOWN:
				break;
		}
	}

	Scalar Scalar::zero(DataType dtype)
	{
		switch (dtype)
		{
			case DataType::UINT8:
				return Scalar(static_cast<uint8_t>(0));
			case DataType::INT8:
				return Scalar(static_cast<int8_t>(0));
			case DataType::INT16:
				return Scalar(static_cast<int16_t>(0));
			case DataType::INT32:
				return Scalar(static_cast<int32_t>(0));
			case DataType::INT64:
				return Scalar(static_cast<int64_t>(0));
			case DataType::FLOAT16:
				return Scalar(DataType::FLOAT16, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
			case DataType::BFLOAT16:
				return Scalar(DataType::BFLOAT16, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
			case DataType::FLOAT32:
				return Scalar(0.0f);
			case DataType::FLOAT64:
				return Scalar(0.0);
			case DataType::COMPLEX32:
				return Scalar(std::complex<float>(0.0f, 0.0f));
			case DataType::COMPLEX64:
				return Scalar(std::complex<double>(0.0, 0.0));
			default:
				return Scalar();
		}
	}
	Scalar Scalar::one(DataType dtype)
	{
		switch (dtype)
		{
			case DataType::UINT8:
				return Scalar(static_cast<uint8_t>(1));
			case DataType::INT8:
				return Scalar(static_cast<int8_t>(1));
			case DataType::INT16:
				return Scalar(static_cast<int16_t>(1));
			case DataType::INT32:
				return Scalar(static_cast<int32_t>(1));
			case DataType::INT64:
				return Scalar(static_cast<int64_t>(1));
			case DataType::FLOAT16:
				return Scalar(DataType::FLOAT16, { 0, 0x3c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
			case DataType::BFLOAT16:
				return Scalar(DataType::BFLOAT16, { 0x80, 0x3f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
			case DataType::FLOAT32:
				return Scalar(1.0f);
			case DataType::FLOAT64:
				return Scalar(1.0);
			case DataType::COMPLEX32:
				return Scalar(std::complex<float>(1.0f, 0.0f));
			case DataType::COMPLEX64:
				return Scalar(std::complex<double>(1.0, 0.0));
			default:
				return Scalar();
		}
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

