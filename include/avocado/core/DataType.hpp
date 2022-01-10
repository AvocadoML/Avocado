/*
 * DataType.hpp
 *
 *  Created on: May 10, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_CORE_DATATYPE_HPP_
#define AVOCADO_CORE_DATATYPE_HPP_

#include <string>
#include <type_traits>
#include <stdexcept>
#include <complex>

namespace avocado /* forward declarations */
{
	class float16;
	class bfloat16;
}

namespace avocado
{
	enum class DataType
	{
		UNKNOWN,
		UINT8,
		INT8,
		INT16,
		INT32,
		INT64,
		FLOAT16,
		BFLOAT16,
		FLOAT32,
		FLOAT64,
		COMPLEX32,
		COMPLEX64
	};

	template<typename T>
	DataType typeOf() noexcept
	{
		DataType result = DataType::UNKNOWN;
		if (std::is_same<T, uint8_t>::value)
			result = DataType::UINT8;
		if (std::is_same<T, int8_t>::value)
			result = DataType::INT8;
		if (std::is_same<T, int16_t>::value)
			result = DataType::INT16;
		if (std::is_same<T, int32_t>::value)
			result = DataType::INT32;
		if (std::is_same<T, int64_t>::value)
			result = DataType::INT64;
		if (std::is_same<T, float16>::value)
			result = DataType::FLOAT16;
		if (std::is_same<T, bfloat16>::value)
			result = DataType::BFLOAT16;
		if (std::is_same<T, float>::value)
			result = DataType::FLOAT32;
		if (std::is_same<T, double>::value)
			result = DataType::FLOAT64;
		if (std::is_same<T, std::complex<float>>::value)
			result = DataType::COMPLEX32;
		if (std::is_same<T, std::complex<double>>::value)
			result = DataType::COMPLEX64;
		return result;
	}
	size_t sizeOf(DataType t) noexcept;
	bool isInteger(DataType t) noexcept;
	bool isFloatingPoint(DataType t) noexcept;
	bool isReal(DataType t) noexcept;
	bool isComplex(DataType t) noexcept;

	DataType typeFromString(const std::string &str) noexcept;
	std::string toString(DataType t);

	std::ostream& operator<<(std::ostream &stream, DataType dtype);
	std::string operator+(const std::string &lhs, DataType rhs);
	std::string operator+(DataType lhs, const std::string &rhs);

	class DataTypeNotSupported: public std::logic_error
	{
		public:
			DataTypeNotSupported(const char *function, const std::string &comment);
			DataTypeNotSupported(const char *function, const char *comment);
			DataTypeNotSupported(const char *function, DataType dtype);
			DataTypeNotSupported(const char *function, std::initializer_list<DataType> supported_types);
			DataTypeNotSupported(const char *function, DataType dtype, std::initializer_list<DataType> supported_types);
	};

	class DataTypeMismatch: public std::logic_error
	{
		public:
			DataTypeMismatch(const char *function, const std::string &comment);
			DataTypeMismatch(const char *function, DataType expected, DataType got);
	};

} /* namespace avocado */

#endif /* AVOCADO_CORE_DATATYPE_HPP_ */
