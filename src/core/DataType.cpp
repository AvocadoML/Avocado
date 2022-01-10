/*
 * DataType.cpp
 *
 *  Created on: May 10, 2020
 *      Author: Maciej Kozarzewski
 */

#include <avocado/core/DataType.hpp>

namespace
{
	std::string convert_list(const std::initializer_list<avocado::DataType> &list)
	{
		std::string result;
		if (list.size() == 1)
			result = "supported type is {";
		else
			result = "supported types are {";
		for (auto i = list.begin(); i < list.end(); i++)
		{
			if (i != list.begin())
				result += ", ";
			result += toString(*i);
		}
		return result;
	}
}

namespace avocado
{

	size_t sizeOf(DataType t) noexcept
	{
		switch (t)
		{
			case DataType::INT8:
			case DataType::UINT8:
				return 1;
			case DataType::INT16:
				return 2;
			case DataType::INT32:
				return 4;
			case DataType::INT64:
				return 8;
			case DataType::FLOAT16:
			case DataType::BFLOAT16:
				return 2;
			case DataType::FLOAT32:
				return 4;
			case DataType::FLOAT64:
				return 8;
			case DataType::COMPLEX32:
				return 8;
			case DataType::COMPLEX64:
				return 16;
			default:
				return 0;
		}
	}
	bool isInteger(DataType t) noexcept
	{
		return t >= DataType::UINT8 && t <= DataType::INT64;
	}
	bool isFloatingPoint(DataType t) noexcept
	{
		return t >= DataType::FLOAT16;
	}
	bool isReal(DataType t) noexcept
	{
		return t >= DataType::UINT8 && t <= DataType::FLOAT64;
	}
	bool isComplex(DataType t) noexcept
	{
		return t == DataType::COMPLEX32 || t == DataType::COMPLEX64;
	}

	DataType typeFromString(const std::string &str) noexcept
	{
		if (str == "uint8" || str == "UINT8")
			return DataType::UINT8;
		if (str == "int8" || str == "INT8")
			return DataType::INT8;
		if (str == "int16" || str == "INT16")
			return DataType::INT16;
		if (str == "int32" || str == "INT32")
			return DataType::INT32;
		if (str == "int64" || str == "INT64")
			return DataType::INT64;
		if (str == "float16" || str == "FLOAT16")
			return DataType::FLOAT16;
		if (str == "bfloat16" || str == "BFLOAT16")
			return DataType::BFLOAT16;
		if (str == "float32" || str == "FLOAT32")
			return DataType::FLOAT32;
		if (str == "float64" || str == "FLOAT64")
			return DataType::FLOAT64;
		if (str == "complex32" || str == "COMPLEX32")
			return DataType::COMPLEX32;
		if (str == "complex64" || str == "COMPLEX64")
			return DataType::COMPLEX64;
		return DataType::UNKNOWN;
	}
	std::string toString(DataType t)
	{
		switch (t)
		{
			case DataType::UINT8:
				return std::string("UINT8");
			case DataType::INT8:
				return std::string("INT8");
			case DataType::INT16:
				return std::string("INT16");
			case DataType::INT32:
				return std::string("INT32");
			case DataType::INT64:
				return std::string("INT64");
			case DataType::FLOAT16:
				return std::string("FLOAT16");
			case DataType::BFLOAT16:
				return std::string("BFLOAT16");
			case DataType::FLOAT32:
				return std::string("FLOAT32");
			case DataType::FLOAT64:
				return std::string("FLOAT64");
			case DataType::COMPLEX32:
				return std::string("COMPLEX32");
			case DataType::COMPLEX64:
				return std::string("COMPLEX64");
			default:
				return std::string("UNKNOWN");
		}
	}

	std::ostream& operator<<(std::ostream &stream, DataType t)
	{
		stream << toString(t);
		return stream;
	}
	std::string operator+(const std::string &lhs, DataType rhs)
	{
		return lhs + toString(rhs);
	}
	std::string operator+(DataType lhs, const std::string &rhs)
	{
		return toString(lhs) + rhs;
	}

	DataTypeNotSupported::DataTypeNotSupported(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}
	DataTypeNotSupported::DataTypeNotSupported(const char *function, const char *comment) :
			DataTypeNotSupported(function, std::string(comment))
	{
	}
	DataTypeNotSupported::DataTypeNotSupported(const char *function, DataType dtype) :
			std::logic_error(std::string(function) + " : " + dtype + " is not supported")
	{
	}
	DataTypeNotSupported::DataTypeNotSupported(const char *function, std::initializer_list<DataType> supported_types) :
			std::logic_error(std::string(function) + " : " + convert_list(supported_types))
	{
	}
	DataTypeNotSupported::DataTypeNotSupported(const char *function, DataType dtype, std::initializer_list<DataType> supported_types) :
			std::logic_error(std::string(function) + " : " + convert_list(supported_types) + ", got " + dtype)
	{
	}

	DataTypeMismatch::DataTypeMismatch(const char *function, const std::string &comment) :
			logic_error(std::string(function) + " : " + comment)
	{
	}
	DataTypeMismatch::DataTypeMismatch(const char *function, DataType expected, DataType got) :
			std::logic_error(std::string(function) + " : expected type " + expected + ", got " + got)
	{
	}

} /* namespace avocado */
