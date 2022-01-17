/*
 * testing_util.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/DataType.hpp>
#include <Avocado/utils/testing_helpers.hpp>

#include <complex>
#include <memory>

namespace
{
	using namespace avocado;
	template<typename T>
	void init_for_test_float(void *ptr, size_t length, T shift)
	{
		for (size_t i = 0; i < length; i++)
			reinterpret_cast<T*>(ptr)[i] = sin(i / 10.0f + shift);
	}
//	void init_for_test_float16(void *ptr, size_t length, float shift)
//	{
//		for (size_t i = 0; i < length; i++)
//			reinterpret_cast<float16*>(ptr)[i] = sinf(i / 10.0f + shift);
//	}
	template<typename T>
	void init_for_test_int(void *ptr, size_t length, T shift)
	{
		for (size_t i = 0; i < length; i++)
			reinterpret_cast<T*>(ptr)[i] = (17 * (i + shift)) % 97 - 49;
	}

	template<typename T>
	float diff_for_test(const void *ptr1, const void *ptr2, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += fabs(reinterpret_cast<const T*>(ptr1)[i] - reinterpret_cast<const T*>(ptr2)[i]);
		return result / length;
	}
//	template<>
//	float diff_for_test<float16>(const void *ptr1, const void *ptr2, size_t length)
//	{
//		double result = 0.0;
//		for (size_t i = 0; i < length; i++)
//			result += fabsf(
//					static_cast<float>(reinterpret_cast<const float16*>(ptr1)[i]) - static_cast<float>(reinterpret_cast<const float16*>(ptr2)[i]));
//		return result / length;
//	}

	template<typename T>
	float norm_for_test(const void *ptr, size_t length)
	{
		double result = 0.0;
		for (size_t i = 0; i < length; i++)
			result += fabs(reinterpret_cast<const T*>(ptr)[i]);
		return result;
	}
//	template<>
//	float norm_for_test<float16>(const void *ptr, size_t length)
//	{
//		double result = 0.0;
//		for (size_t i = 0; i < length; i++)
//			result += fabsf(static_cast<float>(reinterpret_cast<const float16*>(ptr)[i]));
//		return result;
//	}

	template<typename T>
	void abs_for_test(void *ptr, size_t length)
	{
		for (size_t i = 0; i < length; i++)
			reinterpret_cast<T*>(ptr)[i] = fabs(reinterpret_cast<T*>(ptr)[i]);
	}
//	template<>
//	void abs_for_test<float16>(void *ptr, size_t length)
//	{
//		for (size_t i = 0; i < length; i++)
//			reinterpret_cast<float16*>(ptr)[i] = fabsf(static_cast<float>(reinterpret_cast<float16*>(ptr)[i]));
//	}
}

namespace avocado
{
	bool isDeviceAvailable(const std::string &str)
	{
		try
		{
			[[maybe_unused]] Device d = Device::fromString(str);
			return true;
		} catch (std::exception &e)
		{
			return false;
		}
	}

	void initForTest(Tensor &t, double shift)
	{
		std::unique_ptr<char[]> tmp = std::make_unique<char[]>(t.sizeInBytes());
		switch (t.dtype())
		{
			case DataType::UINT8:
				init_for_test_int<uint8_t>(tmp.get(), t.volume(), shift);
				break;
			case DataType::INT8:
				init_for_test_int<int8_t>(tmp.get(), t.volume(), shift);
				break;
			case DataType::INT16:
				init_for_test_int<int16_t>(tmp.get(), t.volume(), shift);
				break;
			case DataType::INT32:
				init_for_test_int<int32_t>(tmp.get(), t.volume(), shift);
				break;
			case DataType::INT64:
				init_for_test_int<int64_t>(tmp.get(), t.volume(), shift);
				break;
			case DataType::FLOAT16:
//				init_for_test_float16(tmp.get(), t.volume(), shift);
				break;
			case DataType::BFLOAT16:
				throw DataTypeNotSupported(METHOD_NAME, t.dtype());
			case DataType::FLOAT32:
				init_for_test_float<float>(tmp.get(), t.volume(), shift);
				break;
			case DataType::FLOAT64:
				init_for_test_float<double>(tmp.get(), t.volume(), shift);
				break;
			case DataType::COMPLEX32:
				init_for_test_float<float>(tmp.get(), 2 * t.volume(), shift);
				break;
			case DataType::COMPLEX64:
				init_for_test_float<double>(tmp.get(), 2 * t.volume(), shift);
				break;
			case DataType::UNKNOWN:
				throw DataTypeNotSupported(METHOD_NAME, t.dtype());
		}
		t.copyFrom(tmp.get(), t.volume());
	}
	double diffForTest(const Tensor &lhs, const Tensor &rhs)
	{
		assert(lhs.shape() == rhs.shape());
		assert(lhs.dtype() == rhs.dtype());

		if (lhs.volume() == 0)
			return 0.0;

		std::unique_ptr<char[]> tmp_lhs = std::make_unique<char[]>(lhs.sizeInBytes());
		std::unique_ptr<char[]> tmp_rhs = std::make_unique<char[]>(rhs.sizeInBytes());
		lhs.copyTo(tmp_lhs.get(), lhs.volume());
		rhs.copyTo(tmp_rhs.get(), rhs.volume());
		switch (lhs.dtype())
		{
			case DataType::UINT8:
				return diff_for_test<uint8_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
			case DataType::INT8:
				return diff_for_test<int8_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
			case DataType::INT16:
				return diff_for_test<int16_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
			case DataType::INT32:
				return diff_for_test<int32_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
			case DataType::INT64:
				return diff_for_test<int64_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
			case DataType::FLOAT16:
//				return diff_for_test<float16>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
			case DataType::BFLOAT16:
				throw DataTypeNotSupported(METHOD_NAME, lhs.dtype());
			case DataType::FLOAT32:
				return diff_for_test<float>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
			case DataType::FLOAT64:
				return diff_for_test<double>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
			case DataType::COMPLEX32:
				return diff_for_test<float>(tmp_lhs.get(), tmp_rhs.get(), 2 * lhs.volume());
			case DataType::COMPLEX64:
				return diff_for_test<double>(tmp_lhs.get(), tmp_rhs.get(), 2 * lhs.volume());
			default:
			case DataType::UNKNOWN:
				throw DataTypeNotSupported(METHOD_NAME, lhs.dtype());
		}
	}
	double normForTest(const Tensor &tensor)
	{
		std::unique_ptr<char[]> tmp = std::make_unique<char[]>(tensor.sizeInBytes());
		tensor.copyTo(tmp.get(), tensor.volume());
		switch (tensor.dtype())
		{
			case DataType::UINT8:
				return norm_for_test<uint8_t>(tmp.get(), tensor.volume());
			case DataType::INT8:
				return norm_for_test<int8_t>(tmp.get(), tensor.volume());
			case DataType::INT16:
				return norm_for_test<int16_t>(tmp.get(), tensor.volume());
			case DataType::INT32:
				return norm_for_test<int32_t>(tmp.get(), tensor.volume());
			case DataType::INT64:
				return norm_for_test<int64_t>(tmp.get(), tensor.volume());
			case DataType::FLOAT16:
//				return norm_for_test<float16>(tmp.get(), tensor.volume());
			case DataType::BFLOAT16:
				throw DataTypeNotSupported(METHOD_NAME, tensor.dtype());
			case DataType::FLOAT32:
				return norm_for_test<float>(tmp.get(), tensor.volume());
			case DataType::FLOAT64:
				return norm_for_test<double>(tmp.get(), tensor.volume());
			case DataType::COMPLEX32:
				return norm_for_test<float>(tmp.get(), 2 * tensor.volume());
			case DataType::COMPLEX64:
				return norm_for_test<double>(tmp.get(), 2 * tensor.volume());
			default:
			case DataType::UNKNOWN:
				throw DataTypeNotSupported(METHOD_NAME, tensor.dtype());
		}
	}
	void absForTest(Tensor &tensor)
	{
		std::unique_ptr<char[]> tmp = std::make_unique<char[]>(tensor.sizeInBytes());
		tensor.copyTo(tmp.get(), tensor.volume());
		switch (tensor.dtype())
		{
			case DataType::UINT8:
				abs_for_test<uint8_t>(tmp.get(), tensor.volume());
				break;
			case DataType::INT8:
				abs_for_test<int8_t>(tmp.get(), tensor.volume());
				break;
			case DataType::INT16:
				abs_for_test<int16_t>(tmp.get(), tensor.volume());
				break;
			case DataType::INT32:
				abs_for_test<int32_t>(tmp.get(), tensor.volume());
				break;
			case DataType::INT64:
				abs_for_test<int64_t>(tmp.get(), tensor.volume());
				break;
			case DataType::FLOAT16:
//				abs_for_test<float16>(tmp.get(), tensor.volume());
				break;
			case DataType::BFLOAT16:
				throw DataTypeNotSupported(METHOD_NAME, tensor.dtype());
			case DataType::FLOAT32:
				abs_for_test<float>(tmp.get(), tensor.volume());
				break;
			case DataType::FLOAT64:
				abs_for_test<double>(tmp.get(), tensor.volume());
				break;
			case DataType::COMPLEX32:
				abs_for_test<float>(tmp.get(), tensor.volume());
				break;
			case DataType::COMPLEX64:
				abs_for_test<double>(tmp.get(), tensor.volume());
				break;
			case DataType::UNKNOWN:
				throw DataTypeNotSupported(METHOD_NAME, tensor.dtype());
		}
		tensor.copyFrom(tmp.get(), tensor.volume());
	}
	void printForTest(const Tensor &tensor)
	{
		if (tensor.numberOfDimensions() == 1)
		{
			for (int i = 0; i < tensor.volume(); i++)
				std::cout << tensor.get<float>( { i }) << ' ';
			std::cout << '\n';
		}
		if (tensor.numberOfDimensions() == 2)
		{
			for (int i = 0; i < tensor.firstDim(); i++)
			{
				for (int j = 0; j < tensor.lastDim(); j++)
					std::cout << tensor.get<float>( { i, j }) << ' ';
				std::cout << '\n';
			}
		}
	}
} /* namespace avocado */

