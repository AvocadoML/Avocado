/*
 * Scalar.hpp
 *
 *  Created on: May 10, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_CORE_SCALAR_HPP_
#define AVOCADO_CORE_SCALAR_HPP_

#include <Avocado/core/DataType.hpp>
#include <Avocado/core/error_handling.hpp>
#include <Avocado/math/conversions.hpp>
#include <Avocado/backend/backend_defs.h>

#include <string>
#include <cstring>

namespace avocado
{
	class Scalar
	{
		private:
			uint8_t m_data[16];
			DataType m_dtype = DataType::UNKNOWN;
		public:

			Scalar() = default;
			Scalar(DataType dtype);
			template<typename T>
			Scalar(T value) :
					m_dtype(typeOf<T>())
			{
				static_assert(sizeof(T) <= 16, "max Scalar size is 16 bytes");
				if (m_dtype == DataType::UNKNOWN)
					throw DataTypeNotSupported( METHOD_NAME, "unknown data type");

				std::memcpy(m_data, &value, sizeof(T));
			}
			template<typename T>
			Scalar& operator=(const T &value)
			{
				static_assert(sizeof(T) <= 16, "max Scalar size is 16 bytes");
				if (typeOf<T>() == DataType::UNKNOWN)
					throw DataTypeNotSupported( METHOD_NAME, "unknown data type");

				m_dtype = typeOf<T>();
				std::memcpy(m_data, &value, sizeof(T));
				return *this;
			}

			std::string toString() const;
			const void* data() const noexcept;
			void* data() noexcept;
			size_t sizeInBytes() const noexcept;

			DataType dtype() const noexcept;

			template<typename T>
			T get() const
			{
				if (typeOf<T>() == DataType::UNKNOWN)
					throw DataTypeNotSupported( METHOD_NAME, "unknown data type");
				T result;
				math::changeType(&result, typeOf<T>(), m_data, m_dtype, 1);
				return result;
			}
			Scalar asType(DataType newType) const;

	};

	std::ostream& operator<<(std::ostream &stream, const Scalar &s);
	std::string operator+(const std::string &str, const Scalar &scalar);
	std::string operator+(const Scalar &scalar, const std::string &str);

} /* namespace avocado */

#endif /* AVOCADO_CORE_SCALAR_HPP_ */
