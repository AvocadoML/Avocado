/*
 * serialization.hpp
 *
 *  Created on: May 6, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_UTILS_SERIALIZATION_HPP_
#define AVOCADO_UTILS_SERIALIZATION_HPP_

#include <vector>
#include <string>

namespace avocado
{
	class Json;
}

namespace avocado
{
	class SerializedObject
	{
		private:
			std::vector<char> m_data;
		public:
			explicit SerializedObject(size_t size = 0);
			size_t size() const noexcept;
			size_t capacity() const noexcept;
			void clear() noexcept;
			const char* data() const noexcept;
			char* data() noexcept;
			/**
			 * @brief Appends binary data to the end of internal array.
			 */
			void save(const void *src, size_t sizeInBytes);
			/**
			 * @brief Copies bytes into dst from internal array with specific offset.
			 */
			void load(void *dst, size_t offset, size_t sizeInBytes) const;
			/**
			 * @brief Appends given object to the end of internal array.
			 */
			template<typename T>
			void save(T value)
			{
				this->save(&value, sizeof(T));
			}
			/**
			 * @brief Loads object from internal array with specific offset.
			 */
			template<typename T>
			T load(size_t offset) const
			{
				T result;
				this->load(&result, offset, sizeof(T));
				return result;
			}
	};

} /* namespace avocado */

#endif /* AVOCADO_UTILS_SERIALIZATION_HPP_ */
