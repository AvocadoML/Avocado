/*
 * SerializedObject.cpp
 *
 *  Created on: May 6, 2020
 *      Author: Maciej Kozarzewski
 */

#include <avocado/utils/serialization.hpp>
#include <avocado/core/error_handling.hpp>

#include <cassert>
#include <cstring>

namespace avocado
{
	SerializedObject::SerializedObject(size_t size)
	{
		m_data.reserve(size);
	}
	size_t SerializedObject::size() const noexcept
	{
		return m_data.size();
	}
	size_t SerializedObject::capacity() const noexcept
	{
		return m_data.capacity();
	}
	void SerializedObject::clear() noexcept
	{
		m_data.clear();
	}
	const char* SerializedObject::data() const noexcept
	{
		return m_data.data();
	}
	char* SerializedObject::data() noexcept
	{
		return m_data.data();
	}
	void SerializedObject::save(const void *src, size_t size_in_bytes)
	{
		if (src == nullptr)
			throw IllegalArgument(METHOD_NAME, "'src' pointer must not be null");
		const char *ptr = reinterpret_cast<const char*>(src);
		m_data.insert(m_data.end(), ptr, ptr + size_in_bytes);
	}
	void SerializedObject::load(void *dst, size_t offset, size_t size_in_bytes) const
	{
		if (dst == nullptr)
			throw IllegalArgument(METHOD_NAME, "'dst' pointer must not be null");
		if (offset + size_in_bytes > m_data.size())
			throw IllegalArgument(METHOD_NAME, "trying to load more bytes than available");
		std::memcpy(reinterpret_cast<char*>(dst), m_data.data() + offset, size_in_bytes);
	}
} /* namespace avocado */

