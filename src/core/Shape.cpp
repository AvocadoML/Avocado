/*
 * Shape.cpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <avocado/core/Shape.hpp>
#include <avocado/utils/json.hpp>
#include <avocado/utils/serialization.hpp>
#include <avocado/core/error_handling.hpp>

#include <avocado/backend/backend_defs.h>

#include <cstring>

namespace avocado
{
	Shape::Shape() noexcept
	{
		std::memset(m_dim, 0, sizeof(m_dim));
	}
	Shape::Shape(const Json &json) :
			m_length(json.size())
	{
		std::memset(m_dim, 0, sizeof(m_dim));
		if (m_length < 0)
			throw IllegalArgument(METHOD_NAME, "length", "must be greater or equal 0", m_length);
		if (m_length > max_dimension)
			throw IllegalArgument(METHOD_NAME, "length", "must not exceed " + std::to_string(max_dimension), m_length);

		for (int i = 0; i < m_length; i++)
			m_dim[i] = json[i];
	}
	Shape::Shape(std::initializer_list<int> dims) :
			m_length(dims.size())
	{
		std::memset(m_dim, 0, sizeof(m_dim));
		std::memcpy(m_dim, dims.begin(), sizeof(int) * m_length);
	}
	Shape::Shape(const std::vector<int> &dims) :
			m_length(dims.size())
	{
		std::memset(m_dim, 0, sizeof(m_dim));
		std::memcpy(m_dim, dims.data(), sizeof(int) * m_length);
	}
	std::string Shape::toString() const
	{
		std::string result = "[";
		for (int i = 0; i < m_length; i++)
		{
			if (i != 0)
				result += 'x';
			result += std::to_string(m_dim[i]);
		}
		result += ']';
		return result;
	}
	int Shape::length() const noexcept
	{
		return m_length;
	}
	int Shape::at(int index) const
	{
		if (index < 0 || index >= m_length)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_length);
		return m_dim[index];
	}
	int& Shape::at(int index)
	{
		if (index < 0 || index >= m_length)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_length);
		return m_dim[index];
	}
	int Shape::operator[](int index) const
	{
		if (index < 0 || index >= m_length)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_length);
		return m_dim[index];
	}
	int& Shape::operator[](int index)
	{
		if (index < 0 || index >= m_length)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_length);
		return m_dim[index];
	}
	const int* Shape::data() const noexcept
	{
		return m_dim;
	}

	int Shape::firstDim() const noexcept
	{
		if (m_length == 0)
			return 0;
		else
			return m_dim[0];
	}
	int Shape::lastDim() const noexcept
	{
		if (m_length == 0)
			return 0;
		else
			return m_dim[m_length - 1];
	}
	int Shape::volume() const noexcept
	{
		if (m_length == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < m_length; i++)
				result *= m_dim[i];
			return result;
		}
	}
	int Shape::volumeWithoutFirstDim() const noexcept
	{
		if (m_length <= 1)
			return 0;
		else
		{
			int result = 1;
			for (int i = 1; i < m_length; i++)
				result *= m_dim[i];
			return result;
		}
	}
	int Shape::volumeWithoutLastDim() const noexcept
	{
		if (m_length <= 1)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < m_length - 1; i++)
				result *= m_dim[i];
			return result;
		}
	}
	int Shape::volume(std::initializer_list<int> dims) const
	{
		if (m_length == 0 || dims.size() == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < static_cast<int>(dims.size()); i++)
			{
				int index = dims.begin()[i];
				if (index < 0 || index >= m_length)
					throw IndexOutOfBounds(METHOD_NAME, "index" + std::to_string(i), index, m_length);
				result *= m_dim[index];
			}
			return result;
		}
	}

	bool operator==(const Shape &lhs, const Shape &rhs) noexcept
	{
		if (lhs.m_length != rhs.m_length)
			return false;
		for (int i = 0; i < lhs.m_length; i++)
			if (lhs.m_dim[i] != rhs.m_dim[i])
				return false;
		return true;
	}
	bool operator!=(const Shape &lhs, const Shape &rhs) noexcept
	{
		return !(lhs == rhs);
	}

	Json Shape::toJson() const
	{
		return Json(m_dim, m_length);
	}

	std::ostream& operator<<(std::ostream &stream, const Shape &s)
	{
		stream << s.toString();
		return stream;
	}
	std::string operator+(const std::string &str, const Shape &shape)
	{
		return str + shape.toString();
	}
	std::string operator+(const Shape &shape, const std::string &str)
	{
		return shape.toString() + str;
	}

	ShapeMismatch::ShapeMismatch(const char *function, const std::string &what_arg) :
			logic_error(std::string(function) + " : " + what_arg)
	{
	}
	ShapeMismatch::ShapeMismatch(const char *function, int expected_dim, int actual_dim) :
			logic_error(std::string(function) + " : expected " + std::to_string(expected_dim) + "D shape, got " + std::to_string(actual_dim) + "D")
	{
	}
	ShapeMismatch::ShapeMismatch(const char *function, const Shape &expected, const Shape &got) :
			logic_error(std::string(function) + " : expected shape " + expected + ", got " + got)
	{
	}

} /* namespace avocado */

