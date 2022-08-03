/*
 * Shape.cpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/core/Shape.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>
#include <Avocado/core/error_handling.hpp>

#include <cstring>

namespace avocado
{
	Shape::Shape() noexcept
	{
		m_dimensions.fill(0);
	}
	Shape::Shape(const Json &json) :
			m_rank(json.size())
	{
		m_dimensions.fill(0);
		if (m_rank < 0)
			throw IllegalArgument(METHOD_NAME, "length", "must be greater or equal 0", m_rank);
		if (m_rank > max_dimension)
			throw IllegalArgument(METHOD_NAME, "length", "must not exceed " + std::to_string(max_dimension), m_rank);

		for (int i = 0; i < m_rank; i++)
			m_dimensions[i] = json[i];
	}
	Shape::Shape(std::initializer_list<int> dims) :
			m_rank(dims.size())
	{
		m_dimensions.fill(0);
		std::copy(dims.begin(), dims.end(), m_dimensions.begin());
	}
	Shape::Shape(const std::vector<int> &dims) :
			m_rank(dims.size())
	{
		m_dimensions.fill(0);
		std::copy(dims.begin(), dims.end(), m_dimensions.begin());
	}
	std::string Shape::toString() const
	{
		std::string result = "[";
		for (int i = 0; i < m_rank; i++)
		{
			if (i != 0)
				result += 'x';
			result += std::to_string(m_dimensions[i]);
		}
		result += ']';
		return result;
	}
	int Shape::length() const noexcept
	{
		return m_rank;
	}
	int Shape::rank() const noexcept
	{
		return m_rank;
	}
	int Shape::at(int index) const
	{
		if (index < 0 || index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);
		return m_dimensions[index];
	}
	int& Shape::at(int index)
	{
		if (index < 0 || index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);
		return m_dimensions[index];
	}
	int Shape::operator[](int index) const
	{
		if (index < 0 || index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);
		return m_dimensions[index];
	}
	int& Shape::operator[](int index)
	{
		if (index < 0 || index >= m_rank)
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_rank);
		return m_dimensions[index];
	}
	const int* Shape::data() const noexcept
	{
		return m_dimensions.data();
	}

	int Shape::firstDim() const noexcept
	{
		if (m_rank == 0)
			return 0;
		else
			return m_dimensions[0];
	}
	int Shape::lastDim() const noexcept
	{
		if (m_rank == 0)
			return 0;
		else
			return m_dimensions[m_rank - 1];
	}
	int Shape::volume() const noexcept
	{
		if (m_rank == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < m_rank; i++)
				result *= m_dimensions[i];
			return result;
		}
	}
	int Shape::volumeWithoutFirstDim() const noexcept
	{
		if (m_rank <= 1)
			return 0;
		else
		{
			int result = 1;
			for (int i = 1; i < m_rank; i++)
				result *= m_dimensions[i];
			return result;
		}
	}
	int Shape::volumeWithoutLastDim() const noexcept
	{
		if (m_rank <= 1)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < m_rank - 1; i++)
				result *= m_dimensions[i];
			return result;
		}
	}
	int Shape::volume(std::initializer_list<int> dims) const
	{
		return volumeOverDims(dims);
	}
	int Shape::volumeOverDims(std::initializer_list<int> dims) const
	{
		return volumeOverDims(dims);
	}
	int Shape::volumeOverDims(const std::vector<int> &dims) const
	{
		if (m_rank == 0 || dims.size() == 0)
			return 0;
		else
		{
			int result = 1;
			for (int i = 0; i < static_cast<int>(dims.size()); i++)
			{
				int index = dims.begin()[i];
				if (index < 0 || index >= m_rank)
					throw IndexOutOfBounds(METHOD_NAME, "index" + std::to_string(i), index, m_rank);
				result *= m_dimensions[index];
			}
			return result;
		}
	}

	bool operator==(const Shape &lhs, const Shape &rhs) noexcept
	{
		if (lhs.m_rank != rhs.m_rank)
			return false;
		for (int i = 0; i < lhs.m_rank; i++)
			if (lhs.m_dimensions[i] != rhs.m_dimensions[i])
				return false;
		return true;
	}
	bool operator!=(const Shape &lhs, const Shape &rhs) noexcept
	{
		return !(lhs == rhs);
	}

	Json Shape::toJson() const
	{
		return Json(m_dimensions.data(), m_rank);
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

