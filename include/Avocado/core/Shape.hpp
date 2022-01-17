/*
 * Shape.hpp
 *
 *  Created on: May 8, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_CORE_SHAPE_HPP_
#define AVOCADO_CORE_SHAPE_HPP_

#include <Avocado/backend/backend_defs.h>
#include <stddef.h>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace avocado /* forward declarations */
{
	class Json;
}

namespace avocado
{
	class Shape
	{
		public:
			static const int max_dimension = backend::AVOCADO_MAX_TENSOR_DIMENSIONS;
		private:
			int m_dim[max_dimension];
			int m_length = 0;
		public:
			Shape() noexcept;
			Shape(const Json &json);
			Shape(std::initializer_list<int> dims);
			Shape(const std::vector<int> &dims);

			std::string toString() const;

			int length() const noexcept;
			int at(int index) const;
			int& at(int index);
			int operator[](int index) const;
			int& operator[](int index);
			const int* data() const noexcept;

			int firstDim() const noexcept;
			int lastDim() const noexcept;
			int volume() const noexcept;
			int volumeWithoutFirstDim() const noexcept;
			int volumeWithoutLastDim() const noexcept;
			int volume(std::initializer_list<int> dims) const;

			friend bool operator==(const Shape &lhs, const Shape &rhs) noexcept;
			friend bool operator!=(const Shape &lhs, const Shape &rhs) noexcept;

			Json toJson() const;
	};

	std::ostream& operator<<(std::ostream &stream, const Shape &s);
	std::string operator+(const std::string &lhs, const Shape &rhs);
	std::string operator+(const Shape &lhs, const std::string &rhs);

	class ShapeMismatch: public std::logic_error
	{
		public:
			ShapeMismatch(const char *function, const std::string &what_arg);
			ShapeMismatch(const char *function, int expected_dim, int actual_dim);
			ShapeMismatch(const char *function, const Shape &expected, const Shape &got);
	};

} /* namespace avocado */

#endif /* AVOCADO_CORE_SHAPE_HPP_ */
