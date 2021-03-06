/*
 * conversions.hpp
 *
 *  Created on: Nov 30, 2021
 *      Author: maciek
 */

#ifndef AVOCADO_MATH_CONVERSIONS_HPP_
#define AVOCADO_MATH_CONVERSIONS_HPP_

#include <stddef.h>

namespace avocado
{
	class Context;
	class Tensor;
	enum class DataType;
}

namespace avocado
{
	namespace math
	{
		void changeType(void *dst, DataType dstType, const void *src, DataType srcType, size_t elements);

		void changeType(const Context &context, Tensor &dst, const Tensor &src);
		void changeType(const Context &context, const Tensor &tensor, DataType dstType);
	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_CONVERSIONS_HPP_ */
