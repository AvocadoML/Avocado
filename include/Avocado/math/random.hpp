/*
 * random.hpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_RANDOM_HPP_
#define AVOCADO_MATH_RANDOM_HPP_

#include <cstdint>

namespace avocado
{
	namespace math
	{
		float randFloat();
		double randDouble();
		float randGaussian();
		int32_t randInt();
		int32_t randInt(int r);
		int32_t randInt(int r0, int r1);
		int64_t randLong();
		bool randBool();
	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_RANDOM_HPP_ */
