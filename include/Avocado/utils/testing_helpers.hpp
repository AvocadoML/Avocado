/*
 * testing_util.hpp
 *
 *  Created on: Sep 13, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_UTILS_TESTING_HELPERS_HPP_
#define AVOCADO_UTILS_TESTING_HELPERS_HPP_

#include <string>

namespace avocado
{
	class Tensor;
}

namespace avocado
{
	bool isDeviceAvailable(const std::string &str);

	void initForTest(Tensor &t, double shift);
	double diffForTest(const Tensor &lhs, const Tensor &rhs);
	double normForTest(const Tensor &tensor);
	void absForTest(Tensor &tensor);
	void printForTest(const Tensor &tensor);
} /* namespace avocado */

#endif /* AVOCADO_UTILS_TESTING_HELPERS_HPP_ */
