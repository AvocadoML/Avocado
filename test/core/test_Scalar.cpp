/*
 * test_Scalar.cpp
 *
 *  Created on: Feb 23, 2021
 *      Author: maciek
 */

#include <gtest/gtest.h>
#include <Avocado/core/Scalar.hpp>

namespace avocado
{
	TEST(TestScalar, asType)
	{
		Scalar s1(1);
		Scalar s2 = s1.asType(DataType::FLOAT64);

		EXPECT_EQ(s2.dtype(), DataType::FLOAT64);
		EXPECT_DOUBLE_EQ(s2.get<double>(), 1.0);
	}

} /* namespace avocado */

