/*
 * test_testing_util.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: maciek
 */

//#include <Avocado/utils/testing_helpers.hpp>
#include <Avocado/core/Device.hpp>
#include <Avocado/core/Tensor.hpp>

#include <gtest/gtest.h>

namespace avocado
{
//	TEST(TestTestingUtil, init_int32)
//	{
//		Tensor t( { 6 }, "int32", Device::cpu());
//		initForTest(t, 0.0);
//		EXPECT_EQ(t.get<int>( { 0 }), -49);
//		EXPECT_EQ(t.get<int>( { 1 }), -32);
//		EXPECT_EQ(t.get<int>( { 2 }), -15);
//		EXPECT_EQ(t.get<int>( { 3 }), 2);
//		EXPECT_EQ(t.get<int>( { 4 }), 19);
//		EXPECT_EQ(t.get<int>( { 5 }), 36);
//	}
//	TEST(TestTestingUtil, init_float32)
//	{
//		Tensor t( { 6 }, "float32", Device::cpu());
//		initForTest(t, 0.1);
//
//		EXPECT_FLOAT_EQ(t.get<float>( { 0 }), 0.09983341f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 1 }), 0.19866933f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 2 }), 0.29552020f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 3 }), 0.38941834f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 4 }), 0.47942553f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 5 }), 0.56464247f);
//	}
//	TEST(TestTestingUtil, init_cuda_float32)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Tensor t( { 6 }, "float32", Device::cuda(0));
//		initForTest(t, 0.1);
//
//		EXPECT_FLOAT_EQ(t.get<float>( { 0 }), 0.09983341f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 1 }), 0.19866933f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 2 }), 0.29552020f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 3 }), 0.38941834f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 4 }), 0.47942553f);
//		EXPECT_FLOAT_EQ(t.get<float>( { 5 }), 0.56464247f);
//	}
//	TEST(TestTestingUtil, diff_int32)
//	{
//		Tensor t1( { 10 }, "int32", Device::cpu());
//		Tensor t2( { 10 }, "int32", Device::cpu());
//		initForTest(t1, 0.0);
//		initForTest(t2, 10.0);
//		double diff1 = diffForTest(t1, t2);
//		EXPECT_GT(diff1, 10.0f);
//
//		initForTest(t2, 0.0);
//		double diff2 = diffForTest(t1, t2);
//		EXPECT_DOUBLE_EQ(diff2, 0.0);
//	}
//	TEST(TestTestingUtil, diff_float32)
//	{
//		Tensor t1( { 10 }, "float32", Device::cpu());
//		Tensor t2( { 10 }, "float32", Device::cpu());
//		initForTest(t1, 0.0);
//		initForTest(t2, 10.0);
//		double diff1 = diffForTest(t1, t2);
//		EXPECT_GT(diff1, 0.1);
//
//		initForTest(t2, 0.0);
//		double diff2 = diffForTest(t1, t2);
//		EXPECT_DOUBLE_EQ(diff2, 0.0);
//	}

} /* namespace avocado */

