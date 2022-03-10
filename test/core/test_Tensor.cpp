/*
 * test_Tensor.cpp
 *
 *  Created on: Sep 11, 2020
 *      Author: maciek
 */

#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>
#include <Avocado/utils/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace avocado
{
//	TEST(TestTensorOnCPU, construct_default)
//	{
//		Tensor tensor;
//
//		EXPECT_EQ(tensor.device(), Device::cpu());
//		EXPECT_EQ(tensor.dtype(), DataType::UNKNOWN);
//		EXPECT_EQ(tensor.sizeInBytes(), 0ull);
//		EXPECT_EQ(tensor.volume(), 0);
//		EXPECT_TRUE(tensor.isOwning());
//		EXPECT_FALSE(tensor.isView());
//		EXPECT_FALSE(tensor.isExpired());
//		EXPECT_TRUE(tensor.isEmpty());
//		EXPECT_EQ(tensor.dim(), 0);
//		EXPECT_FALSE(tensor.isPageLocked());
//		EXPECT_EQ(tensor.viewCount(), 1);
//		EXPECT_EQ(tensor.data(), nullptr);
//	}
//	TEST(TestTensorOnCPU, construct)
//	{
//		Shape ts( { 10 });
//		Tensor tensor(ts, "int32", Device::cpu());
//
//		EXPECT_EQ(tensor.device(), Device::cpu());
//		EXPECT_EQ(tensor.dtype(), DataType::INT32);
//		EXPECT_EQ(tensor.sizeInBytes(), 40ull);
//		EXPECT_EQ(tensor.volume(), 10);
//		EXPECT_TRUE(tensor.isOwning());
//		EXPECT_FALSE(tensor.isView());
//		EXPECT_FALSE(tensor.isExpired());
//		EXPECT_FALSE(tensor.isEmpty());
//		EXPECT_EQ(tensor.dim(), 1);
//		EXPECT_FALSE(tensor.isPageLocked());
//		EXPECT_EQ(tensor.viewCount(), 1);
//		EXPECT_NE(tensor.data(), nullptr);
//
//		for (int i = 0; i < tensor.volume(); i++)
//		{
//			EXPECT_EQ(tensor.get<int>( { i }), 0);
//		}
//	}
//	TEST(TestTensorOnCPU, construct_copy_from_owning)
//	{
//		Tensor tensor( { 10 }, "int32", Device::cpu());
//		Tensor copy = tensor;
//
//		EXPECT_EQ(copy.device(), tensor.device());
//		EXPECT_EQ(copy.dtype(), tensor.dtype());
//		EXPECT_EQ(copy.sizeInBytes(), tensor.sizeInBytes());
//		EXPECT_EQ(copy.volume(), tensor.volume());
//		EXPECT_EQ(copy.isOwning(), tensor.isOwning());
//		EXPECT_EQ(copy.isView(), tensor.isView());
//		EXPECT_EQ(copy.isExpired(), tensor.isExpired());
//		EXPECT_EQ(copy.dim(), tensor.dim());
//		EXPECT_EQ(copy.isPageLocked(), tensor.isPageLocked());
//		EXPECT_EQ(copy.viewCount(), tensor.viewCount());
//		EXPECT_NE(copy.data(), tensor.data());
//	}
//	TEST(TestTensorOnCPU, assign_copy_from_owning)
//	{
//		Tensor tensor( { 10 }, "int32", Device::cpu());
//		Tensor copy;
//		copy = tensor;
//
//		EXPECT_EQ(copy.device(), tensor.device());
//		EXPECT_EQ(copy.dtype(), tensor.dtype());
//		EXPECT_EQ(copy.sizeInBytes(), tensor.sizeInBytes());
//		EXPECT_EQ(copy.volume(), tensor.volume());
//		EXPECT_EQ(copy.isOwning(), tensor.isOwning());
//		EXPECT_EQ(copy.isView(), tensor.isView());
//		EXPECT_EQ(copy.isExpired(), tensor.isExpired());
//		EXPECT_EQ(copy.dim(), tensor.dim());
//		EXPECT_EQ(copy.isPageLocked(), tensor.isPageLocked());
//		EXPECT_EQ(copy.viewCount(), tensor.viewCount());
//		EXPECT_NE(copy.data(), tensor.data());
//	}
//
//	TEST(TestTensorOnCPU, set_get)
//	{
//		Shape ts( { 5, 4, 3, 2 });
//		Tensor tensor(ts, DataType::INT32, Device::cpu());
//		for (int i = 0; i < ts[0]; i++)
//			for (int i1 = 0; i1 < ts[1]; i1++)
//				for (int i2 = 0; i2 < ts[2]; i2++)
//					for (int i3 = 0; i3 < ts[3]; i3++)
//						tensor.set(i + i1 + i2 + i3, { i, i1, i2, i3 });
//
//		for (int i = 0; i < ts[0]; i++)
//			for (int i1 = 0; i1 < ts[1]; i1++)
//				for (int i2 = 0; i2 < ts[2]; i2++)
//					for (int i3 = 0; i3 < ts[3]; i3++)
//					{
//						EXPECT_EQ(tensor.get<int>( { i, i1, i2, i3 }), (i + i1 + i2 + i3));
//					}
//	}
//	TEST(TestTensorOnCPU, setall)
//	{
//		Shape ts( { 10 });
//		Tensor tensor(ts, DataType::INT32, Device::cpu());
//		tensor.setall(DeviceContext(), 11);
//		for (int i = 0; i < tensor.volume(); i++)
//		{
//			EXPECT_EQ(tensor.get<int>( { i }), 11);
//		}
//		tensor.zeroall(DeviceContext());
//		for (int i = 0; i < tensor.volume(); i++)
//		{
//			EXPECT_EQ(tensor.get<int>( { i }), 0);
//		}
//	}
//	TEST(TestTensorOnCPU, serialize)
//	{
//		Shape ts( { 10, 11, 12, 13 });
//		Tensor base(ts, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(base, 0.0);
//
//		SerializedObject so;
//		Json j = base.serialize(so);
//
//		EXPECT_EQ(so.size(), base.sizeInBytes());
//
//		Tensor loaded(j, so);
//
//		EXPECT_EQ(loaded.shape(), base.shape());
//		EXPECT_EQ(testing::diffForTest(base, loaded), 0.0);
//	}
//	TEST(TestTensorOnCPU, convert_to)
//	{
//		Shape ts( { 123 });
//		Tensor base(ts, DataType::FLOAT32, Device::cpu());
//		Tensor changed(ts, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(base, 0.0);
//		testing::initForTest(changed, 0.0);
//
//		changed.convertTo(DeviceContext(), DataType::INT32);
//		for (int i = 0; i < ts[0]; i++)
//			EXPECT_EQ(changed.get<int>( { i }), static_cast<int>(base.get<float>( { i })));
//	}
//
//	TEST(TestTensorOnCUDA, init)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Shape ts( { 10 });
//		Tensor tensor(ts, DataType::INT32, Device::cuda(0));
//		EXPECT_EQ(tensor.volume(), 10);
//		EXPECT_EQ(tensor.sizeInBytes(), 40ull);
//		for (int i = 0; i < tensor.volume(); i++)
//		{
//			EXPECT_EQ(tensor.get<int>( { i }), 0);
//		}
//	}
//	TEST(TestTensorOnCUDA, set_get)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Shape ts( { 5, 4, 3, 2 });
//		Tensor tensor(ts, DataType::INT32, Device::cuda(0));
//		for (int i = 0; i < ts[0]; i++)
//			for (int i1 = 0; i1 < ts[1]; i1++)
//				for (int i2 = 0; i2 < ts[2]; i2++)
//					for (int i3 = 0; i3 < ts[3]; i3++)
//						tensor.set(i + i1 + i2 + i3, { i, i1, i2, i3 });
//
//		for (int i = 0; i < ts[0]; i++)
//			for (int i1 = 0; i1 < ts[1]; i1++)
//				for (int i2 = 0; i2 < ts[2]; i2++)
//					for (int i3 = 0; i3 < ts[3]; i3++)
//					{
//						EXPECT_EQ(tensor.get<int>( { i, i1, i2, i3 }), (i + i1 + i2 + i3));
//					}
//	}
//	TEST(TestTensorOnCUDA, setall)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		DeviceContext context(Device::cuda(0));
//		Shape ts( { 10 });
//		Tensor tensor(ts, DataType::INT32, Device::cuda(0));
//		tensor.setall(context, 11);
//		for (int i = 0; i < tensor.volume(); i++)
//		{
//			EXPECT_EQ(tensor.get<int>( { i }), 11);
//		}
//		tensor.zeroall(context);
//		for (int i = 0; i < tensor.volume(); i++)
//		{
//			EXPECT_EQ(tensor.get<int>( { i }), 0);
//		}
//	}
//	TEST(TestTensorOnCUDA, serialize)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Shape ts( { 10, 11, 12, 13 });
//		Tensor base(ts, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(base, 0.0);
//
//		SerializedObject so;
//		Json j = base.serialize(so);
//
//		EXPECT_EQ(so.size(), base.sizeInBytes());
//
//		Tensor loaded(j, so);
//
//		EXPECT_EQ(loaded.shape(), base.shape());
//		EXPECT_EQ(testing::diffForTest(base, loaded), 0.0);
//	}
//	TEST(TestTensorOnCUDA, convert_to)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Shape ts( { 123 });
//		Tensor base(ts, DataType::FLOAT32, Device::cuda(0));
//		Tensor changed(ts, DataType::FLOAT32, Device::cuda(0));
//		testing::initForTest(base, 0.0);
//		testing::initForTest(changed, 0.0);
//
//		DeviceContext context(Device::cuda(0));
//		changed.convertTo(context, DataType::INT32);
//		for (int i = 0; i < ts[0]; i++)
//			EXPECT_EQ(changed.get<int>( { i }), static_cast<int>(base.get<float>( { i })));
//	}
//	TEST(TestTensorOnCUDA, move_to)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		Shape ts( { 10, 11, 12, 13 });
//		Tensor base(ts, DataType::FLOAT32, Device::cpu());
//		Tensor moved(ts, DataType::FLOAT32, Device::cpu());
//		testing::initForTest(base, 0.0);
//		testing::initForTest(moved, 0.0);
//
//		for (int i = 0; i < Device::numberOfCudaDevices(); i++)
//		{
//			moved.moveTo(Device::cuda(i));
//			EXPECT_EQ(testing::diffForTest(base, moved), 0.0);
//		}
//	}
//
//	TEST(TestTensorView, create_default)
//	{
//		Tensor tensor( { 10, 10 }, "float32", Device::cpu());
//		Tensor view = tensor.view();
//
//		tensor.set(1.0f, { 4, 5 });
//		EXPECT_EQ(view.get<float>( { 4, 5 }), 1.0f);
//	}
//	TEST(TestTensorView, create_with_offset)
//	{
//		Tensor tensor( { 10, 10 }, "float32", Device::cpu());
//		Tensor view = tensor.view( { 4, 4 }, 50);
//
//		tensor.set(1.0f, { 5, 9 });
//		EXPECT_EQ(view.get<float>( { 2, 1 }), 1.0f);
//	}
//	TEST(TestTensorView, create_and_expire)
//	{
//		Tensor view;
//		{
//			Tensor tensor( { 10, 10 }, "float32", Device::cpu());
//			view = tensor.view( { 4, 4 }, 50);
//		}
//		EXPECT_THROW(view.device(), LogicError);
//		EXPECT_THROW(view.dtype(), LogicError);
//		EXPECT_THROW(view.sizeInBytes(), LogicError);
//		EXPECT_EQ(view.volume(), 16);
//		EXPECT_FALSE(view.isOwning());
//		EXPECT_TRUE(view.isView());
//		EXPECT_TRUE(view.isExpired());
//		EXPECT_EQ(view.dim(), 2);
//		EXPECT_THROW(view.isPageLocked(), LogicError);
//		EXPECT_EQ(view.viewCount(), 1);
//		EXPECT_THROW(view.data(), LogicError);
//	}

} /* namespace avocado */

