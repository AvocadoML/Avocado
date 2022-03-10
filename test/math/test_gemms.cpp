/*
 * test_gemms.cpp
 *
 *  Created on: Sep 12, 2020
 *      Author: maciek
 */

#include <Avocado/math/tensor_operations.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/testing_helpers.hpp>

#include <Avocado/backend/backend_defs.h>
#include <ReferenceBackend/reference_backend.h>

#include <gtest/gtest.h>

namespace
{
	using namespace avocado;

	class GemmTester
	{
		private:
			math::GemmOp op_A;
			math::GemmOp op_B;
		public:

			Tensor A;
			Tensor B;
			Tensor C_baseline;
			Tensor C_tested;

			GemmTester(int M, int N, int K, math::GemmOp opA, math::GemmOp opB, DataType dtype, DataType compute_type) :
					op_A(opA),
					op_B(opB)
			{
				Shape sh_A = (opA == math::GemmOp::OP_N) ? Shape( { M, K }) : Shape( { K, M });
				Shape sh_B = (opB == math::GemmOp::OP_N) ? Shape( { K, N }) : Shape( { N, K });
				A = Tensor(sh_A, dtype, Device::cpu());
				B = Tensor(sh_B, dtype, Device::cpu());

				initForTest(A, 0.0);
				initForTest(B, 1.57);

				C_baseline = Tensor( { M, N }, compute_type, Device::cpu());
				C_tested = Tensor( { M, N }, compute_type, Device::cpu());
			}
			GemmTester(int M, int N, int K, math::GemmOp opA, math::GemmOp opB, DataType dtype) :
					GemmTester(M, N, K, opA, opB, dtype, dtype)
			{
			}
			template<typename C_type, typename AB_type, typename T = C_type>
			void gemm_baseline(Scalar alpha, Scalar beta) noexcept
			{
				alpha.toScalingTypeFor(typeOf<C_type>());
				beta.toScalingTypeFor(typeOf<C_type>());
				backend::refGemm(0, static_cast<backend::avGemmOperation_t>(op_A), static_cast<backend::avGemmOperation_t>(op_B), alpha.data(),
						A.getDescriptor(), A.getMemory(), B.getDescriptor(), B.getMemory(), beta.data(), C_baseline.getDescriptor(),
						C_baseline.getMemory());
			}
			double getDifference() const noexcept
			{
				return diffForTest(C_baseline, C_tested);
			}
			void moveTo(Device device)
			{
				A.moveTo(device);
				B.moveTo(device);
				C_tested.moveTo(device);
			}
	};
}

namespace avocado
{

	TEST(TestGemmOnCPU, float32_AB)
	{
		math::GemmOp opA = math::GemmOp::OP_N;
		math::GemmOp opB = math::GemmOp::OP_N;
		GemmTester data(23, 45, 67, opA, opB, DataType::FLOAT32);
		data.gemm_baseline<float, float>(1.1, 0.1);

//		Context context;
//		math::gemm(context, opA, opB, data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
	}
//	TEST(TestGemmOnCPU, float32_ABT)
//	{
//		GemmTester data(23, 45, 67, 'n', 't', DataType::FLOAT32);
//		data.gemm_baseline<float, float>(1.1, 0.1);
//
//		math::gemm(DeviceContext(), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//	TEST(TestGemmOnCPU, float32_ATB)
//	{
//		GemmTester data(23, 45, 67, 't', 'n', DataType::FLOAT32);
//		data.gemm_baseline<float, float>(1.1, 0.1);
//
//		math::gemm(DeviceContext(), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//	TEST(TestGemmOnCPU, float32_ATBT)
//	{
//		GemmTester data(23, 45, 67, 't', 't', DataType::FLOAT32);
//		data.gemm_baseline<float, float>(1.1, 0.1);
//
//		math::gemm(DeviceContext(), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//
//	TEST(TestGemmOnCPU, float64_AB)
//	{
//		GemmTester data(23, 45, 67, 'n', 'n', DataType::FLOAT64);
//		data.gemm_baseline<double, double>(1.1, 0.1);
//
//		math::gemm(DeviceContext(), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-6);
//	}
//	TEST(TestGemmOnCPU, float64_ABT)
//	{
//		GemmTester data(23, 45, 67, 'n', 't', DataType::FLOAT64);
//		data.gemm_baseline<double, double>(1.1, 0.1);
//
//		math::gemm(DeviceContext(), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-6);
//	}
//	TEST(TestGemmOnCPU, float64_ATB)
//	{
//		GemmTester data(23, 45, 67, 't', 'n', DataType::FLOAT64);
//		data.gemm_baseline<double, double>(1.1, 0.1);
//
//		math::gemm(DeviceContext(), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-6);
//	}
//	TEST(TestGemmOnCPU, float64_ATBT)
//	{
//		GemmTester data(23, 45, 67, 't', 't', DataType::FLOAT64);
//		data.gemm_baseline<double, double>(1.1, 0.1);
//
//		math::gemm(DeviceContext(), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-6);
//	}
//
//	TEST(TestGemmOnCUDA, float16_AB)
//	{
//		if (Device::numberOfCudaDevices() == 0 || !Device::cuda(0).hasHalfFloatMath())
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 'n', 'n', DataType::FLOAT16);
//		data.gemm_baseline<float16, float16, float>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-3);
//	}
//	TEST(TestGemmOnCUDA, float16_ABT)
//	{
//		if (Device::numberOfCudaDevices() == 0 || !Device::cuda(0).hasHalfFloatMath())
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 'n', 't', DataType::FLOAT16);
//		data.gemm_baseline<float16, float16, float>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 2.0e-2);
//	}
//	TEST(TestGemmOnCUDA, float16_ATB)
//	{
//		if (Device::numberOfCudaDevices() == 0 || !Device::cuda(0).hasHalfFloatMath())
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 't', 'n', DataType::FLOAT16);
//		data.gemm_baseline<float16, float16, float>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 2.0e-3);
//	}
//	TEST(TestGemmOnCUDA, float16_ATBT)
//	{
//		if (Device::numberOfCudaDevices() == 0 || !Device::cuda(0).hasHalfFloatMath())
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 't', 't', DataType::FLOAT16);
//		data.gemm_baseline<float16, float16, float>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-3);
//	}
//
//	TEST(TestGemmOnCUDA, float32_AB)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 'n', 'n', DataType::FLOAT32);
//		data.gemm_baseline<float, float>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//	TEST(TestGemmOnCUDA, float32_ABT)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 'n', 't', DataType::FLOAT32);
//		data.gemm_baseline<float, float>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//	TEST(TestGemmOnCUDA, float32_ATB)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 't', 'n', DataType::FLOAT32);
//		data.gemm_baseline<float, float>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//	TEST(TestGemmOnCUDA, float32_ATBT)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 't', 't', DataType::FLOAT32);
//		data.gemm_baseline<float, float>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//
//	TEST(TestGemmOnCUDA, float64_AB)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 'n', 'n', DataType::FLOAT64);
//		data.gemm_baseline<double, double>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 'n', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//	TEST(TestGemmOnCUDA, float64_ABT)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 'n', 't', DataType::FLOAT64);
//		data.gemm_baseline<double, double>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 'n', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//	TEST(TestGemmOnCUDA, float64_ATB)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 't', 'n', DataType::FLOAT64);
//		data.gemm_baseline<double, double>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 't', 'n', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//	TEST(TestGemmOnCUDA, float64_ATBT)
//	{
//		if (Device::numberOfCudaDevices() == 0)
//			GTEST_SKIP();
//		GemmTester data(23, 29, 37, 't', 't', DataType::FLOAT64);
//		data.gemm_baseline<double, double>(1.1, 0.1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 't', 't', data.C_tested, data.A, data.B, 1.1, 0.1);
//		EXPECT_LT(data.getDifference(), 1.0e-4);
//	}
//
//	TEST(TestGemmOnCUDA, int8_ABT)
//	{
//		if (Device::numberOfCudaDevices() == 0 || !Device::cuda(0).hasDP4A())
//			GTEST_SKIP();
//		GemmTester data(23, 45, 68, 'n', 't', DataType::INT8,  DataType::INT32);
//		data.gemm_baseline<int32_t, int8_t, int32_t>(1, 1);
//
//		data.moveTo(Device::cuda(0));
//		math::gemm(DeviceContext(Device::cuda(0)), 'n', 't', data.C_tested, data.A, data.B, 1, 1);
//		EXPECT_EQ(data.getDifference(), 0.0);
//	}

} /* namespace avocado */

