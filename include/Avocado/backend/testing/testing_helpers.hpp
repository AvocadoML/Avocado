/*
 * testing_helpers.hpp
 *
 *  Created on: Sep 13, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_UTILS_TESTING_HELPERS_HPP_
#define AVOCADO_UTILS_TESTING_HELPERS_HPP_

#include "../backend_defs.h"
#include "wrappers.hpp"

#include <string>
#include <gtest/gtest.h>

namespace avocado
{
	namespace backend
	{
		bool supportsType(avDataType_t dtype);
		bool isDeviceAvailable(const std::string &str);

		void initForTest(TensorWrapper &t, double offset, double shift = 0.0);
		double diffForTest(const TensorWrapper &lhs, const TensorWrapper &rhs);
		double normForTest(const TensorWrapper &tensor);
		void absForTest(TensorWrapper &tensor);

		class ActivationTester
		{
			private:
				avDeviceIndex_t device_index;
				avActivationType_t act;
				TensorWrapper input;
				TensorWrapper gradientOut;
				TensorWrapper output_baseline;
				TensorWrapper output_tested;
				TensorWrapper gradientIn_baseline;
				TensorWrapper gradientIn_tested;
			public:
				ActivationTester(avDeviceIndex_t idx, avActivationType_t activation, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifferenceForward(const void *alpha, const void *beta) noexcept;
				double getDifferenceBackward(const void *alpha, const void *beta) noexcept;
		};

		class SoftmaxTester
		{
			private:
				avDeviceIndex_t device_index;
				avSoftmaxMode_t mode;
				TensorWrapper input;
				TensorWrapper gradientOut;
				TensorWrapper output_baseline;
				TensorWrapper output_tested;
				TensorWrapper gradientIn_baseline;
				TensorWrapper gradientIn_tested;
			public:
				SoftmaxTester(avDeviceIndex_t idx, avSoftmaxMode_t mode, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifferenceForward(const void *alpha, const void *beta) noexcept;
				double getDifferenceBackward(const void *alpha, const void *beta) noexcept;
		};

		class GemmTester
		{
			private:
				avDeviceIndex_t device_index;
				avGemmOperation_t op_A, op_B;
				TensorWrapper A;
				TensorWrapper B;
				TensorWrapper C_baseline;
				TensorWrapper C_tested;
			public:
				GemmTester(avDeviceIndex_t idx, int M, int N, int K, avGemmOperation_t opA, avGemmOperation_t opB, avDataType_t C_type,
						avDataType_t AB_type);
				GemmTester(avDeviceIndex_t idx, int M, int N, int K, avGemmOperation_t opA, avGemmOperation_t opB, avDataType_t dtype);
				double getDifference(const void *alpha, const void *beta) noexcept;
		};

		class ConcatTester
		{
			private:
				avDeviceIndex_t device_index;
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				ConcatTester(avDeviceIndex_t idx, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference() noexcept;
		};
		class SplitTester
		{
			private:
				avDeviceIndex_t device_index;
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				SplitTester(avDeviceIndex_t idx, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference() noexcept;
		};
		class TransposeTester
		{
			private:
				avDeviceIndex_t device_index;
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				TransposeTester(avDeviceIndex_t idx, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference(const std::vector<int> &ordering) noexcept;
		};

		class UnaryOpTester
		{
			private:
				avDeviceIndex_t device_index;
				avUnaryOp_t op;
				TensorWrapper input;
				TensorWrapper output_baseline;
				TensorWrapper output_tested;
			public:
				UnaryOpTester(avDeviceIndex_t idx, avUnaryOp_t operation, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference(const void *alpha, const void *beta) noexcept;
		};

		class BinaryOpTester
		{
			private:
				avDeviceIndex_t device_index;
				avBinaryOp_t op;
				TensorWrapper input;
				TensorWrapper input_same;
				TensorWrapper input_1d;
				TensorWrapper input_single;
				TensorWrapper output_baseline;
				TensorWrapper output_tested;
			public:
				BinaryOpTester(avDeviceIndex_t idx, avBinaryOp_t operation, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifferenceSame(const void *alpha1, const void *alpha2, const void *beta) noexcept;
				double getDifference1D(const void *alpha1, const void *alpha2, const void *beta) noexcept;
				double getDifferenceSingle(const void *alpha1, const void *alpha2, const void *beta) noexcept;
		};

		class ReductionTester
		{
			private:
				avDeviceIndex_t device_index;
				avReduceOp_t op;
				TensorWrapper input;
				TensorWrapper output_baseline_1d;
				TensorWrapper output_tested_1d;
				TensorWrapper output_baseline_single;
				TensorWrapper output_tested_single;
			public:
				ReductionTester(avDeviceIndex_t idx, avReduceOp_t operation, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference1D(const void *alpha, const void *beta) noexcept;
				double getDifferenceSingle(const void *alpha, const void *beta) noexcept;
		};

		class BatchNormTester
		{
			private:
				avDeviceIndex_t device_index;
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				BatchNormTester(avDeviceIndex_t idx, std::vector<int> shape, avDataType_t dtype);
				double getDifferenceInference(const void *alpha, const void *beta) noexcept;
				double getDifferenceForward(const void *alpha, const void *beta) noexcept;
				double getDifferenceBackward(const void *alpha, const void *beta) noexcept;
		};

	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_UTILS_TESTING_HELPERS_HPP_ */
