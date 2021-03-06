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

namespace avocado
{
	namespace backend
	{
		void setMasterContext(avDeviceIndex_t deviceIndex, bool useDefault);
		const ContextWrapper& getMasterContext();

		bool supportsType(avDataType_t dtype);
		bool isDeviceAvailable(const std::string &str);

		void initForTest(TensorWrapper &t, double offset, double minValue = -1.0, double maxValue = 1.0);
		double diffForTest(const TensorWrapper &lhs, const TensorWrapper &rhs);
		double normForTest(const TensorWrapper &tensor);
		void absForTest(TensorWrapper &tensor);

		double epsilonForTest(avDataType_t dtype);

		class ActivationTester
		{
			private:
				avActivationType_t act;
				TensorWrapper input;
				TensorWrapper gradientOut;
				TensorWrapper output_baseline;
				TensorWrapper output_tested;
				TensorWrapper gradientIn_baseline;
				TensorWrapper gradientIn_tested;
			public:
				ActivationTester(avActivationType_t activation, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifferenceForward(const void *alpha, const void *beta) noexcept;
				double getDifferenceBackward(const void *alpha, const void *beta) noexcept;
		};

		class SoftmaxTester
		{
			private:
				avSoftmaxMode_t mode;
				TensorWrapper input;
				TensorWrapper gradientOut;
				TensorWrapper output_baseline;
				TensorWrapper output_tested;
				TensorWrapper gradientIn_baseline;
				TensorWrapper gradientIn_tested;
			public:
				SoftmaxTester(avSoftmaxMode_t mode, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifferenceForward(const void *alpha, const void *beta) noexcept;
				double getDifferenceBackward(const void *alpha, const void *beta) noexcept;
		};

		class GemmTester
		{
			private:
				avGemmOperation_t op_A, op_B;
				TensorWrapper A;
				TensorWrapper B;
				TensorWrapper C_baseline;
				TensorWrapper C_tested;
			public:
				GemmTester(int M, int N, int K, avGemmOperation_t opA, avGemmOperation_t opB, avDataType_t C_type, avDataType_t AB_type);
				GemmTester(int M, int N, int K, avGemmOperation_t opA, avGemmOperation_t opB, avDataType_t dtype);
				double getDifference(const void *alpha, const void *beta) noexcept;
		};

		class ConcatTester
		{
			private:
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				ConcatTester(std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference() noexcept;
		};
		class SplitTester
		{
			private:
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				SplitTester(std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference() noexcept;
		};
		class TransposeTester
		{
			private:
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				TransposeTester(std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference(const std::vector<int> &ordering) noexcept;
		};
		class ScaleTester
		{
			private:
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				ScaleTester(std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference(const void *alpha) noexcept;
		};
		class AddScalarTester
		{
			private:
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				AddScalarTester(std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference(const void *scalar) noexcept;
		};
		class AddTensorsTester
		{
			private:
				std::vector<int> shape;
				avDataType_t input_dtype;
				avDataType_t output_dtype;
			public:
				AddTensorsTester(std::initializer_list<int> shape, avDataType_t dtype);
				AddTensorsTester(std::initializer_list<int> shape, avDataType_t input_dtype, avDataType_t output_dtype);
				double getDifference(const void *alpha, const void *beta) noexcept;
		};
		class AddBiasTester
		{
			private:
				std::vector<int> shape;
				avDataType_t input_dtype;
				avDataType_t output_dtype;
				avDataType_t bias_dtype;
			public:
				AddBiasTester(std::initializer_list<int> shape, avDataType_t dtype);
				AddBiasTester(std::initializer_list<int> shape, avDataType_t input_dtype, avDataType_t output_dtype, avDataType_t bias_dtype);
				double getDifference(const void *alpha1, const void *alpha2, const void *beta1, const void *beta2, const void *beta3) noexcept;
		};

		class UnaryOpTester
		{
			private:
				avUnaryOp_t op;
				TensorWrapper input;
				TensorWrapper output_baseline;
				TensorWrapper output_tested;
			public:
				UnaryOpTester(avUnaryOp_t operation, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference(const void *alpha, const void *beta) noexcept;
		};

		class BinaryOpTester
		{
			private:
				avBinaryOp_t op;
				TensorWrapper input;
				TensorWrapper input_same;
				TensorWrapper input_1d;
				TensorWrapper input_single;
				TensorWrapper output_baseline;
				TensorWrapper output_tested;
			public:
				BinaryOpTester(avBinaryOp_t operation, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifferenceSame(const void *alpha1, const void *alpha2, const void *beta) noexcept;
				double getDifference1D(const void *alpha1, const void *alpha2, const void *beta) noexcept;
				double getDifferenceSingle(const void *alpha1, const void *alpha2, const void *beta) noexcept;
		};

		class ReductionTester
		{
			private:
				avReduceOp_t op;
				TensorWrapper input;
				TensorWrapper output_baseline_1d;
				TensorWrapper output_tested_1d;
				TensorWrapper output_baseline_single;
				TensorWrapper output_tested_single;
			public:
				ReductionTester(avReduceOp_t operation, std::initializer_list<int> shape, avDataType_t dtype);
				double getDifference1D(const void *alpha, const void *beta) noexcept;
				double getDifferenceSingle(const void *alpha, const void *beta) noexcept;
		};

		class BatchNormTester
		{
			private:
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				BatchNormTester(std::vector<int> shape, avDataType_t dtype);
				double getDifferenceInference(const void *alpha, const void *beta) noexcept;
				double getDifferenceForward(const void *alpha, const void *beta) noexcept;
				double getDifferenceBackward(const void *alpha, const void *beta, const void *alpha2, const void *beta2) noexcept;
		};

		class LossFunctionTester
		{
			private:
				avLossType_t loss_type;
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				LossFunctionTester(avLossType_t loss_type, std::vector<int> shape, avDataType_t dtype);
				double getDifferenceLoss() noexcept;
				double getDifferenceGradient(const void *alpha, const void *beta, bool isFused) noexcept;
		};

		class OptimizerTester
		{
			private:
				OptimizerWrapper optimizer;
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				OptimizerTester(std::vector<int> shape, avDataType_t dtype);
				void set(avOptimizerType_t type, int64_t steps, double learningRate, const std::array<double, 4> &coefficients, const std::array<bool, 4> &flags);
				double getDifference(const void *alpha, const void *beta) noexcept;
		};

		class RegularizerTest
		{
			private:
				std::vector<int> shape;
				avDataType_t dtype;
			public:
				RegularizerTest(std::vector<int> shape, avDataType_t dtype);
				double getDifference(const void *scale, const void *offset) noexcept;
		};

		class Im2rowTest
		{
			private:
				ConvolutionWrapper config;
				std::vector<int> input_shape;
				std::vector<int> filter_shape;
				avDataType_t dtype;
			public:
				Im2rowTest(std::vector<int> inputShape, std::vector<int> filterShape, avDataType_t dtype);
				void set(avConvolutionMode_t mode, const std::array<int, 3> &padding, const std::array<int, 3> &strides,
						const std::array<int, 3> &dilation, int groups, const void *paddingValue);
				double getDifference() noexcept;
		};

		class WinogradTest
		{
			private:
				ConvolutionWrapper config;
				std::vector<int> input_shape;
				std::vector<int> filter_shape;
				avDataType_t dtype;
				int transform_size;
			public:
				WinogradTest(std::vector<int> inputShape, std::vector<int> filterShape, avDataType_t dtype, int transformSize);
				void set(avConvolutionMode_t mode, const std::array<int, 3> &strides, const std::array<int, 3> &padding, int groups,
						const void *paddingValue);
				double getDifferenceWeight() noexcept;
				double getDifferenceInput() noexcept;
				double getDifferenceOutput(const void *alpha1, const void *alpha2, const void *beta, bool useBias, bool useExt) noexcept;
				double getDifferenceGradient() noexcept;
				double getDifferenceUpdate(const void *alpha, const void *beta) noexcept;
		};

		class ConvolutionTest
		{
			private:
				ConvolutionWrapper config;
				std::vector<int> input_shape;
				std::vector<int> filter_shape;
				avDataType_t dtype;
			public:
				ConvolutionTest(std::vector<int> inputShape, std::vector<int> filterShape, avDataType_t dtype);
				void set(avConvolutionMode_t mode, const std::array<int, 3> &strides, const std::array<int, 3> &padding,
						const std::array<int, 3> &dilation, int groups, const void *paddingValue);
				double getDifferenceInference(const void *alpha, const void *beta) noexcept;
				double getDifferenceForward(const void *alpha, const void *beta) noexcept;
				double getDifferenceBackward(const void *alpha, const void *beta) noexcept;
				double getDifferenceUpdate(const void *alpha, const void *beta) noexcept;
		};

	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_UTILS_TESTING_HELPERS_HPP_ */
