/*
 * tensor_operations.hpp
 *
 *  Created on: Aug 30, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_TENSOR_OPERATIONS_HPP_
#define AVOCADO_MATH_TENSOR_OPERATIONS_HPP_

#include <Avocado/backend/backend_defs.h>

#include <cstddef>
#include <vector>

namespace avocado
{
	class Tensor;
	class Context;
	class Device;
	class Scalar;
	enum class DataType
	;
	enum class NonlinearityType
	;
}

namespace avocado
{
	enum class TensorBinaryOp
	{
		ADD, /**< The operation to be performed is addition. */
		ADD_SQUARE, /**< The operation to be performed is addition between the first tensor and the square of the second tensor. */
		SUB, /**< The operation to be performed is subtraction. */
		MUL, /**< The operation to be performed is multiplication. */
		DIV, /**< The operation to be performed is division. */
		MOD, /**< The operation to be performed is floating-point remainder of the first tensor's division by the second tensor. */
		POW, /**< The operation to be performed is value from the first tensor to the power of the second tensor. */
		MIN, /**< The operation to be performed is a minimum comparison. */
		MAX, /**< The operation to be performed is a maximum comparison. */
		COMPARE_EQ, /**< The operation to be performed is truth value of the first tensor equal to the second tensor. */
		COMPARE_NEQ, /**< The operation to be performed is truth value of the first tensor not equal to the second tensor. */
		COMPARE_GT, /**< The operation to be performed is truth value of the first tensor greater than to the second tensor. */
		COMPARE_GE, /**< The operation to be performed is truth value of the first tensor greater than equal to the second tensor. */
		COMPARE_LT, /**< The operation to be performed is truth value of the first tensor less than to the second tensor. */
		COMPARE_LE, /**< The operation to be performed is truth value of the first tensor less than equal to the second tensor. */
		LOGICAL_AND, /**< The operation to be performed is truth value of the first tensor logical AND to the second tensor. */
		LOGICAL_OR, /**< The operation to be performed is truth value of the first tensor logical OR to the second tensor. */
		LOGICAL_XOR /**< The operation to be performed is truth value of the first tensor logical XOR to the second tensor. */
	};

	enum class TensorUnaryOp
	{
		ABS, /**< The operation to be performed is absolute value. */
		CEIL, /**< The operation to be performed is ceiling value. */
		COS, /**< The operation to be performed is trigonometric cosine. */
		EXP, /**< The operation to be performed is exponential of a tensor. */
		FLOOR, /**< The operation to be performed is floor value. */
		LN, /**< The operation to be performed is natural logarithm. */
		NEG, /**< The operation to be performed is negation. */
		RCP, /**< The operation to be performed is reciprocal value. */
		RSQRT, /**< The operation to be performed is reciprocal of the square root. */
		SIN, /**< The operation to be performed is trigonometric sine. */
		SQUARE, /**< The operation to be performed is squaring. */
		SQRT, /**< The operation to be performed is square root. */
		TAN, /**< The operation to be performed is trigonometric tangent. */
		LOGICAL_NOT /**< The operation to be performed is logical negation. */
	};

	enum class TensorReduceOp
	{
		ADD, /**< The operation to be performed is addition. */
		MUL, /**< The operation to be performed is multiplication. */
		MIN, /**< The operation to be performed is a minimum comparison. */
		MAX, /**< The operation to be performed is a maximum comparison. */
		AMAX, /**< The operation to be performed is a maximum comparison of absolute values. */
		AVG, /**< The operation to be performed is averaging. */
		NORM1, /**< The operation to be performed is addition of absolute values. */
		NORM2, /**< The operation to be performed is a square root of the sum of squares. */
		MUL_NO_ZEROS, /**< The operation to be performed is multiplication, not including elements of value zero. */
		LOGICAL_OR,
		LOGICAL_AND
	};

	enum class GemmOp
	{
		OP_N, /**< No operation is performed. */
		OP_T, /**< The matrix is transposed. */
		OP_C /**<  */
	};

	namespace math
	{
		void zeroTensor(const Context &context, Tensor &tensor);
		void setTensor(const Context &context, Tensor &tensor, const Scalar &value);
		void copyTensor(const Context &context, Tensor &dst, const Tensor &src);
		void copyTensor(const Context &context, Tensor &dst, const Tensor &src, size_t elements);

		void concatTensors(const Context &context, Tensor &dst, const std::vector<Tensor> &src);
		void splitTensors(const Context &context, std::vector<Tensor> &dst, const Tensor &src);
		void transposeTensor(const Context &context, Tensor &dst, const Tensor &src, std::initializer_list<int> order);

		void scaleTensor(const Context &context, Tensor &dst, Scalar scale);
		void addScalarToTensor(const Context &context, Tensor &dst, Scalar scalar);

		void addTensors(const Context &context, Tensor &dst, const Tensor &src, Scalar alpha, Scalar beta);

		void tensorBinaryOp(const Context &context, TensorBinaryOp operation, Scalar alpha1, const Tensor &src1, Scalar alpha2, const Tensor &src2,
				Scalar beta, Tensor &dst);
		void tensorUnaryOp(const Context &context, TensorUnaryOp operation, Scalar alpha, const Tensor &src, Scalar beta, Tensor &dst);
		void reduceTensor(const Context &context, TensorReduceOp operation, Scalar alpha, Scalar beta, const Tensor &src, Tensor &dst);
		void addBias(const Context &context, Scalar alpha1, Scalar alpha2, const Tensor &input, const Tensor &bias, Scalar beta1, Scalar beta2,
				Scalar beta3, Tensor &output, const Tensor &ext, NonlinearityType activation);

		/**
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		void gemm(const Context &context, GemmOp opA, GemmOp opB, Tensor &C, const Tensor &A, const Tensor &B, Scalar alpha, Scalar beta);
		/**
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		void gemmBatched(const Context &context, GemmOp opA, GemmOp opB, Tensor &C, const Tensor &A, const Tensor &B, Scalar alpha, Scalar beta);

	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_TENSOR_OPERATIONS_HPP_ */
