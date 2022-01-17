/*
 * tensor_operations.hpp
 *
 *  Created on: Aug 30, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_TENSOR_OPERATIONS_HPP_
#define AVOCADO_MATH_TENSOR_OPERATIONS_HPP_

#include <Avocado/backend/backend_defs.h>

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
	namespace math
	{
		enum class TensorOp
		{
			ADD, /**< The operation to be performed is addition. */
			SUB, /**< The operation to be performed is subtraction. */
			MUL, /**< The operation to be performed is multiplication. */
			MIN, /**< The operation to be performed is a minimum comparison. */
			MAX, /**< The operation to be performed is a maximum comparison. */
			SQRT /**< The operation to be performed is square root, performed only on the first input tensor. */
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
			MUL_NO_ZEROS /**< The operation to be performed is multiplication, not including elements of value zero. */
		};

		enum class GemmOp
		{
			OP_N, /**< No operation is performed. */
			OP_T, /**< The matrix is transposed. */
			OP_C /**<  */
		};

		void zeroTensor(const Context &context, Tensor &tensor);
		void setTensor(const Context &context, Tensor &tensor, const Scalar &value);
		void copyTensor(const Context &context, Tensor &dst, const Tensor &src);
		void copyTensor(const Context &context, Tensor &dst, const Tensor &src, size_t elements);

		void concatTensors(const Context &context, Tensor &dst, const std::vector<Tensor> &src);
		void splitTensors(const Context &context, std::vector<Tensor> &dst, const Tensor &src);
		void transposeTensor(const Context &context, Tensor &dst, const Tensor &src, std::initializer_list<int> order);

		void scaleTensor(const Context &context, Tensor &dst, const Scalar src);
		void addScalarToTensor(const Context &context, Tensor &dst, const Scalar src);
		void opTensor(const Context &context, TensorOp operation, const Scalar alpha1, const Tensor &src1, const Scalar alpha2, const Tensor &src2,
				const Scalar beta, Tensor &dst);
		void reduceTensor(const Context &context, TensorReduceOp operation, const Scalar alpha, const Scalar beta, const Tensor &src, Tensor &dst);
		void addTensors(const Context &context, const Scalar alpha, const Scalar beta, const Tensor &src, Tensor &dst, NonlinearityType activation);

		/**
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		void gemm(const Context &context, GemmOp opA, GemmOp opB, Tensor &C, const Tensor &A, const Tensor &B, const Scalar &alpha,
				const Scalar &beta);
		/**
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		void gemmBatched(const Context &context, GemmOp opA, GemmOp opB, Tensor &C, const Tensor &A, const Tensor &B, const Scalar &alpha,
				const Scalar &beta);

	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_TENSOR_OPERATIONS_HPP_ */
