/*
 * tensor_operations.cpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/math/tensor_operations.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>

#include <Avocado/backend/backend_defs.h>
#include <Avocado/backend/backend_libraries.hpp>

namespace avocado
{
	namespace math
	{
		void zeroTensor(const Context &context, Tensor &tensor)
		{
			internal::set_memory(context, tensor.getMemory(), 0, tensor.sizeInBytes(), nullptr, 0);
		}
		void setTensor(const Context &context, Tensor &tensor, const Scalar &value)
		{
			internal::set_memory(context, tensor.getMemory(), 0, tensor.sizeInBytes(), value.data(), value.sizeInBytes());
		}
		void copyTensor(const Context &context, Tensor &dst, const Tensor &src)
		{
			copyTensor(context, dst, src, dst.volume());
		}
		void copyTensor(const Context &context, Tensor &dst, const Tensor &src, size_t elements)
		{
			internal::copy_memory(context, dst.getMemory(), 0, src.getMemory(), 0, elements * sizeOf(src.dtype()));
		}

		void concatTensors(const Context &context, Tensor &dst, const std::vector<Tensor> &src)
		{
		}
		void splitTensors(const Context &context, std::vector<Tensor> &dst, const Tensor &src)
		{
		}
		void transposeTensor(const Context &context, Tensor &dst, const Tensor &src, std::initializer_list<int> order)
		{
		}

		void scaleTensor(const Context &context, Tensor &dst, const Scalar src)
		{
		}
		void addScalarToTensor(const Context &context, Tensor &dst, const Scalar src)
		{
		}
		void opTensor(const Context &context, TensorOp operation, const Scalar alpha1, const Tensor &src1, const Scalar alpha2, const Tensor &src2,
				const Scalar beta, Tensor &dst)
		{
		}
		void reduceTensor(const Context &context, TensorReduceOp operation, const Scalar alpha, const Scalar beta, const Tensor &src, Tensor &dst)
		{
		}
		void addTensors(const Context &context, const Scalar alpha, const Scalar beta, const Tensor &src, Tensor &dst, NonlinearityType activation)
		{
		}

		void gemm(const Context &context, GemmOp opA, GemmOp opB, Tensor &C, const Tensor &A, const Tensor &B, const Scalar &alpha,
				const Scalar &beta)
		{
		}
		void gemmBatched(const Context &context, GemmOp opA, GemmOp opB, Tensor &C, const Tensor &A, const Tensor &B, const Scalar &alpha,
				const Scalar &beta)
		{
		}

	} /* namespace math */
} /* namespace avocado */

