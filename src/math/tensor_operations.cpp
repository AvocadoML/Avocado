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
			if (not same_device(context, dst))
				throw DeviceMismatch(METHOD_NAME, "");
			for (size_t i = 0; i < src.size(); i++)
				if (not same_device(context, src[i]))
					throw DeviceMismatch(METHOD_NAME, "");

			std::vector<backend::avTensorDescriptor_t> srcDesc(src.size());
			std::vector<backend::avMemoryDescriptor_t> srcMem(src.size());
			for (size_t i = 0; i < src.size(); i++)
			{
				srcDesc[i] = src[i].getDescriptor();
				srcMem[i] = src[i].getMemory();
			}

			backend::avTensorDescriptor_t dstDesc = dst.getDescriptor();
			backend::avMemoryDescriptor_t dstMem = dst.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuConcatTensors(context, dstDesc, dstMem, srcDesc.data(), srcMem.data(), src.size());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaConcatTensors(context, dstDesc, dstMem, srcDesc.data(), srcMem.data(), src.size());
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclConcatTensors(context, dstDesc, dstMem, srcDesc.data(), srcMem.data(), src.size());
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void splitTensors(const Context &context, std::vector<Tensor> &dst, const Tensor &src)
		{
			if (not same_device(context, src))
				throw DeviceMismatch(METHOD_NAME, "");
			for (size_t i = 0; i < dst.size(); i++)
				if (not same_device(context, dst[i]))
					throw DeviceMismatch(METHOD_NAME, "");

			std::vector<backend::avTensorDescriptor_t> dstDesc(dst.size());
			std::vector<backend::avMemoryDescriptor_t> dstMem(dst.size());
			for (size_t i = 0; i < dst.size(); i++)
			{
				dstDesc[i] = dst[i].getDescriptor();
				dstMem[i] = dst[i].getMemory();
			}

			backend::avTensorDescriptor_t srcDesc = src.getDescriptor();
			backend::avMemoryDescriptor_t srcMem = src.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuSplitTensors(context, dstDesc.data(), dstMem.data(), srcDesc, srcMem, dst.size());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaSplitTensors(context, dstDesc.data(), dstMem.data(), srcDesc, srcMem, dst.size());
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclSplitTensors(context, dstDesc.data(), dstMem.data(), srcDesc, srcMem, dst.size());
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void transposeTensor(const Context &context, Tensor &dst, const Tensor &src, std::initializer_list<int> order)
		{
			if (not same_device(context, dst, src))
				throw DeviceMismatch(METHOD_NAME, "");

			backend::avTensorDescriptor_t dstDesc = dst.getDescriptor();
			backend::avTensorDescriptor_t srcDesc = src.getDescriptor();

			backend::avMemoryDescriptor_t dstMem = dst.getMemory();
			backend::avMemoryDescriptor_t srcMem = src.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuTranspose(context, dstDesc, dstMem, srcDesc, srcMem, order.begin());
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaTranspose(context, dstDesc, dstMem, srcDesc, srcMem, order.begin());
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclTranspose(context, dstDesc, dstMem, srcDesc, srcMem, order.begin());
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void scaleTensor(const Context &context, Tensor &dst, const Tensor &src, Scalar scale)
		{
			if (not same_device(context, dst))
				throw DeviceMismatch(METHOD_NAME, "");

			backend::avTensorDescriptor_t dstDesc = dst.getDescriptor();
			backend::avMemoryDescriptor_t dstMem = dst.getMemory();
			backend::avTensorDescriptor_t srcDesc = dst.getDescriptor();
			backend::avMemoryDescriptor_t srcMem = dst.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuScaleTensor(context, srcDesc, srcMem, scale.data(), dstDesc, dstMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaScaleTensor(context, srcDesc, srcMem, scale.data(), dstDesc, dstMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclScaleTensor(context, dstDesc, dstMem, scale.data());
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void addScalarToTensor(const Context &context, Tensor &dst, const Tensor &src, Scalar scalar)
		{
			if (not same_device(context, dst))
				throw DeviceMismatch(METHOD_NAME, "");

			backend::avTensorDescriptor_t dstDesc = dst.getDescriptor();
			backend::avMemoryDescriptor_t dstMem = dst.getMemory();
			backend::avTensorDescriptor_t srcDesc = dst.getDescriptor();
			backend::avMemoryDescriptor_t srcMem = dst.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuAddScalarToTensor(context, srcDesc, srcMem, scalar.data(), dstDesc, dstMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaAddScalarToTensor(context, srcDesc, srcMem, scalar.data(), dstDesc, dstMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclAddScalarToTensor(context, dstDesc, dstMem, scalar.data());
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void addTensors(const Context &context, Tensor &dst, const Tensor &src, Scalar alpha, Scalar beta)
		{
			if (not same_device(context, src, dst))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(dst.dtype());
			beta.toScalingTypeFor(dst.dtype());

			backend::avTensorDescriptor_t aDesc = src.getDescriptor();
			backend::avTensorDescriptor_t cDesc = dst.getDescriptor();

			backend::avMemoryDescriptor_t aMem = src.getMemory();
			backend::avMemoryDescriptor_t cMem = dst.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuAddTensors(context, alpha.data(), aDesc, aMem, beta.data(), cDesc, cMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaAddTensors(context, alpha.data(), aDesc, aMem, beta.data(), cDesc, cMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclBinaryOp(context, op, alpha1.data(), aDesc, aMem, alpha2.data(), bDesc, bMem,
//							beta.data(), cDesc, cMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void tensorBinaryOp(const Context &context, TensorBinaryOp operation, Scalar alpha1, const Tensor &src1, Scalar alpha2, const Tensor &src2,
				Scalar beta, Tensor &dst)
		{
			if (not same_device(context, src1, src2, dst))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha1.toScalingTypeFor(dst.dtype());
			alpha2.toScalingTypeFor(dst.dtype());
			beta.toScalingTypeFor(dst.dtype());
			backend::avBinaryOp_t op = static_cast<backend::avBinaryOp_t>(operation);

			backend::avTensorDescriptor_t aDesc = src1.getDescriptor();
			backend::avTensorDescriptor_t bDesc = src2.getDescriptor();
			backend::avTensorDescriptor_t cDesc = dst.getDescriptor();

			backend::avMemoryDescriptor_t aMem = src1.getMemory();
			backend::avMemoryDescriptor_t bMem = src2.getMemory();
			backend::avMemoryDescriptor_t cMem = dst.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuBinaryOp(context, op, alpha1.data(), aDesc, aMem, alpha2.data(), bDesc, bMem,
							beta.data(), cDesc, cMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaBinaryOp(context, op, alpha1.data(), aDesc, aMem, alpha2.data(), bDesc, bMem,
							beta.data(), cDesc, cMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclBinaryOp(context, op, alpha1.data(), aDesc, aMem, alpha2.data(), bDesc, bMem,
//							beta.data(), cDesc, cMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void tensorUnaryOp(const Context &context, TensorUnaryOp operation, Scalar alpha, const Tensor &src, Scalar beta, Tensor &dst)
		{
			if (not same_device(context, src, dst))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(dst.dtype());
			beta.toScalingTypeFor(dst.dtype());
			backend::avUnaryOp_t op = static_cast<backend::avUnaryOp_t>(operation);

			backend::avTensorDescriptor_t aDesc = src.getDescriptor();
			backend::avTensorDescriptor_t cDesc = dst.getDescriptor();

			backend::avMemoryDescriptor_t aMem = src.getMemory();
			backend::avMemoryDescriptor_t cMem = dst.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuUnaryOp(context, op, alpha.data(), aDesc, aMem, beta.data(), cDesc, cMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaUnaryOp(context, op, alpha.data(), aDesc, aMem, beta.data(), cDesc, cMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclUnaryOp(context, op, alpha.data(), aDesc, aMem, beta.data(), cDesc, cMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void reduceTensor(const Context &context, TensorReduceOp operation, Scalar alpha, Scalar beta, const Tensor &src, Tensor &dst)
		{
			if (not same_device(context, src, dst))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(dst.dtype());
			beta.toScalingTypeFor(dst.dtype());
			backend::avReduceOp_t op = static_cast<backend::avReduceOp_t>(operation);

			backend::avTensorDescriptor_t aDesc = src.getDescriptor();
			backend::avTensorDescriptor_t cDesc = dst.getDescriptor();

			backend::avMemoryDescriptor_t aMem = src.getMemory();
			backend::avMemoryDescriptor_t cMem = dst.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuReduceTensor(context, op, alpha.data(), aDesc, aMem, beta.data(), cDesc, cMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaReduceTensor(context, op, alpha.data(), aDesc, aMem, beta.data(), cDesc, cMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclReduceTensor(context, op, alpha.data(), aDesc, aMem, beta.data(), cDesc, cMem);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void addBias(const Context &context, Scalar alpha1, Scalar alpha2, const Tensor &input, const Tensor &bias, Scalar beta1, Scalar beta2,
				Scalar beta3, Tensor &output, const Tensor &ext, NonlinearityType activation)
		{
			if (not same_device(context, input, bias))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha1.toScalingTypeFor(bias.dtype());
			alpha2.toScalingTypeFor(bias.dtype());
			beta1.toScalingTypeFor(bias.dtype());
			beta2.toScalingTypeFor(bias.dtype());
			beta3.toScalingTypeFor(bias.dtype());
			backend::avActivationType_t act = static_cast<backend::avActivationType_t>(activation);

			backend::avTensorDescriptor_t xDesc = input.getDescriptor();
			backend::avTensorDescriptor_t bDesc = bias.getDescriptor();
			backend::avTensorDescriptor_t yDesc = output.getDescriptor();

			backend::avMemoryDescriptor_t xMem = input.getMemory();
			backend::avMemoryDescriptor_t bMem = bias.getMemory();
			backend::avMemoryDescriptor_t yMem = output.getMemory();
			backend::avMemoryDescriptor_t zMem = ext.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuAddBias(context, alpha1.data(), alpha2.data(), xDesc, xMem, bDesc, bMem, yDesc, yMem,
							beta1.data(), beta2.data(), beta3.data(), zMem, act);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaAddBias(context, alpha1.data(), alpha2.data(), xDesc, xMem, bDesc, bMem, yDesc, yMem,
							beta1.data(), beta2.data(), beta3.data(), zMem, act);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
//					backend::avStatus_t status = backend::openclAddBias(context, alpha3.data(), alpha1.data(), aDesc, aMem, alpha2.data(), bDesc,
//							bMem, beta.data(), cDesc, cMem, act);
//					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

		void gemm(const Context &context, GemmOp opA, GemmOp opB, Tensor &C, const Tensor &A, const Tensor &B, Scalar alpha, Scalar beta)
		{
			if (not same_device(context, A, B, C))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(C.dtype());
			beta.toScalingTypeFor(C.dtype());
			backend::avGemmOperation_t operationA = static_cast<backend::avGemmOperation_t>(opA);
			backend::avGemmOperation_t operationB = static_cast<backend::avGemmOperation_t>(opB);

			backend::avTensorDescriptor_t aDesc = A.getDescriptor();
			backend::avTensorDescriptor_t bDesc = B.getDescriptor();
			backend::avTensorDescriptor_t cDesc = C.getDescriptor();

			backend::avMemoryDescriptor_t aMem = A.getMemory();
			backend::avMemoryDescriptor_t bMem = B.getMemory();
			backend::avMemoryDescriptor_t cMem = C.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuGemm(context, operationA, operationB, alpha.data(), aDesc, aMem, bDesc, bMem,
							beta.data(), cDesc, cMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaGemm(context, operationA, operationB, alpha.data(), aDesc, aMem, bDesc, bMem,
							beta.data(), cDesc, cMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
					backend::avStatus_t status = backend::openclGemm(context, operationA, operationB, alpha.data(), aDesc, aMem, bDesc, bMem,
							beta.data(), cDesc, cMem);
					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}
		void gemmBatched(const Context &context, GemmOp opA, GemmOp opB, Tensor &C, const Tensor &A, const Tensor &B, Scalar alpha, Scalar beta)
		{
			if (not same_device(context, A, B, C))
				throw DeviceMismatch(METHOD_NAME, "");

			alpha.toScalingTypeFor(C.dtype());
			beta.toScalingTypeFor(C.dtype());
			backend::avGemmOperation_t operationA = static_cast<backend::avGemmOperation_t>(opA);
			backend::avGemmOperation_t operationB = static_cast<backend::avGemmOperation_t>(opB);

			backend::avTensorDescriptor_t aDesc = A.getDescriptor();
			backend::avTensorDescriptor_t bDesc = B.getDescriptor();
			backend::avTensorDescriptor_t cDesc = C.getDescriptor();

			backend::avMemoryDescriptor_t aMem = A.getMemory();
			backend::avMemoryDescriptor_t bMem = B.getMemory();
			backend::avMemoryDescriptor_t cMem = C.getMemory();

			switch (context.device().type())
			{
				case DeviceType::CPU:
				{
					backend::avStatus_t status = backend::cpuGemmBatched(context, operationA, operationB, alpha.data(), aDesc, aMem, bDesc, bMem,
							beta.data(), cDesc, cMem);
					CHECK_CPU_STATUS(status)
					break;
				}
				case DeviceType::CUDA:
				{
					backend::avStatus_t status = backend::cudaGemmBatched(context, operationA, operationB, alpha.data(), aDesc, aMem, bDesc, bMem,
							beta.data(), cDesc, cMem);
					CHECK_CUDA_STATUS(status)
					break;
				}
				case DeviceType::OPENCL:
				{
					backend::avStatus_t status = backend::openclGemmBatched(context, operationA, operationB, alpha.data(), aDesc, aMem, bDesc, bMem,
							beta.data(), cDesc, cMem);
					CHECK_OPENCL_STATUS(status)
					break;
				}
			}
		}

	} /* namespace math */
} /* namespace avocado */

