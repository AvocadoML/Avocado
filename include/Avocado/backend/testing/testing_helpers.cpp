/*
 * testing_helpers.cpp
 *
 *  Created on: Sep 13, 2020
 *      Author: Maciej Kozarzewski
 */

#include "testing_helpers.hpp"
#include "../backend_defs.h"
#include "../backend_descriptors.hpp"
#if USE_CPU
#  include <CpuBackend/cpu_backend.h>
#elif USE_CUDA
#  include <CudaBackend/cuda_backend.h>
#elif USE_OPENCL
#  include <OpenCLBackend/opencl_backend.h>
#endif
#include <ReferenceBackend/reference_backend.h>

#include <complex>
#include <memory>
#include <iostream>
#include <cassert>
#include "../src/vectors/simd_utils.hpp"

namespace
{
	using namespace avocado;
	using namespace avocado::backend;

	template<typename T>
	void init_for_test_float(void *ptr, size_t elements, T offset, T shift)
	{
		for (size_t i = 0; i < elements; i++)
			reinterpret_cast<T*>(ptr)[i] = shift + sin(i / 10.0f + offset);
	}
	template<typename T>
	void init_for_test_int(void *ptr, size_t elements, T offset, T shift)
	{
		for (size_t i = 0; i < elements; i++)
			reinterpret_cast<T*>(ptr)[i] = shift + (17 * (i + offset)) % 97 - 49;
	}

	template<typename T>
	float diff_for_test(const void *ptr1, const void *ptr2, size_t elements)
	{
		double result = 0.0;
		for (size_t i = 0; i < elements; i++)
			result += fabs(reinterpret_cast<const T*>(ptr1)[i] - reinterpret_cast<const T*>(ptr2)[i]);
		return result / elements;
	}

	template<typename T>
	float norm_for_test(const void *ptr, size_t elements)
	{
		double result = 0.0;
		for (size_t i = 0; i < elements; i++)
			result += fabs(reinterpret_cast<const T*>(ptr)[i]);
		return result;
	}

	template<typename T>
	void abs_for_test(void *ptr, size_t elements)
	{
		for (size_t i = 0; i < elements; i++)
			reinterpret_cast<T*>(ptr)[i] = fabs(reinterpret_cast<T*>(ptr)[i]);
	}

//	template<typename T>
//	TensorWrapper toTensor(std::initializer_list<T> data, avDeviceIndex_t deviceIdx)
//	{
//#if USE_CPU
//		TensorWrapper result( { static_cast<int>(data.size()) }, cpu::typeOf<T>(), deviceIdx);
//#elif USE_CUDA
//		TensorWrapper result( { static_cast<int>(data.size()) }, cuda::typeOf<T>(), deviceIdx);
//#elif USE_OPENCL
//		TensorWrapper result( { static_cast<int>(data.size()) }, opencl::typeOf<T>(), deviceIdx);
//#endif
//		result.copyFromHost(data.begin());
//		return result;
//	}
//	template<typename T>
//	TensorWrapper toTensor(std::initializer_list<std::initializer_list<T>> data)
//	{
//		Shape shape( { static_cast<int>(data.size()), static_cast<int>((data.begin()[0]).size()) });
//		std::unique_ptr<T[]> tmp = std::make_unique<T[]>(shape.volume());
//		for (int i = 0; i < shape[0]; i++)
//		{
//			assert(shape.lastDim() == static_cast<int>((data.begin()[i]).size()));
//			std::memcpy(tmp.get() + i * shape.lastDim(), (data.begin()[i]).begin(), sizeof(T) * shape.lastDim());
//		}
//
//		Tensor result(shape, typeOf<T>(), Device::cpu());
//		result.copyFromHost(tmp.get(), result.volume());
//		return result;
//	}
	template<typename T>
	std::unique_ptr<T[]> toArray(const TensorWrapper &t)
	{
		std::unique_ptr<T[]> result = std::make_unique<T[]>(t.volume());
		t.copyToHost(result.get());
		return result;
	}
	template<typename T>
	void fromArray(TensorWrapper &dst, const std::unique_ptr<T[]> &src)
	{
		dst.copyFromHost(src.get());
	}

	std::ostream& operator<<(std::ostream &stream, bfloat16 x)
	{
		float tmp;
		refChangeTypeHost(0ll, &tmp, AVOCADO_DTYPE_FLOAT32, &x, AVOCADO_DTYPE_BFLOAT16, 1);
		stream << tmp;
		return stream;
	}
	std::ostream& operator<<(std::ostream &stream, float16 x)
	{
		float tmp;
		refChangeTypeHost(0ll, &tmp, AVOCADO_DTYPE_FLOAT32, &x, AVOCADO_DTYPE_FLOAT16, 1);
		stream << tmp;
		return stream;
	}
}

namespace avocado
{
	namespace backend
	{
		bool supportsType(avDataType_t dtype)
		{
			switch (dtype)
			{
				default:
				case AVOCADO_DTYPE_UNKNOWN:
					return false;
				case AVOCADO_DTYPE_UINT8:
				case AVOCADO_DTYPE_INT8:
				case AVOCADO_DTYPE_INT16:
				case AVOCADO_DTYPE_INT32:
				case AVOCADO_DTYPE_INT64:
					return true;
				case AVOCADO_DTYPE_FLOAT16:
				{
					bool result;
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, &result);
					return result;
				}
				case AVOCADO_DTYPE_BFLOAT16:
				{
					bool result;
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_BFLOAT16, &result);
					return result;
				}
				case AVOCADO_DTYPE_FLOAT32:
				case AVOCADO_DTYPE_COMPLEX32:
				{
					bool result;
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION, &result);
					return result;
				}
				case AVOCADO_DTYPE_FLOAT64:
				case AVOCADO_DTYPE_COMPLEX64:
				{
					bool result;
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION, &result);
					return result;
				}
			}
		}

		bool isDeviceAvailable(const std::string &str)
		{
#if USE_CPU
			if (str == "CPU" || str == "cpu")
				return true;
#elif USE_CUDA
			if (str.substr(0, 5) == "CUDA:" || str.substr(0, 5) == "cuda:")
			{
				int idx = std::atoi(str.data() + 5);
				return idx >= 0 and idx < cudaGetNumberOfDevices();
			}
#elif USE_OPENCL
			if (str.substr(0, 7) == "OPENCL:" || str.substr(0, 7) == "opencl:")
			{
				int idx = std::atoi(str.data() + 7);
				return idx >= 0 and idx < openclGetNumberOfDevices();
			}
#endif
			return false;
		}

		void initForTest(TensorWrapper &t, double offset, double shift)
		{
			std::unique_ptr<char[]> tmp = std::make_unique<char[]>(t.sizeInBytes());
			switch (t.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					init_for_test_int<uint8_t>(tmp.get(), t.volume(), offset, shift);
					break;
				case AVOCADO_DTYPE_INT8:
					init_for_test_int<int8_t>(tmp.get(), t.volume(), offset, shift);
					break;
				case AVOCADO_DTYPE_INT16:
					init_for_test_int<int16_t>(tmp.get(), t.volume(), offset, shift);
					break;
				case AVOCADO_DTYPE_INT32:
					init_for_test_int<int32_t>(tmp.get(), t.volume(), offset, shift);
					break;
				case AVOCADO_DTYPE_INT64:
					init_for_test_int<int64_t>(tmp.get(), t.volume(), offset, shift);
					break;
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2 = std::make_unique<float[]>(t.volume());
					init_for_test_float<float>(tmp2.get(), t.volume(), offset, shift);
					refChangeTypeHost(0ll, tmp.get(), static_cast<avDataType_t>(t.dtype()), tmp2.get(), AVOCADO_DTYPE_FLOAT32, t.volume());
					break;
				}
				case AVOCADO_DTYPE_FLOAT32:
					init_for_test_float<float>(tmp.get(), t.volume(), offset, shift);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					init_for_test_float<double>(tmp.get(), t.volume(), offset, shift);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					init_for_test_float<float>(tmp.get(), 2 * t.volume(), offset, shift);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					init_for_test_float<double>(tmp.get(), 2 * t.volume(), offset, shift);
					break;
				default:
					throw std::logic_error("initForTest() : unknown datatype '" + std::to_string(t.dtype()) + "'");
			}
			t.copyFromHost(tmp.get());
		}
		double diffForTest(const TensorWrapper &lhs, const TensorWrapper &rhs)
		{
			assert(lhs.volume() == rhs.volume());
			assert(lhs.dtype() == rhs.dtype());

			if (lhs.volume() == 0)
				return 0.0;

			std::unique_ptr<char[]> tmp_lhs = std::make_unique<char[]>(lhs.sizeInBytes());
			std::unique_ptr<char[]> tmp_rhs = std::make_unique<char[]>(rhs.sizeInBytes());
			lhs.copyToHost(tmp_lhs.get());
			rhs.copyToHost(tmp_rhs.get());
			switch (lhs.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					return diff_for_test<uint8_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_INT8:
					return diff_for_test<int8_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_INT16:
					return diff_for_test<int16_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_INT32:
					return diff_for_test<int32_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_INT64:
					return diff_for_test<int64_t>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2_lhs = std::make_unique<float[]>(lhs.volume());
					std::unique_ptr<float[]> tmp2_rhs = std::make_unique<float[]>(rhs.volume());
					refChangeTypeHost(0ll, tmp2_lhs.get(), AVOCADO_DTYPE_FLOAT32, tmp_lhs.get(), static_cast<avDataType_t>(lhs.dtype()),
							lhs.volume());
					refChangeTypeHost(0ll, tmp2_rhs.get(), AVOCADO_DTYPE_FLOAT32, tmp_rhs.get(), static_cast<avDataType_t>(rhs.dtype()),
							rhs.volume());
					return diff_for_test<float>(tmp2_lhs.get(), tmp2_rhs.get(), lhs.volume());
				}
				case AVOCADO_DTYPE_FLOAT32:
					return diff_for_test<float>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_FLOAT64:
					return diff_for_test<double>(tmp_lhs.get(), tmp_rhs.get(), lhs.volume());
				case AVOCADO_DTYPE_COMPLEX32:
					return diff_for_test<float>(tmp_lhs.get(), tmp_rhs.get(), 2 * lhs.volume());
				case AVOCADO_DTYPE_COMPLEX64:
					return diff_for_test<double>(tmp_lhs.get(), tmp_rhs.get(), 2 * lhs.volume());
				default:
					throw std::logic_error("diffForTest() : unknown datatype '" + std::to_string(lhs.dtype()) + "'");
			}
		}
		double normForTest(const TensorWrapper &tensor)
		{
			std::unique_ptr<char[]> tmp = std::make_unique<char[]>(tensor.sizeInBytes());
			tensor.copyToHost(tmp.get());
			switch (tensor.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					return norm_for_test<uint8_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_INT8:
					return norm_for_test<int8_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_INT16:
					return norm_for_test<int16_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_INT32:
					return norm_for_test<int32_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_INT64:
					return norm_for_test<int64_t>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2 = std::make_unique<float[]>(tensor.volume());
					refChangeTypeHost(0ll, tmp2.get(), AVOCADO_DTYPE_FLOAT32, tmp.get(), static_cast<avDataType_t>(tensor.dtype()), tensor.volume());
					return norm_for_test<float>(tmp2.get(), tensor.volume());
				}
				case AVOCADO_DTYPE_FLOAT32:
					return norm_for_test<float>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_FLOAT64:
					return norm_for_test<double>(tmp.get(), tensor.volume());
				case AVOCADO_DTYPE_COMPLEX32:
					return norm_for_test<float>(tmp.get(), 2 * tensor.volume());
				case AVOCADO_DTYPE_COMPLEX64:
					return norm_for_test<double>(tmp.get(), 2 * tensor.volume());
				default:
					throw std::logic_error("normForTest() : unknown datatype '" + std::to_string(tensor.dtype()) + "'");
			}
		}
		void absForTest(TensorWrapper &tensor)
		{
			std::unique_ptr<char[]> tmp = std::make_unique<char[]>(tensor.sizeInBytes());
			tensor.copyToHost(tmp.get());
			switch (tensor.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					abs_for_test<uint8_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_INT8:
					abs_for_test<int8_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_INT16:
					abs_for_test<int16_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_INT32:
					abs_for_test<int32_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_INT64:
					abs_for_test<int64_t>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2 = std::make_unique<float[]>(tensor.volume());
					refChangeTypeHost(0ll, tmp2.get(), AVOCADO_DTYPE_FLOAT32, tmp.get(), static_cast<avDataType_t>(tensor.dtype()), tensor.volume());
					return abs_for_test<float>(tmp2.get(), tensor.volume());
				}
				case AVOCADO_DTYPE_FLOAT32:
					abs_for_test<float>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_FLOAT64:
					abs_for_test<double>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					abs_for_test<float>(tmp.get(), tensor.volume());
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					abs_for_test<double>(tmp.get(), tensor.volume());
					break;
				default:
					throw std::logic_error("absForTest() : unknown datatype '" + std::to_string(tensor.dtype()) + "'");
			}
			tensor.copyFromHost(tmp.get());
		}
		template<typename T = float>
		void printForTest(const TensorWrapper &tensor)
		{
			if (tensor.numberOfDimensions() == 1)
			{
				std::cout << "------------------------------------------------------------\n";
				for (int i = 0; i < tensor.volume(); i++)
					std::cout << tensor.get<T>( { i }) << ' ';
				std::cout << '\n';
				std::cout << "------------------------------------------------------------\n";
			}
			if (tensor.numberOfDimensions() == 2)
			{
				std::cout << "------------------------------------------------------------\n";
				for (int i = 0; i < tensor.firstDim(); i++)
				{
					for (int j = 0; j < tensor.lastDim(); j++)
						std::cout << tensor.get<T>( { i, j }) << ' ';
					std::cout << '\n';
				}
				std::cout << "------------------------------------------------------------\n";
			}
		}

		ActivationTester::ActivationTester(avDeviceIndex_t idx, avActivationType_t activation, std::initializer_list<int> shape, avDataType_t dtype) :
				device_index(idx),
				act(activation),
				input(shape, dtype, idx),
				gradientOut(shape, dtype, idx),
				output_baseline(shape, dtype, idx),
				output_tested(shape, dtype, idx),
				gradientIn_baseline(shape, dtype, idx),
				gradientIn_tested(shape, dtype, idx)
		{
			initForTest(input, 0.0);
			initForTest(gradientOut, 1.0);
		}
		double ActivationTester::getDifferenceForward(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			refActivationForward(0, act, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory());
#if USE_CPU
			cpuActivationForward(cpuGetDefaultContext(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_CUDA
			cudaActivationForward(cudaGetDefaultContext(device_index), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_OPENCL
			openclActivationForward(openclGetDefaultContext(device_index), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#endif
			return diffForTest(output_baseline, output_tested);
		}
		double ActivationTester::getDifferenceBackward(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			initForTest(gradientIn_baseline, 0.2);
			initForTest(gradientIn_tested, 0.2);

			refActivationForward(0, act, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory());
			refActivationBackward(0, act, alpha, output_baseline.getRefDescriptor(), output_baseline.getRefMemory(), gradientOut.getRefDescriptor(),
					gradientOut.getRefMemory(), beta, gradientIn_baseline.getRefDescriptor(), gradientIn_baseline.getRefMemory());

#if USE_CPU
			cpuActivationForward(cpuGetDefaultContext(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			cpuActivationBackward(0, act, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#elif USE_CUDA
			cudaActivationForward(cudaGetDefaultContext(device_index), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			cudaActivationBackward(cudaGetDefaultContext(device_index), act, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#elif USE_OPENCL
			openclActivationForward(openclGetDefaultContext(device_index), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			openclActivationBackward(openclGetDefaultContext(device_index), act, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#endif
			return diffForTest(gradientIn_baseline, gradientIn_tested);
		}

		SoftmaxTester::SoftmaxTester(avDeviceIndex_t idx, avSoftmaxMode_t mode, std::initializer_list<int> shape, avDataType_t dtype) :
				device_index(idx),
				mode(mode),
				input(shape, dtype, idx),
				gradientOut(shape, dtype, idx),
				output_baseline(shape, dtype, idx),
				output_tested(shape, dtype, idx),
				gradientIn_baseline(shape, dtype, idx),
				gradientIn_tested(shape, dtype, idx)
		{
			initForTest(input, 0.0);
			initForTest(gradientOut, 1.0);
		}
		double SoftmaxTester::getDifferenceForward(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			refSoftmaxForward(0, mode, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory());
#if USE_CPU
			cpuSoftmaxForward(cpuGetDefaultContext(), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_CUDA
			cudaSoftmaxForward(cudaGetDefaultContext(device_index), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_OPENCL
			openclSoftmaxForward(openclGetDefaultContext(device_index), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#endif
			return diffForTest(output_baseline, output_tested);
		}
		double SoftmaxTester::getDifferenceBackward(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			initForTest(gradientIn_baseline, 0.2);
			initForTest(gradientIn_tested, 0.2);

			refSoftmaxForward(0, mode, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory());
			refSoftmaxBackward(0, mode, alpha, output_baseline.getRefDescriptor(), output_baseline.getRefMemory(), gradientOut.getRefDescriptor(),
					gradientOut.getRefMemory(), beta, gradientIn_baseline.getRefDescriptor(), gradientIn_baseline.getRefMemory());

#if USE_CPU
			cpuSoftmaxForward(cpuGetDefaultContext(), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			cpuSoftmaxBackward(0, mode, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#elif USE_CUDA
			cudaSoftmaxForward(cudaGetDefaultContext(device_index), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			cudaSoftmaxBackward(cudaGetDefaultContext(device_index), mode, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#elif USE_OPENCL
			openclSoftmaxForward(openclGetDefaultContext(device_index), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			openclSoftmaxBackward(openclGetDefaultContext(device_index), mode, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#endif
			return diffForTest(gradientIn_baseline, gradientIn_tested);
		}

		GemmTester::GemmTester(avDeviceIndex_t idx, int M, int N, int K, avGemmOperation_t opA, avGemmOperation_t opB, avDataType_t C_type,
				avDataType_t AB_type) :
				device_index(idx),
				op_A(opA),
				op_B(opB)
		{
			if (opA == AVOCADO_GEMM_OPERATION_N)
				A = TensorWrapper( { M, K }, AB_type, device_index);
			else
				A = TensorWrapper( { K, M }, AB_type, device_index);
			if (opB == AVOCADO_GEMM_OPERATION_N)
				B = TensorWrapper( { K, N }, AB_type, device_index);
			else
				B = TensorWrapper( { N, K }, AB_type, device_index);

			initForTest(A, 0.0);
			initForTest(B, 1.57);

			C_baseline = TensorWrapper( { M, N }, C_type, device_index);
			C_tested = TensorWrapper( { M, N }, C_type, device_index);
			initForTest(C_baseline, 0.1);
			initForTest(C_tested, 0.1);
		}
		GemmTester::GemmTester(avDeviceIndex_t idx, int M, int N, int K, avGemmOperation_t opA, avGemmOperation_t opB, avDataType_t dtype) :
				GemmTester(idx, M, N, K, opA, opB, dtype, dtype)
		{
		}
		double GemmTester::getDifference(const void *alpha, const void *beta) noexcept
		{
			refGemm(0, op_A, op_B, alpha, A.getRefDescriptor(), A.getRefMemory(), B.getRefDescriptor(), B.getRefMemory(), beta,
					C_baseline.getRefDescriptor(), C_baseline.getRefMemory());
#if USE_CPU
			cpuGemm(cpuGetDefaultContext(), op_A, op_B, alpha, A.getDescriptor(), A.getMemory(), B.getDescriptor(), B.getMemory(), beta,
					C_tested.getDescriptor(), C_tested.getMemory());
#elif USE_CUDA
			cudaGemm(cudaGetDefaultContext(device_index), op_A, op_B, alpha, A.getDescriptor(), A.getMemory(), B.getDescriptor(), B.getMemory(), beta,
								C_tested.getDescriptor(), C_tested.getMemory());
#elif USE_OPENCL
			openclGemm(openclGetDefaultContext(device_index), op_A, op_B, alpha, A.getDescriptor(), A.getMemory(), B.getDescriptor(), B.getMemory(), beta,
								C_tested.getDescriptor(), C_tested.getMemory());
#endif
			return diffForTest(C_baseline, C_tested);
		}

		ConcatTester::ConcatTester(avDeviceIndex_t idx, std::initializer_list<int> shape, avDataType_t dtype) :
				device_index(idx),
				shape(shape),
				dtype(dtype)
		{
		}
		double ConcatTester::getDifference() noexcept
		{
			assert(shape.back() >= 3);
			std::vector<int> shape1(shape);
			std::vector<int> shape2(shape);
			std::vector<int> shape3(shape);
			shape1.back() = std::max(1, shape.back() * 2 / 10);
			shape2.back() = std::max(1, shape.back() * 5 / 10);
			shape3.back() = shape.back() - shape1.back() - shape2.back();
			assert(shape3.back() >= 0);
			TensorWrapper input1(shape1, dtype, device_index);
			TensorWrapper input2(shape2, dtype, device_index);
			TensorWrapper input3(shape3, dtype, device_index);

			TensorWrapper output_baseline(shape, dtype, device_index);
			TensorWrapper output_tested(shape, dtype, device_index);

			initForTest(input1, 0.0);
			initForTest(input2, 1.0);
			initForTest(input3, 2.0);

			std::vector<avTensorDescriptor_t> desc = { input1.getRefDescriptor(), input2.getRefDescriptor(), input3.getRefDescriptor() };
			std::vector<avMemoryDescriptor_t> mem = { input1.getRefMemory(), input2.getRefMemory(), input3.getRefMemory() };

			refConcatTensors(0, output_baseline.getRefDescriptor(), output_baseline.getRefMemory(), desc.data(), mem.data(), 3);

			desc = { input1.getDescriptor(), input2.getDescriptor(), input3.getDescriptor() };
			mem = { input1.getMemory(), input2.getMemory(), input3.getMemory() };

#if USE_CPU
			cpuConcatTensors(cpuGetDefaultContext(), output_tested.getDescriptor(), output_tested.getMemory(), desc.data(), mem.data(), 3);
#elif USE_CUDA
			cudaConcatTensors(cudaGetDefaultContext(device_index), output_tested.getDescriptor(), output_tested.getMemory(), desc.data(), mem.data(), 3);
#elif USE_OPENCL
			openclConcatTensors(openclGetDefaultContext(device_index), output_tested.getDescriptor(), output_tested.getMemory(), desc.data(), mem.data(), 3);
#endif

			return diffForTest(output_baseline, output_tested);
		}

		SplitTester::SplitTester(avDeviceIndex_t idx, std::initializer_list<int> shape, avDataType_t dtype) :
				device_index(idx),
				shape(shape),
				dtype(dtype)
		{
		}
		double SplitTester::getDifference() noexcept
		{
			assert(shape.back() >= 3);
			std::vector<int> shape1(shape);
			std::vector<int> shape2(shape);
			std::vector<int> shape3(shape);
			shape1.back() = std::max(1, shape.back() * 2 / 10);
			shape2.back() = std::max(1, shape.back() * 5 / 10);
			shape3.back() = shape.back() - shape1.back() - shape2.back();
			assert(shape3.back() >= 0);
			TensorWrapper output1_baseline(shape1, dtype, device_index);
			TensorWrapper output2_baseline(shape2, dtype, device_index);
			TensorWrapper output3_baseline(shape3, dtype, device_index);

			TensorWrapper output1_tested(shape1, dtype, device_index);
			TensorWrapper output2_tested(shape2, dtype, device_index);
			TensorWrapper output3_tested(shape3, dtype, device_index);

			TensorWrapper input(shape, dtype, device_index);
			initForTest(input, 0.0);

			std::vector<avTensorDescriptor_t> desc = { output1_baseline.getRefDescriptor(), output2_baseline.getRefDescriptor(),
					output3_baseline.getRefDescriptor() };
			std::vector<avMemoryDescriptor_t> mem = { output1_baseline.getRefMemory(), output2_baseline.getRefMemory(),
					output3_baseline.getRefMemory() };

			refSplitTensors(0, desc.data(), mem.data(), input.getRefDescriptor(), input.getRefMemory(), 3);

			desc = { output1_tested.getDescriptor(), output2_tested.getDescriptor(), output3_tested.getDescriptor() };
			mem = { output1_tested.getMemory(), output2_tested.getMemory(), output3_tested.getMemory() };
#if USE_CPU
			cpuSplitTensors(cpuGetDefaultContext(), desc.data(), mem.data(), input.getDescriptor(), input.getMemory(), 3);
#elif USE_CUDA
			cudaSplitTensors(cudaGetDefaultContext(device_index), desc.data(), mem.data(), input.getDescriptor(), input.getMemory(), 3);
#elif USE_OPENCL
			openclSplitTensors(openclGetDefaultContext(device_index), desc.data(), mem.data(), input.getDescriptor(), input.getMemory(), 3);
#endif

			return diffForTest(output1_baseline, output1_tested) + diffForTest(output2_baseline, output2_tested)
					+ diffForTest(output3_baseline, output3_tested);
		}

		TransposeTester::TransposeTester(avDeviceIndex_t idx, std::initializer_list<int> shape, avDataType_t dtype) :
				device_index(idx),
				shape(shape),
				dtype(dtype)
		{
		}
		double TransposeTester::getDifference(const std::vector<int> &ordering) noexcept
		{
			assert(ordering.size() == shape.size());
			TensorWrapper input(shape, dtype, device_index);
			initForTest(input, 0.0);

			std::vector<int> transposed_shape(shape.size());
			for (size_t i = 0; i < ordering.size(); i++)
				transposed_shape[i] = shape[ordering[i]];
			TensorWrapper output_baseline(transposed_shape, dtype, device_index);
			TensorWrapper output_tested(transposed_shape, dtype, device_index);

			refTranspose(0, output_baseline.getRefDescriptor(), output_baseline.getRefMemory(), input.getRefDescriptor(), input.getRefMemory(),
					ordering.data());
#if USE_CPU
			cpuTranspose(cpuGetDefaultContext(), output_tested.getDescriptor(), output_tested.getMemory(), input.getDescriptor(), input.getMemory(),
					ordering.data());
#elif USE_CUDA
			cudaTranspose(cudaGetDefaultContext(device_index), output_tested.getDescriptor(), output_tested.getMemory(), input.getDescriptor(), input.getMemory(),
					ordering.data());
#elif USE_OPENCL
			openclTranspose(openclGetDefaultContext(device_index), output_tested.getDescriptor(), output_tested.getMemory(), input.getDescriptor(), input.getMemory(),
					ordering.data());
#endif

			return diffForTest(output_baseline, output_tested);
		}

		UnaryOpTester::UnaryOpTester(avDeviceIndex_t idx, avUnaryOp_t operation, std::initializer_list<int> shape, avDataType_t dtype) :
				device_index(idx),
				op(operation),
				input(shape, dtype, idx),
				output_baseline(shape, dtype, idx),
				output_tested(shape, dtype, idx)
		{
			initForTest(input, 0.0, 2.0);
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
		}
		double UnaryOpTester::getDifference(const void *alpha, const void *beta) noexcept
		{
			refUnaryOp(0, op, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory());
#if USE_CPU
			cpuUnaryOp(cpuGetDefaultContext(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_CUDA
			cudaUnaryOp(cudaGetDefaultContext(device_index), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_OPENCL
			openclUnaryOp(openclGetDefaultContext(device_index), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
				output_tested.getMemory());
#endif
			return diffForTest(output_baseline, output_tested);
		}

		BinaryOpTester::BinaryOpTester(avDeviceIndex_t idx, avBinaryOp_t operation, std::initializer_list<int> shape, avDataType_t dtype) :
				device_index(idx),
				op(operation),
				input(shape, dtype, idx),
				input_same(shape, dtype, idx),
				input_1d( { shape.begin()[shape.size() - 1] }, dtype, idx),
				input_single( { 1 }, dtype, idx),
				output_baseline(shape, dtype, idx),
				output_tested(shape, dtype, idx)
		{
			initForTest(input, 0.0, 2.0);
			initForTest(input_same, 1.0, 2.0);
			initForTest(input_1d, 1.0, 2.0);
			initForTest(input_single, 1.0, 2.0);
		}
		double BinaryOpTester::getDifferenceSame(const void *alpha1, const void *alpha2, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			refBinaryOp(0, op, alpha1, input.getRefDescriptor(), input.getRefMemory(), alpha2, input_same.getRefDescriptor(),
					input_same.getRefMemory(), beta, output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
#if USE_CPU
			cpuBinaryOp(cpuGetDefaultContext(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_same.getDescriptor(),
					input_same.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_CUDA
			cudaBinaryOp(cudaGetDefaultContext(device_index), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_same.getDescriptor(),
					input_same.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclBinaryOp(openclGetDefaultContext(device_index), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_same.getDescriptor(),
					input_same.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#endif

			return diffForTest(output_baseline, output_tested);
		}
		double BinaryOpTester::getDifference1D(const void *alpha1, const void *alpha2, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			refBinaryOp(0, op, alpha1, input.getRefDescriptor(), input.getRefMemory(), alpha2, input_1d.getRefDescriptor(), input_1d.getRefMemory(),
					beta, output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
#if USE_CPU
			cpuBinaryOp(cpuGetDefaultContext(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_1d.getDescriptor(),
					input_1d.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_CUDA
			cudaBinaryOp(cudaGetDefaultContext(device_index), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_1d.getDescriptor(),
					input_1d.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclBinaryOp(openclGetDefaultContext(device_index), op, alpha1, input.getRefDescriptor(), input.getMemory(), alpha2, input_1d.getDescriptor(),
					input_1d.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#endif
			return diffForTest(output_baseline, output_tested);
		}
		double BinaryOpTester::getDifferenceSingle(const void *alpha1, const void *alpha2, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			refBinaryOp(0, op, alpha1, input.getRefDescriptor(), input.getRefMemory(), alpha2, input_single.getRefDescriptor(),
					input_single.getRefMemory(), beta, output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
#if USE_CPU
			cpuBinaryOp(cpuGetDefaultContext(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_single.getDescriptor(),
					input_single.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_CUDA
			cudaBinaryOp(cudaGetDefaultContext(device_index), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_single.getDescriptor(),
					input_single.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclBinaryOp(openclGetDefaultContext(device_index), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_single.getDescriptor(),
					input_single.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#endif
			return diffForTest(output_baseline, output_tested);
		}

		ReductionTester::ReductionTester(avDeviceIndex_t idx, avReduceOp_t operation, std::initializer_list<int> shape, avDataType_t dtype) :
				device_index(idx),
				op(operation),
				input(shape, dtype, idx),
				output_baseline_1d( { shape.begin()[shape.size() - 1] }, dtype, idx),
				output_tested_1d( { shape.begin()[shape.size() - 1] }, dtype, idx),
				output_baseline_single( { 1 }, dtype, idx),
				output_tested_single( { 1 }, dtype, idx)
		{
			initForTest(input, 0.0, 1.0);
		}
		double ReductionTester::getDifference1D(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline_1d, 0.1);
			initForTest(output_tested_1d, 0.1);

			refReduceTensor(0, op, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline_1d.getRefDescriptor(),
					output_baseline_1d.getRefMemory());
#if USE_CPU
			cpuReduceTensor(cpuGetDefaultContext(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_1d.getDescriptor(),
					output_tested_1d.getMemory());
#elif USE_CUDA
			cudaReduceTensor(cudaGetDefaultContext(device_index), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_1d.getDescriptor(),
					output_tested_1d.getMemory());
#elif USE_OPENCL
			openclReduceTensor(openclGetDefaultContext(device_index), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_1d.getDescriptor(),
					output_tested_1d.getMemory());
#endif
			return diffForTest(output_baseline_1d, output_tested_1d);
		}
		double ReductionTester::getDifferenceSingle(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline_single, 0.1);
			initForTest(output_tested_single, 0.1);

			refReduceTensor(0, op, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline_single.getRefDescriptor(),
					output_baseline_single.getRefMemory());
#if USE_CPU
			cpuReduceTensor(cpuGetDefaultContext(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_single.getDescriptor(),
					output_tested_single.getMemory());
#elif USE_CUDA
			cudaReduceTensor(cudaGetDefaultContext(device_index), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_single.getDescriptor(),
					output_tested_single.getMemory());
#elif USE_OPENCL
			openclReduceTensor(openclGetDefaultContext(device_index), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_single.getDescriptor(),
					output_tested_single.getMemory());
#endif
			return diffForTest(output_baseline_single, output_tested_single);
		}

		BatchNormTester::BatchNormTester(avDeviceIndex_t idx, std::vector<int> shape, avDataType_t dtype) :
				device_index(idx),
				shape(shape),
				dtype(dtype)
		{
		}
		double BatchNormTester::getDifferenceInference(const void *alpha, const void *beta) noexcept
		{
			avActivationType_t activation = AVOCADO_ACTIVATION_SIGMOID;
			double epsilon = 1.0e-3;

			TensorWrapper input(shape, dtype, device_index);
			TensorWrapper output_baseline(shape, dtype, device_index);
			TensorWrapper output_tested(shape, dtype, device_index);

			initForTest(input, 0.0);
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			TensorWrapper scale( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper bias( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper mean( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper variance( { shape[shape.size() - 1] }, dtype, device_index);
			initForTest(scale, 0.0, 1.0);
			initForTest(bias, 1.0);
			initForTest(mean, 2.0);
			initForTest(variance, 3.0, 2.0);

			refBatchNormInference(0, activation, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory(), scale.getRefDescriptor(), scale.getRefMemory(), bias.getRefMemory(), mean.getRefMemory(),
					variance.getRefMemory(), epsilon);
#if USE_CPU
			cpuBatchNormInference(cpuGetDefaultContext(), activation, alpha, input.getDescriptor(), input.getMemory(), beta,
					output_tested.getDescriptor(), output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(),
					mean.getMemory(), variance.getMemory(), epsilon);
#elif USE_CUDA
			cudaBatchNormInference(cpuGetDefaultContext(device_index), activation, alpha, input.getDescriptor(), input.getMemory(), beta,
								output_tested.getDescriptor(), output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(),
								mean.getMemory(), variance.getMemory(), epsilon);
#elif USE_OPENCL
			openclBatchNormInference(cpuGetDefaultContext(device_index), activation, alpha, input.getDescriptor(), input.getMemory(), beta,
								output_tested.getDescriptor(), output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(),
								mean.getMemory(), variance.getMemory(), epsilon);
#endif
			return diffForTest(output_baseline, output_tested);
		}
		double BatchNormTester::getDifferenceForward(const void *alpha, const void *beta) noexcept
		{
			avActivationType_t activation = AVOCADO_ACTIVATION_SIGMOID;
			double epsilon = 1.0e-3;

			TensorWrapper input(shape, dtype, device_index);
			TensorWrapper output_baseline(shape, dtype, device_index);
			TensorWrapper output_tested(shape, dtype, device_index);

			initForTest(input, 0.0);
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			TensorWrapper scale( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper bias( { shape[shape.size() - 1] }, dtype, device_index);
			initForTest(scale, 0.0, 1.0);
			initForTest(bias, 1.0);

			TensorWrapper mean_baseline( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper variance_baseline( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper mean_tested( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper variance_tested( { shape[shape.size() - 1] }, dtype, device_index);

			refBatchNormForward(0, activation, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory(), scale.getRefDescriptor(), scale.getRefMemory(), bias.getRefMemory(), mean_baseline.getRefMemory(),
					variance_baseline.getRefMemory(), epsilon);
#if USE_CPU
			cpuBatchNormForward(cpuGetDefaultContext(), activation, alpha, input.getDescriptor(), input.getMemory(), beta,
					output_tested.getDescriptor(), output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(),
					mean_tested.getMemory(), variance_tested.getMemory(), epsilon);
#elif USE_CUDA
			cudaBatchNormInference(cudaGetDefaultContext(device_index), activation, alpha, input.getDescriptor(), input.getMemory(), beta,
					output_tested.getDescriptor(), output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(),
					mean_tested.getMemory(), variance_tested.getMemory(), epsilon);
#elif USE_OPENCL
			openclBatchNormInference(openclGetDefaultContext(device_index), activation, alpha, input.getDescriptor(), input.getMemory(), beta,
					output_tested.getDescriptor(), output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(),
					mean_tested.getMemory(), variance_tested.getMemory(), epsilon);
#endif
			return diffForTest(mean_baseline, mean_tested) + diffForTest(variance_baseline, variance_tested)
					+ diffForTest(output_baseline, output_tested);
		}
		double BatchNormTester::getDifferenceBackward(const void *alpha, const void *beta) noexcept
		{
			avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR;
			double epsilon = 1.0e-3;

			TensorWrapper input(shape, dtype, device_index);
			TensorWrapper output(shape, dtype, device_index);
			TensorWrapper gradientOut_baseline(shape, dtype, device_index);
			TensorWrapper gradientIn_baseline(shape, dtype, device_index);

			TensorWrapper gradientOut_tested(shape, dtype, device_index);
			TensorWrapper gradientIn_tested(shape, dtype, device_index);

			initForTest(input, 0.0);
			initForTest(output, 0.1);
			initForTest(gradientOut_baseline, 1.0);
			initForTest(gradientOut_tested, 1.0);
			initForTest(gradientIn_baseline, 1.0);
			initForTest(gradientIn_tested, 1.0);

			TensorWrapper scale( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper bias( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper mean( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper variance( { shape[shape.size() - 1] }, dtype, device_index);
			initForTest(scale, 0.0, 1.0);
			initForTest(bias, 1.0);

			TensorWrapper scaleUpdate_baseline( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper biasUpdate_baseline( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper scaleUpdate_tested( { shape[shape.size() - 1] }, dtype, device_index);
			TensorWrapper biasUpdate_tested( { shape[shape.size() - 1] }, dtype, device_index);

			initForTest(scaleUpdate_baseline, 0.0, 1.0);
			initForTest(biasUpdate_baseline, 1.0);
			initForTest(scaleUpdate_tested, 0.0, 1.0);
			initForTest(biasUpdate_tested, 1.0);

			refBatchNormForward(0, activation, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output.getRefDescriptor(),
					output.getRefMemory(), scale.getRefDescriptor(), scale.getRefMemory(), bias.getRefMemory(), mean.getRefMemory(),
					variance.getRefMemory(), epsilon);
			refBatchNormBackward(0, activation, alpha, input.getRefDescriptor(), input.getRefMemory(), output.getRefDescriptor(),
					output.getRefMemory(), beta, gradientIn_baseline.getRefDescriptor(), gradientIn_baseline.getRefMemory(),
					gradientOut_baseline.getRefDescriptor(), gradientOut_baseline.getRefMemory(), scale.getRefDescriptor(), scale.getRefMemory(),
					mean.getRefMemory(), variance.getRefMemory(), alpha, beta, scaleUpdate_baseline.getRefMemory(),
					biasUpdate_baseline.getRefMemory(), epsilon);

			initForTest(output, 0.1);
#if USE_CPU
			cpuBatchNormForward(cpuGetDefaultContext(), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output.getDescriptor(),
					output.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean.getMemory(), variance.getMemory(), epsilon);
			cpuBatchNormBackward(cpuGetDefaultContext(), activation, alpha, input.getDescriptor(), input.getMemory(), output.getDescriptor(),
					output.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory(), gradientOut_tested.getDescriptor(),
					gradientOut_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), mean.getMemory(), variance.getMemory(), alpha, beta,
					scaleUpdate_tested.getMemory(), biasUpdate_tested.getMemory(), epsilon);
#elif USE_CUDA
			cudaBatchNormForward(cudaGetDefaultContext(device_index), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output.getDescriptor(),
					output.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean.getMemory(), variance.getMemory(), epsilon);
			cudaBatchNormBackward(cudaGetDefaultContext(device_index), activation, alpha, input.getDescriptor(), input.getMemory(), output.getDescriptor(),
					output.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory(), gradientOut_tested.getDescriptor(),
					gradientOut_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), mean.getMemory(), variance.getMemory(), alpha, beta,
					scaleUpdate_tested.getMemory(), biasUpdate_tested.getMemory(), epsilon);
#elif USE_OPENCL
			openclBatchNormForward(openclGetDefaultContext(device_index), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output.getDescriptor(),
					output.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean.getMemory(), variance.getMemory(), epsilon);
			openclBatchNormBackward(openclGetDefaultContext(device_index), activation, alpha, input.getDescriptor(), input.getMemory(), output.getDescriptor(),
					output.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory(), gradientOut_tested.getDescriptor(),
					gradientOut_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), mean.getMemory(), variance.getMemory(), alpha, beta,
					scaleUpdate_tested.getMemory(), biasUpdate_tested.getMemory(), epsilon);
#endif
			return diffForTest(scaleUpdate_baseline, scaleUpdate_tested) + diffForTest(biasUpdate_baseline, biasUpdate_tested)
					+ diffForTest(gradientOut_tested, gradientOut_baseline) + diffForTest(gradientIn_tested, gradientIn_baseline);
		}

	} /* namespace backend */
} /* namespace avocado */

