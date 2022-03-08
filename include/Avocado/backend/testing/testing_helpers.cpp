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
#  include "../src/vectors/simd_utils.hpp"
#  include "../src/vectors/fp16_simd.hpp"
#elif USE_CUDA
#  include <CudaBackend/cuda_backend.h>
#  include "../src/numbers/numbers.cuh"
#elif USE_OPENCL
#  include <OpenCLBackend/opencl_backend.h>
#endif
#include <ReferenceBackend/reference_backend.h>

#include <complex>
#include <cmath>
#include <memory>
#include <iostream>
#include <cassert>
#include <gtest/gtest.h>

namespace
{
	using namespace avocado;
	using namespace avocado::backend;

	template<typename T>
	void init_for_test(void *ptr, size_t elements, T offset, T minValue, T maxValue)
	{
		for (size_t i = 0; i < elements; i++)
			reinterpret_cast<T*>(ptr)[i] = minValue + 0.5 * (1.0 + sin(i / 10.0f + offset)) * (maxValue - minValue);
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

	template<typename T>
	void print(const std::vector<T> &vec)
	{
		for (size_t i = 0; i < vec.size(); i++)
			std::cout << vec[i] << ' ';
		std::cout << '\n';
	}

	int dtypeSize(avDataType_t dtype) noexcept
	{
		switch (dtype)
		{
			default:
			case AVOCADO_DTYPE_UNKNOWN:
				return 0;
			case AVOCADO_DTYPE_UINT8:
			case AVOCADO_DTYPE_INT8:
				return 1;
			case AVOCADO_DTYPE_INT16:
			case AVOCADO_DTYPE_FLOAT16:
			case AVOCADO_DTYPE_BFLOAT16:
				return 2;
			case AVOCADO_DTYPE_INT32:
			case AVOCADO_DTYPE_FLOAT32:
				return 4;
			case AVOCADO_DTYPE_INT64:
			case AVOCADO_DTYPE_FLOAT64:
			case AVOCADO_DTYPE_COMPLEX32:
				return 8;
			case AVOCADO_DTYPE_COMPLEX64:
				return 16;
		}
	}

	std::vector<int> winograd_matrices_shape(const std::vector<int> &inputShape, const std::vector<int> &filterShape, int transformSize) noexcept
	{
		int nb_tiles = inputShape[0]; // batch size
		for (size_t i = 1; i < inputShape.size() - 1; i++)
			nb_tiles *= ((inputShape[i] + transformSize - 1) / transformSize);
		int tile_size = filterShape[1] + transformSize - 1;

		return std::vector<int>( { tile_size * tile_size, nb_tiles, inputShape.back() });
	}
	template<typename T>
	T square(T x) noexcept
	{
		return x * x;
	}
}

namespace avocado
{
	namespace backend
	{
		void setMasterContext(avDeviceIndex_t deviceIndex, bool useDefault)
		{
//			master_context = ContextWrapper(deviceIndex);
		}
		const ContextWrapper& getMasterContext()
		{
			static const ContextWrapper master_context(0, true, true);
			return master_context;
		}

		avContextDescriptor_t getContextDesc()
		{
			return getMasterContext().getDescriptor();
		}
		avContextDescriptor_t getContextRefDesc()
		{
			return getMasterContext().getRefDescriptor();
		}
		avDeviceIndex_t getDevice()
		{
			return getMasterContext().getDeviceIndex();
		}

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
#if USE_CPU
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, &result);
#elif USE_CUDA
					cudaGetDeviceProperty(getDevice(), AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION, &result);
#endif
					return result;
				}
				case AVOCADO_DTYPE_BFLOAT16:
				{
					bool result;
#if USE_CPU
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_BFLOAT16, &result);
#elif USE_CUDA
					cudaGetDeviceProperty(getDevice(), AVOCADO_DEVICE_SUPPORTS_BFLOAT16, &result);
#endif
					return result;
				}
				case AVOCADO_DTYPE_FLOAT32:
				case AVOCADO_DTYPE_COMPLEX32:
				{
					bool result;
#if USE_CPU
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION, &result);
#else
					cudaGetDeviceProperty(getDevice(), AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION, &result);
#endif
					return result;
				}
				case AVOCADO_DTYPE_FLOAT64:
				case AVOCADO_DTYPE_COMPLEX64:
				{
					bool result;
#if USE_CPU
					cpuGetDeviceProperty(AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION, &result);
#else
					cudaGetDeviceProperty(getDevice(), AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION, &result);
#endif
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

		void initForTest(TensorWrapper &t, double offset, double minValue, double maxValue)
		{
			std::unique_ptr<char[]> tmp = std::make_unique<char[]>(t.sizeInBytes());
			switch (t.dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					init_for_test<uint8_t>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_INT8:
					init_for_test<int8_t>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_INT16:
					init_for_test<int16_t>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_INT32:
					init_for_test<int32_t>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_INT64:
					init_for_test<int64_t>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
				{
					std::unique_ptr<float[]> tmp2 = std::make_unique<float[]>(t.volume());
					init_for_test<float>(tmp2.get(), t.volume(), offset, minValue, maxValue);
					refChangeTypeHost(0ll, tmp.get(), static_cast<avDataType_t>(t.dtype()), tmp2.get(), AVOCADO_DTYPE_FLOAT32, t.volume());
					break;
				}
				case AVOCADO_DTYPE_FLOAT32:
					init_for_test<float>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					init_for_test<double>(tmp.get(), t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					init_for_test<float>(tmp.get(), 2 * t.volume(), offset, minValue, maxValue);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					init_for_test<double>(tmp.get(), 2 * t.volume(), offset, minValue, maxValue);
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
					abs_for_test<float>(tmp2.get(), tensor.volume());
					break;
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
			if (tensor.numberOfDimensions() == 4)
			{
				std::cout << "------------------------------------------------------------\n";
				for (int b = 0; b < tensor.firstDim(); b++)
				{
					std::cout << "--batch " << b << '\n';
					for (int f = 0; f < tensor.lastDim(); f++)
					{
						std::cout << "----channel " << f << '\n';
						for (int h = 0; h < tensor.dimension(1); h++)
						{
							for (int w = 0; w < tensor.dimension(2); w++)
								std::cout << tensor.get<T>( { b, h, w, f }) << ' ';
							std::cout << '\n';
						}
					}
				}
				std::cout << "------------------------------------------------------------\n";
			}
		}

		double epsilonForTest(avDataType_t dtype)
		{
			switch (dtype)
			{
				default:
					return 0.0;
				case AVOCADO_DTYPE_FLOAT16:
				case AVOCADO_DTYPE_BFLOAT16:
					return 1.0e-2;
				case AVOCADO_DTYPE_FLOAT32:
					return 1.0e-4;
				case AVOCADO_DTYPE_FLOAT64:
					return 1.0e-6;
			}
		}

		ActivationTester::ActivationTester(avActivationType_t activation, std::initializer_list<int> shape, avDataType_t dtype) :
				act(activation),
				input(shape, dtype, getDevice()),
				gradientOut(shape, dtype, getDevice()),
				output_baseline(shape, dtype, getDevice()),
				output_tested(shape, dtype, getDevice()),
				gradientIn_baseline(shape, dtype, getDevice()),
				gradientIn_tested(shape, dtype, getDevice())
		{
			initForTest(input, 0.0);
			initForTest(gradientOut, 1.0);
		}
		double ActivationTester::getDifferenceForward(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			refActivationForward(getContextRefDesc(), act, alpha, input.getRefDescriptor(), input.getRefMemory(), beta,
					output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
#if USE_CPU
			cpuActivationForward(getMasterContext().getDescriptor(), act, alpha, input.getDescriptor(), input.getMemory(), beta,
					output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_CUDA
			cudaActivationForward(getContextDesc(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_OPENCL
			openclActivationForward(getContextDesc(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		double ActivationTester::getDifferenceBackward(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			initForTest(gradientIn_baseline, 0.2);
			initForTest(gradientIn_tested, 0.2);

			refActivationForward(getContextRefDesc(), act, alpha, input.getRefDescriptor(), input.getRefMemory(), beta,
					output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
			refActivationBackward(getContextRefDesc(), act, alpha, output_baseline.getRefDescriptor(), output_baseline.getRefMemory(),
					gradientOut.getRefDescriptor(), gradientOut.getRefMemory(), beta, gradientIn_baseline.getRefDescriptor(),
					gradientIn_baseline.getRefMemory());

#if USE_CPU
			cpuActivationForward(getContextDesc(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			cpuActivationBackward(getContextDesc(), act, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#elif USE_CUDA
			cudaActivationForward(getContextDesc(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			cudaActivationBackward(getContextDesc(), act, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#elif USE_OPENCL
			openclActivationForward(getContextDesc(), act, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			openclActivationBackward(getContextDesc(), act, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(gradientIn_baseline, gradientIn_tested);
		}

		SoftmaxTester::SoftmaxTester(avSoftmaxMode_t mode, std::initializer_list<int> shape, avDataType_t dtype) :
				mode(mode),
				input(shape, dtype, getDevice()),
				gradientOut(shape, dtype, getDevice()),
				output_baseline(shape, dtype, getDevice()),
				output_tested(shape, dtype, getDevice()),
				gradientIn_baseline(shape, dtype, getDevice()),
				gradientIn_tested(shape, dtype, getDevice())
		{
			initForTest(input, 0.0);
			initForTest(gradientOut, 1.0);
		}
		double SoftmaxTester::getDifferenceForward(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			refSoftmaxForward(getContextRefDesc(), mode, alpha, input.getRefDescriptor(), input.getRefMemory(), beta,
					output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
#if USE_CPU
			cpuSoftmaxForward(getContextDesc(), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_CUDA
			cudaSoftmaxForward(getContextDesc(), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_OPENCL
			openclSoftmaxForward(getContextDesc(), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		double SoftmaxTester::getDifferenceBackward(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			initForTest(gradientIn_baseline, 0.2);
			initForTest(gradientIn_tested, 0.2);

			refSoftmaxForward(getContextRefDesc(), mode, alpha, input.getRefDescriptor(), input.getRefMemory(), beta,
					output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
			refSoftmaxBackward(getContextRefDesc(), mode, alpha, output_baseline.getRefDescriptor(), output_baseline.getRefMemory(),
					gradientOut.getRefDescriptor(), gradientOut.getRefMemory(), beta, gradientIn_baseline.getRefDescriptor(),
					gradientIn_baseline.getRefMemory());

#if USE_CPU
			cpuSoftmaxForward(getContextDesc(), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			cpuSoftmaxBackward(getContextDesc(), mode, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#elif USE_CUDA
			cudaSoftmaxForward(getContextDesc(), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			cudaSoftmaxBackward(getContextDesc(), mode, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#elif USE_OPENCL
			openclSoftmaxForward(getContextDesc(), mode, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
			openclSoftmaxBackward(getContextDesc(), mode, alpha, output_tested.getDescriptor(), output_tested.getMemory(), gradientOut.getDescriptor(),
					gradientOut.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(gradientIn_baseline, gradientIn_tested);
		}

		GemmTester::GemmTester(int M, int N, int K, avGemmOperation_t opA, avGemmOperation_t opB, avDataType_t C_type, avDataType_t AB_type) :
				op_A(opA),
				op_B(opB)
		{
			if (opA == AVOCADO_GEMM_OPERATION_N)
				A = TensorWrapper( { M, K }, AB_type, getDevice());
			else
				A = TensorWrapper( { K, M }, AB_type, getDevice());
			if (opB == AVOCADO_GEMM_OPERATION_N)
				B = TensorWrapper( { K, N }, AB_type, getDevice());
			else
				B = TensorWrapper( { N, K }, AB_type, getDevice());

			initForTest(A, 0.0);
			initForTest(B, 1.57);

			C_baseline = TensorWrapper( { M, N }, C_type, getDevice());
			C_tested = TensorWrapper( { M, N }, C_type, getDevice());
			initForTest(C_baseline, 0.1);
			initForTest(C_tested, 0.1);
		}
		GemmTester::GemmTester(int M, int N, int K, avGemmOperation_t opA, avGemmOperation_t opB, avDataType_t dtype) :
				GemmTester(M, N, K, opA, opB, dtype, dtype)
		{
		}
		double GemmTester::getDifference(const void *alpha, const void *beta) noexcept
		{
			refGemm(getContextRefDesc(), op_A, op_B, alpha, A.getRefDescriptor(), A.getRefMemory(), B.getRefDescriptor(), B.getRefMemory(), beta,
					C_baseline.getRefDescriptor(), C_baseline.getRefMemory());
#if USE_CPU
			cpuGemm(getContextDesc(), op_A, op_B, alpha, A.getDescriptor(), A.getMemory(), B.getDescriptor(), B.getMemory(), beta,
					C_tested.getDescriptor(), C_tested.getMemory());
#elif USE_CUDA
			cudaGemm(getContextDesc(), op_A, op_B, alpha, A.getDescriptor(), A.getMemory(), B.getDescriptor(), B.getMemory(), beta, C_tested.getDescriptor(),
					C_tested.getMemory());
#elif USE_OPENCL
			openclGemm(getContextDesc(), op_A, op_B, alpha, A.getDescriptor(), A.getMemory(), B.getDescriptor(), B.getMemory(), beta,
					C_tested.getDescriptor(), C_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(C_baseline, C_tested);
		}

		ConcatTester::ConcatTester(std::initializer_list<int> shape, avDataType_t dtype) :
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
			TensorWrapper input1(shape1, dtype, getDevice());
			TensorWrapper input2(shape2, dtype, getDevice());
			TensorWrapper input3(shape3, dtype, getDevice());

			TensorWrapper output_baseline(shape, dtype, getDevice());
			TensorWrapper output_tested(shape, dtype, getDevice());

			initForTest(input1, 0.0);
			initForTest(input2, 1.0);
			initForTest(input3, 2.0);

			std::vector<avTensorDescriptor_t> desc = { input1.getRefDescriptor(), input2.getRefDescriptor(), input3.getRefDescriptor() };
			std::vector<avMemoryDescriptor_t> mem = { input1.getRefMemory(), input2.getRefMemory(), input3.getRefMemory() };

			refConcatTensors(getContextRefDesc(), output_baseline.getRefDescriptor(), output_baseline.getRefMemory(), desc.data(), mem.data(), 3);

			desc = { input1.getDescriptor(), input2.getDescriptor(), input3.getDescriptor() };
			mem = { input1.getMemory(), input2.getMemory(), input3.getMemory() };

#if USE_CPU
			cpuConcatTensors(getContextDesc(), output_tested.getDescriptor(), output_tested.getMemory(), desc.data(), mem.data(), 3);
#elif USE_CUDA
			cudaConcatTensors(getContextDesc(), output_tested.getDescriptor(), output_tested.getMemory(), desc.data(), mem.data(), 3);
#elif USE_OPENCL
			openclConcatTensors(getContextDesc(), output_tested.getDescriptor(), output_tested.getMemory(), desc.data(), mem.data(), 3);
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}

		SplitTester::SplitTester(std::initializer_list<int> shape, avDataType_t dtype) :
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
			TensorWrapper output1_baseline(shape1, dtype, getDevice());
			TensorWrapper output2_baseline(shape2, dtype, getDevice());
			TensorWrapper output3_baseline(shape3, dtype, getDevice());

			TensorWrapper output1_tested(shape1, dtype, getDevice());
			TensorWrapper output2_tested(shape2, dtype, getDevice());
			TensorWrapper output3_tested(shape3, dtype, getDevice());

			TensorWrapper input(shape, dtype, getDevice());
			initForTest(input, 0.0);

			std::vector<avTensorDescriptor_t> desc = { output1_baseline.getRefDescriptor(), output2_baseline.getRefDescriptor(),
					output3_baseline.getRefDescriptor() };
			std::vector<avMemoryDescriptor_t> mem = { output1_baseline.getRefMemory(), output2_baseline.getRefMemory(),
					output3_baseline.getRefMemory() };

			refSplitTensors(getContextRefDesc(), desc.data(), mem.data(), input.getRefDescriptor(), input.getRefMemory(), 3);

			desc = { output1_tested.getDescriptor(), output2_tested.getDescriptor(), output3_tested.getDescriptor() };
			mem = { output1_tested.getMemory(), output2_tested.getMemory(), output3_tested.getMemory() };
#if USE_CPU
			cpuSplitTensors(getContextDesc(), desc.data(), mem.data(), input.getDescriptor(), input.getMemory(), 3);
#elif USE_CUDA
			cudaSplitTensors(getContextDesc(), desc.data(), mem.data(), input.getDescriptor(), input.getMemory(), 3);
#elif USE_OPENCL
			openclSplitTensors(getContextDesc(), desc.data(), mem.data(), input.getDescriptor(), input.getMemory(), 3);
#endif
			getMasterContext().synchronize();
			return diffForTest(output1_baseline, output1_tested) + diffForTest(output2_baseline, output2_tested)
					+ diffForTest(output3_baseline, output3_tested);
		}

		TransposeTester::TransposeTester(std::initializer_list<int> shape, avDataType_t dtype) :
				shape(shape),
				dtype(dtype)
		{
		}
		double TransposeTester::getDifference(const std::vector<int> &ordering) noexcept
		{
			assert(ordering.size() == shape.size());
			TensorWrapper input(shape, dtype, getDevice());
			initForTest(input, 0.0);

			std::vector<int> transposed_shape(shape.size());
			for (size_t i = 0; i < ordering.size(); i++)
				transposed_shape[i] = shape[ordering[i]];
			TensorWrapper output_baseline(transposed_shape, dtype, getDevice());
			TensorWrapper output_tested(transposed_shape, dtype, getDevice());

			refTranspose(getContextRefDesc(), output_baseline.getRefDescriptor(), output_baseline.getRefMemory(), input.getRefDescriptor(),
					input.getRefMemory(), ordering.data());
#if USE_CPU
			cpuTranspose(getContextDesc(), output_tested.getDescriptor(), output_tested.getMemory(), input.getDescriptor(), input.getMemory(),
					ordering.data());
#elif USE_CUDA
			cudaTranspose(getContextDesc(), output_tested.getDescriptor(), output_tested.getMemory(), input.getDescriptor(), input.getMemory(),
					ordering.data());
#elif USE_OPENCL
			openclTranspose(getContextDesc(), output_tested.getDescriptor(), output_tested.getMemory(), input.getDescriptor(), input.getMemory(),
					ordering.data());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}

		ScaleTester::ScaleTester(std::initializer_list<int> shape, avDataType_t dtype) :
				shape(shape),
				dtype(dtype)
		{
		}
		double ScaleTester::getDifference(const void *alpha) noexcept
		{
			TensorWrapper input(shape, dtype, getDevice());
			initForTest(input, 0.0);

			TensorWrapper output_baseline(shape, dtype, getDevice());
			TensorWrapper output_tested(shape, dtype, getDevice());

			refScaleTensor(getContextRefDesc(), input.getRefDescriptor(), input.getRefMemory(), alpha, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory());
#if USE_CPU
			cpuScaleTensor(getContextDesc(), input.getDescriptor(), input.getMemory(), alpha, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_CUDA
			cudaScaleTensor(getContextDesc(), input.getDescriptor(), input.getMemory(), alpha, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclScaleTensor(getContextDesc(), input.getDescriptor(), input.getMemory(), alpha, output_tested.getDescriptor(), output_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		AddScalarTester::AddScalarTester(std::initializer_list<int> shape, avDataType_t dtype) :
				shape(shape),
				dtype(dtype)
		{
		}
		double AddScalarTester::getDifference(const void *scalar) noexcept
		{
			TensorWrapper input(shape, dtype, getDevice());
			initForTest(input, 0.0);

			TensorWrapper output_baseline(shape, dtype, getDevice());
			TensorWrapper output_tested(shape, dtype, getDevice());

			refAddScalarToTensor(getContextRefDesc(), input.getRefDescriptor(), input.getRefMemory(), scalar, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory());
#if USE_CPU
			cpuAddScalarToTensor(getContextDesc(), input.getDescriptor(), input.getMemory(), scalar, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_CUDA
			cudaAddScalarToTensor(getContextDesc(), input.getDescriptor(), input.getMemory(), scalar, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclAddScalarToTensor(getContextDesc(), input.getDescriptor(), input.getMemory(), scalar, output_tested.getDescriptor(), output_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		AddBiasTester::AddBiasTester(std::initializer_list<int> shape, avDataType_t dtype) :
				AddBiasTester(shape, dtype, dtype, dtype)
		{
		}
		AddBiasTester::AddBiasTester(std::initializer_list<int> shape, avDataType_t input_dtype, avDataType_t output_dtype, avDataType_t bias_dtype) :
				shape(shape),
				input_dtype(input_dtype),
				output_dtype(output_dtype),
				bias_dtype(bias_dtype)
		{
		}
		double AddBiasTester::getDifference(const void *alpha1, const void *alpha2, const void *beta1, const void *beta2, const void *beta3) noexcept
		{
			avActivationType_t activation = AVOCADO_ACTIVATION_SIGMOID;
			TensorWrapper input(shape, input_dtype, getDevice());
			TensorWrapper ext(shape, input_dtype, getDevice());
			TensorWrapper bias( { shape.back() }, bias_dtype, getDevice());
			initForTest(input, 0.0);
			initForTest(ext, 0.5);
			initForTest(bias, 1.0);

			TensorWrapper output_baseline(shape, output_dtype, getDevice());
			TensorWrapper output_tested(shape, output_dtype, getDevice());

			refAddBias(getContextRefDesc(), alpha1, alpha2, input.getRefDescriptor(), input.getRefMemory(), bias.getRefDescriptor(),
					bias.getRefMemory(), output_baseline.getRefDescriptor(), output_baseline.getRefMemory(), beta1, beta2, beta3, ext.getRefMemory(),
					activation);
#if USE_CPU
			cpuAddBias(getContextDesc(), alpha1, alpha2, input.getDescriptor(), input.getMemory(), bias.getDescriptor(), bias.getMemory(),
					output_tested.getDescriptor(), output_tested.getMemory(), beta1, beta2, beta3, ext.getMemory(), activation);
#elif USE_CUDA
			cudaAddBias(getContextDesc(), alpha1, alpha2, input.getDescriptor(), input.getMemory(), bias.getDescriptor(), bias.getMemory(),
					output_tested.getDescriptor(), output_tested.getMemory(), beta1, beta2, beta3, ext.getMemory(), activation);
#elif USE_OPENCL
			openclAddBias(getContextDesc(), alpha1, alpha2, input.getDescriptor(), input.getMemory(), bias.getDescriptor(), bias.getMemory(),
					output_tested.getDescriptor(), output_tested.getMemory(), beta1, beta2, beta3, ext.getMemory(), activation);
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}

		UnaryOpTester::UnaryOpTester(avUnaryOp_t operation, std::initializer_list<int> shape, avDataType_t dtype) :
				op(operation),
				input(shape, dtype, getDevice()),
				output_baseline(shape, dtype, getDevice()),
				output_tested(shape, dtype, getDevice())
		{
			initForTest(input, 0.0, 1.1);
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
		}
		double UnaryOpTester::getDifference(const void *alpha, const void *beta) noexcept
		{
			refUnaryOp(getContextRefDesc(), op, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory());
#if USE_CPU
			cpuUnaryOp(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#elif USE_CUDA
			cudaUnaryOp(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclUnaryOp(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}

		BinaryOpTester::BinaryOpTester(avBinaryOp_t operation, std::initializer_list<int> shape, avDataType_t dtype) :
				op(operation),
				input(shape, dtype, getDevice()),
				input_same(shape, dtype, getDevice()),
				input_1d( { shape.begin()[shape.size() - 1] }, dtype, getDevice()),
				input_single( { 1 }, dtype, getDevice()),
				output_baseline(shape, dtype, getDevice()),
				output_tested(shape, dtype, getDevice())
		{
			initForTest(input, 0.0, 1.1);
			initForTest(input_same, 1.0, 1.1);
			initForTest(input_1d, 1.0, 1.1);
			initForTest(input_single, 1.0, 1.1);
		}
		double BinaryOpTester::getDifferenceSame(const void *alpha1, const void *alpha2, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			refBinaryOp(0, op, alpha1, input.getRefDescriptor(), input.getRefMemory(), alpha2, input_same.getRefDescriptor(),
					input_same.getRefMemory(), beta, output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
#if USE_CPU
			cpuBinaryOp(getContextDesc(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_same.getDescriptor(),
					input_same.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_CUDA
			cudaBinaryOp(getContextDesc(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_same.getDescriptor(), input_same.getMemory(),
					beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclBinaryOp(getContextDesc(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_same.getDescriptor(),
					input_same.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		double BinaryOpTester::getDifference1D(const void *alpha1, const void *alpha2, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			refBinaryOp(0, op, alpha1, input.getRefDescriptor(), input.getRefMemory(), alpha2, input_1d.getRefDescriptor(), input_1d.getRefMemory(),
					beta, output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
#if USE_CPU
			cpuBinaryOp(getContextDesc(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_1d.getDescriptor(),
					input_1d.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_CUDA
			cudaBinaryOp(getContextDesc(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_1d.getDescriptor(), input_1d.getMemory(), beta,
					output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclBinaryOp(getContextDesc(), op, alpha1, input.getRefDescriptor(), input.getMemory(), alpha2, input_1d.getDescriptor(),
					input_1d.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		double BinaryOpTester::getDifferenceSingle(const void *alpha1, const void *alpha2, const void *beta) noexcept
		{
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			refBinaryOp(0, op, alpha1, input.getRefDescriptor(), input.getRefMemory(), alpha2, input_single.getRefDescriptor(),
					input_single.getRefMemory(), beta, output_baseline.getRefDescriptor(), output_baseline.getRefMemory());
#if USE_CPU
			cpuBinaryOp(getContextDesc(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_single.getDescriptor(),
					input_single.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_CUDA
			cudaBinaryOp(getContextDesc(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_single.getDescriptor(), input_single.getMemory(),
					beta, output_tested.getDescriptor(), output_tested.getMemory());
#elif USE_OPENCL
			openclBinaryOp(getContextDesc(), op, alpha1, input.getDescriptor(), input.getMemory(), alpha2, input_single.getDescriptor(),
					input_single.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}

		ReductionTester::ReductionTester(avReduceOp_t operation, std::initializer_list<int> shape, avDataType_t dtype) :
				op(operation),
				input(shape, dtype, getDevice()),
				output_baseline_1d( { shape.begin()[shape.size() - 1] }, dtype, getDevice()),
				output_tested_1d( { shape.begin()[shape.size() - 1] }, dtype, getDevice()),
				output_baseline_single( { 1 }, dtype, getDevice()),
				output_tested_single( { 1 }, dtype, getDevice())
		{
			initForTest(input, 0.0, 0.0, 2.0);
		}
		double ReductionTester::getDifference1D(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline_1d, 0.1);
			initForTest(output_tested_1d, 0.1);

			refReduceTensor(0, op, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline_1d.getRefDescriptor(),
					output_baseline_1d.getRefMemory());
#if USE_CPU
			cpuReduceTensor(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_1d.getDescriptor(),
					output_tested_1d.getMemory());
#elif USE_CUDA
			cudaReduceTensor(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_1d.getDescriptor(),
					output_tested_1d.getMemory());
#elif USE_OPENCL
			openclReduceTensor(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_1d.getDescriptor(),
					output_tested_1d.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline_1d, output_tested_1d);
		}
		double ReductionTester::getDifferenceSingle(const void *alpha, const void *beta) noexcept
		{
			initForTest(output_baseline_single, 0.1);
			initForTest(output_tested_single, 0.1);

			refReduceTensor(0, op, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline_single.getRefDescriptor(),
					output_baseline_single.getRefMemory());
#if USE_CPU
			cpuReduceTensor(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_single.getDescriptor(),
					output_tested_single.getMemory());
#elif USE_CUDA
			cudaReduceTensor(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_single.getDescriptor(),
					output_tested_single.getMemory());
#elif USE_OPENCL
			openclReduceTensor(getContextDesc(), op, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested_single.getDescriptor(),
					output_tested_single.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline_single, output_tested_single);
		}

		BatchNormTester::BatchNormTester(std::vector<int> shape, avDataType_t dtype) :
				shape(shape),
				dtype(dtype)
		{
		}
		double BatchNormTester::getDifferenceInference(const void *alpha, const void *beta) noexcept
		{
			avActivationType_t activation = AVOCADO_ACTIVATION_SIGMOID;
			double epsilon = 1.0e-3;

			TensorWrapper input(shape, dtype, getDevice());
			TensorWrapper output_baseline(shape, dtype, getDevice());
			TensorWrapper output_tested(shape, dtype, getDevice());

			initForTest(input, 0.0);
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			TensorWrapper scale( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper bias( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper mean( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper variance( { shape[shape.size() - 1] }, dtype, getDevice());
			initForTest(scale, 0.0, 1.0);
			initForTest(bias, 1.0);
			initForTest(mean, 2.0);
			initForTest(variance, 3.0, 2.0);

			refBatchNormInference(0, activation, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory(), scale.getRefDescriptor(), scale.getRefMemory(), bias.getRefMemory(), mean.getRefMemory(),
					variance.getRefMemory(), epsilon);
#if USE_CPU
			cpuBatchNormInference(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean.getMemory(), variance.getMemory(),
					epsilon);
#elif USE_CUDA
			cudaBatchNormInference(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean.getMemory(), variance.getMemory(), epsilon);
#elif USE_OPENCL
			openclBatchNormInference(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta,
					output_tested.getDescriptor(), output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(),
					mean.getMemory(), variance.getMemory(), epsilon);
#endif
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		double BatchNormTester::getDifferenceForward(const void *alpha, const void *beta) noexcept
		{
			avActivationType_t activation = AVOCADO_ACTIVATION_SIGMOID;
			double epsilon = 1.0e-3;

			TensorWrapper input(shape, dtype, getDevice());
			TensorWrapper output_baseline(shape, dtype, getDevice());
			TensorWrapper output_tested(shape, dtype, getDevice());

			initForTest(input, 0.0);
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			TensorWrapper scale( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper bias( { shape[shape.size() - 1] }, dtype, getDevice());
			initForTest(scale, 0.0, 1.0);
			initForTest(bias, 1.0);

			TensorWrapper mean_baseline( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper variance_baseline( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper mean_tested( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper variance_tested( { shape[shape.size() - 1] }, dtype, getDevice());

			refBatchNormForward(0, activation, alpha, input.getRefDescriptor(), input.getRefMemory(), beta, output_baseline.getRefDescriptor(),
					output_baseline.getRefMemory(), scale.getRefDescriptor(), scale.getRefMemory(), bias.getRefMemory(), mean_baseline.getRefMemory(),
					variance_baseline.getRefMemory(), epsilon);
#if USE_CPU
			cpuBatchNormForward(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean_tested.getMemory(),
					variance_tested.getMemory(), epsilon);
#elif USE_CUDA
			cudaBatchNormInference(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output_tested.getDescriptor(),
					output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean_tested.getMemory(), variance_tested.getMemory(),
					epsilon);
#elif USE_OPENCL
			openclBatchNormInference(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta,
					output_tested.getDescriptor(), output_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(),
					mean_tested.getMemory(), variance_tested.getMemory(), epsilon);
#endif
			getMasterContext().synchronize();
			return diffForTest(mean_baseline, mean_tested); // + diffForTest(variance_baseline, variance_tested) + diffForTest(output_baseline, output_tested);
		}
		double BatchNormTester::getDifferenceBackward(const void *alpha, const void *beta) noexcept
		{
			avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR;
			double epsilon = 1.0e-3;

			TensorWrapper input(shape, dtype, getDevice());
			TensorWrapper output(shape, dtype, getDevice());
			TensorWrapper gradientOut_baseline(shape, dtype, getDevice());
			TensorWrapper gradientIn_baseline(shape, dtype, getDevice());

			TensorWrapper gradientOut_tested(shape, dtype, getDevice());
			TensorWrapper gradientIn_tested(shape, dtype, getDevice());

			initForTest(input, 0.0);
			initForTest(output, 0.1);
			initForTest(gradientOut_baseline, 1.0);
			initForTest(gradientOut_tested, 1.0);
			initForTest(gradientIn_baseline, 1.0);
			initForTest(gradientIn_tested, 1.0);

			TensorWrapper scale( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper bias( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper mean( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper variance( { shape[shape.size() - 1] }, dtype, getDevice());
			initForTest(scale, 0.0, 1.0);
			initForTest(bias, 1.0);

			TensorWrapper scaleUpdate_baseline( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper biasUpdate_baseline( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper scaleUpdate_tested( { shape[shape.size() - 1] }, dtype, getDevice());
			TensorWrapper biasUpdate_tested( { shape[shape.size() - 1] }, dtype, getDevice());

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
			cpuBatchNormForward(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output.getDescriptor(),
					output.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean.getMemory(), variance.getMemory(), epsilon);
			cpuBatchNormBackward(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), output.getDescriptor(),
					output.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory(), gradientOut_tested.getDescriptor(),
					gradientOut_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), mean.getMemory(), variance.getMemory(), alpha, beta,
					scaleUpdate_tested.getMemory(), biasUpdate_tested.getMemory(), epsilon);
#elif USE_CUDA
			cudaBatchNormForward(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output.getDescriptor(),
					output.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean.getMemory(), variance.getMemory(), epsilon);
			cudaBatchNormBackward(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), output.getDescriptor(), output.getMemory(),
					beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory(), gradientOut_tested.getDescriptor(), gradientOut_tested.getMemory(),
					scale.getDescriptor(), scale.getMemory(), mean.getMemory(), variance.getMemory(), alpha, beta, scaleUpdate_tested.getMemory(),
					biasUpdate_tested.getMemory(), epsilon);
#elif USE_OPENCL
			openclBatchNormForward(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), beta, output.getDescriptor(),
					output.getMemory(), scale.getDescriptor(), scale.getMemory(), bias.getMemory(), mean.getMemory(), variance.getMemory(), epsilon);
			openclBatchNormBackward(getContextDesc(), activation, alpha, input.getDescriptor(), input.getMemory(), output.getDescriptor(),
					output.getMemory(), beta, gradientIn_tested.getDescriptor(), gradientIn_tested.getMemory(), gradientOut_tested.getDescriptor(),
					gradientOut_tested.getMemory(), scale.getDescriptor(), scale.getMemory(), mean.getMemory(), variance.getMemory(), alpha, beta,
					scaleUpdate_tested.getMemory(), biasUpdate_tested.getMemory(), epsilon);
#endif
			getMasterContext().synchronize();
			return diffForTest(scaleUpdate_baseline, scaleUpdate_tested) + diffForTest(biasUpdate_baseline, biasUpdate_tested)
					+ diffForTest(gradientOut_tested, gradientOut_baseline) + diffForTest(gradientIn_tested, gradientIn_baseline);
		}

		LossFunctionTester::LossFunctionTester(avLossType_t loss_type, std::vector<int> shape, avDataType_t dtype) :
				loss_type(loss_type),
				shape(shape),
				dtype(dtype)
		{
		}
		double LossFunctionTester::getDifferenceLoss() noexcept
		{
			TensorWrapper output(shape, dtype, getDevice());
			TensorWrapper target(shape, dtype, getDevice());
			initForTest(output, 0.0, 0.01, 0.99);
			initForTest(target, 1.0, 0.01, 0.99);

			uint32_t result_baseline[4] = { 0, 0, 0, 0 };
			uint32_t result_tested[4] = { 0, 0, 0, 0 };

			refLossFunction(0, loss_type, output.getRefDescriptor(), output.getRefMemory(), target.getRefDescriptor(), target.getRefMemory(),
					&result_baseline);
#if USE_CPU
			cpuLossFunction(getContextDesc(), loss_type, output.getDescriptor(), output.getMemory(), target.getDescriptor(), target.getMemory(),
					&result_tested);
#elif USE_CUDA
			cudaLossFunction(getContextDesc(), loss_type, output.getDescriptor(), output.getMemory(), target.getDescriptor(), target.getMemory(),
					&result_tested);
#elif USE_OPENCL
			openclLossFunction(getContextDesc(), loss_type, output.getDescriptor(), output.getMemory(), target.getDescriptor(),
					target.getMemory(), &result_tested);
#endif
			getMasterContext().synchronize();
			switch (dtype)
			{
				case AVOCADO_DTYPE_FLOAT32:
					return std::fabs(reinterpret_cast<float*>(result_baseline)[0] - reinterpret_cast<float*>(result_tested)[0]) / output.volume();
				case AVOCADO_DTYPE_FLOAT64:
					return std::fabs(reinterpret_cast<double*>(result_baseline)[0] - reinterpret_cast<double*>(result_tested)[0]) / output.volume();
				default:
					return 1.0;
			}
		}
		double LossFunctionTester::getDifferenceGradient(const void *alpha, const void *beta, bool isFused) noexcept
		{
			TensorWrapper output(shape, dtype, getDevice());
			TensorWrapper target(shape, dtype, getDevice());
			initForTest(output, 0.0, 0.01, 0.99);
			initForTest(target, 1.0, 0.01, 0.99);

			TensorWrapper gradient_baseline(shape, dtype, getDevice());
			TensorWrapper gradient_tested(shape, dtype, getDevice());
			initForTest(gradient_baseline, 0.1);
			initForTest(gradient_tested, 0.1);

			refLossGradient(0, loss_type, alpha, output.getRefDescriptor(), output.getRefMemory(), target.getRefDescriptor(), target.getRefMemory(),
					beta, gradient_baseline.getRefDescriptor(), gradient_baseline.getRefMemory(), isFused);
#if USE_CPU
			cpuLossGradient(getContextDesc(), loss_type, alpha, output.getDescriptor(), output.getMemory(), target.getDescriptor(),
					target.getMemory(), beta, gradient_tested.getDescriptor(), gradient_tested.getMemory(), isFused);
#elif USE_CUDA
			cudaLossGradient(getContextDesc(), loss_type, alpha, output.getDescriptor(), output.getMemory(), target.getDescriptor(), target.getMemory(), beta,
					gradient_tested.getDescriptor(), gradient_tested.getMemory(), isFused);
#elif USE_OPENCL
			openclLossGradient(getContextDesc(), loss_type, alpha, output.getDescriptor(), output.getMemory(), target.getDescriptor(), target.getMemory(),
					beta, gradient_tested.getDescriptor(), gradient_tested.getMemory(), isFused);
#endif
			getMasterContext().synchronize();
			return diffForTest(gradient_baseline, gradient_tested);
		}

		OptimizerTester::OptimizerTester(std::vector<int> shape, avDataType_t dtype) :
				optimizer(getDevice()),
				shape(shape),
				dtype(dtype)
		{
		}
		void OptimizerTester::set(avOptimizerType_t type, double learningRate, const std::array<double, 4> &coefficients,
				const std::array<bool, 4> &flags)
		{
			optimizer.set(type, learningRate, coefficients, flags);
		}
		double OptimizerTester::getDifference(const void *alpha, const void *beta) noexcept
		{
			TensorWrapper gradient(shape, dtype, getDevice());
			initForTest(gradient, 0.0);

			TensorWrapper weights_baseline(shape, dtype, getDevice());
			TensorWrapper weights_tested(shape, dtype, getDevice());
			initForTest(weights_baseline, 0.1);
			initForTest(weights_tested, 0.1);

			int workspace_size = optimizer.getWorkspaceSize(weights_baseline) / dtypeSize(dtype);
			TensorWrapper workspace_baseline( { workspace_size }, dtype, getDevice());
			TensorWrapper workspace_tested( { workspace_size }, dtype, getDevice());

			initForTest(workspace_baseline, 1.0);
			initForTest(workspace_tested, 1.0);
			absForTest(workspace_baseline);
			absForTest(workspace_tested);

			refOptimizerLearn(0, optimizer.getRefDescriptor(), alpha, gradient.getRefDescriptor(), gradient.getRefMemory(), beta,
					weights_baseline.getRefDescriptor(), weights_baseline.getRefMemory(), workspace_baseline.getRefMemory());
#if USE_CPU
			cpuOptimizerLearn(getContextDesc(), optimizer.getDescriptor(), alpha, gradient.getDescriptor(), gradient.getMemory(), beta,
					weights_tested.getDescriptor(), weights_tested.getMemory(), workspace_tested.getMemory());
#elif USE_CUDA
			cudaOptimizerLearn(getContextDesc(), optimizer.getDescriptor(), alpha, gradient.getDescriptor(), gradient.getMemory(), beta,
					weights_tested.getDescriptor(), weights_tested.getMemory(), workspace_tested.getMemory());
#elif USE_OPENCL
			openclOptimizerLearn(getContextDesc(), optimizer.getDescriptor(), alpha, gradient.getDescriptor(), gradient.getMemory(), beta,
					weights_tested.getDescriptor(), weights_tested.getMemory(), workspace_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(weights_baseline, weights_tested) + diffForTest(workspace_baseline, workspace_tested);
		}

		RegularizerTest::RegularizerTest(std::vector<int> shape, avDataType_t dtype) :
				shape(shape),
				dtype(dtype)
		{
		}
		double RegularizerTest::getDifference(const void *scale, const void *offset) noexcept
		{
			TensorWrapper gradient_baseline(shape, dtype, getDevice());
			TensorWrapper gradient_tested(shape, dtype, getDevice());
			initForTest(gradient_baseline, 0.1);
			initForTest(gradient_tested, 0.1);

			TensorWrapper weights(shape, dtype, getDevice());
			initForTest(weights, 1.0);

			uint32_t loss_baseline[4] = { 0, 0, 0, 0 };
			uint32_t loss_tested[4] = { 0, 0, 0, 0 };

			refRegularizerL2(0, gradient_baseline.getRefDescriptor(), gradient_baseline.getRefMemory(), weights.getRefDescriptor(),
					weights.getRefMemory(), scale, offset, loss_baseline);
#if USE_CPU
			cpuRegularizerL2(getContextDesc(), gradient_tested.getDescriptor(), gradient_tested.getMemory(), weights.getDescriptor(),
					weights.getMemory(), scale, offset, loss_tested);
#elif USE_CUDA
			cudaRegularizerL2(getContextDesc(), gradient_tested.getDescriptor(), gradient_tested.getMemory(), weights.getDescriptor(), weights.getMemory(),
					scale, offset, loss_tested);
#elif USE_OPENCL
			openclRegularizerL2(getContextDesc(), gradient_tested.getDescriptor(), gradient_tested.getMemory(), weights.getDescriptor(),
					weights.getMemory(), scale, offset, loss_tested);
#endif
			getMasterContext().synchronize();
			double diff = diffForTest(gradient_baseline, gradient_tested);
			switch (dtype)
			{
				case AVOCADO_DTYPE_FLOAT32:
					diff += std::fabs(reinterpret_cast<float*>(loss_baseline)[0] - reinterpret_cast<float*>(loss_tested)[0]) / weights.volume();
					break;
				case AVOCADO_DTYPE_FLOAT64:
					diff += std::fabs(reinterpret_cast<double*>(loss_baseline)[0] - reinterpret_cast<double*>(loss_tested)[0]) / weights.volume();
					break;
				default:
					diff += 1.0;
					break;
			}
			return diff;
		}

		Im2rowTest::Im2rowTest(std::vector<int> inputShape, std::vector<int> filterShape, avDataType_t dtype) :
				config(getDevice(), inputShape.size() - 2),
				input_shape(inputShape),
				filter_shape(filterShape),
				dtype(dtype)
		{
		}
		void Im2rowTest::set(avConvolutionMode_t mode, const std::array<int, 3> &strides, const std::array<int, 3> &padding,
				const std::array<int, 3> &dilation, int groups, const void *paddingValue)
		{
			config.set(mode, strides, padding, dilation, groups, paddingValue);
		}
		double Im2rowTest::getDifference() noexcept
		{
			TensorWrapper input(input_shape, dtype, getDevice());
			TensorWrapper filter(filter_shape, dtype, getDevice());
			initForTest(input, 1.0);
			std::vector<int> output_shape = config.getOutputShape(input, filter);

			int output_tiles = 1;
			for (size_t i = 0; i < output_shape.size() - 1; i++)
				output_tiles *= output_shape[i];

			int filters_tiles = 1;
			for (size_t i = 1; i < filter_shape.size(); i++)
				filters_tiles *= filter_shape[i];
			std::vector<int> matrix_shape { output_tiles, filters_tiles };

			TensorWrapper matrix_baseline(matrix_shape, dtype, getDevice());
			TensorWrapper matrix_tested(matrix_shape, dtype, getDevice());

			refIm2Row(0, config.getRefDescriptor(), filter.getRefDescriptor(), input.getRefDescriptor(), input.getRefMemory(),
					matrix_baseline.getRefDescriptor(), matrix_baseline.getRefMemory());
#if USE_CPU
			cpuIm2Row(getContextDesc(), config.getDescriptor(), filter.getDescriptor(), input.getDescriptor(), input.getMemory(),
					matrix_tested.getDescriptor(), matrix_tested.getMemory());
#elif USE_CUDA
			cudaIm2Row(getContextDesc(), config.getDescriptor(), filter.getDescriptor(), input.getDescriptor(), input.getMemory(),
					matrix_tested.getDescriptor(), matrix_tested.getMemory());
#elif USE_OPENCL
			openclIm2Row(getContextDesc(), config.getDescriptor(), filter.getDescriptor(), input.getDescriptor(), input.getMemory(),
					matrix_tested.getDescriptor(), matrix_tested.getMemory());
#endif
			getMasterContext().synchronize();
			return diffForTest(matrix_baseline, matrix_tested);
		}

		WinogradTest::WinogradTest(std::vector<int> inputShape, std::vector<int> filterShape, avDataType_t dtype, int transformSize) :
				config(getDevice(), inputShape.size() - 2),
				input_shape(inputShape),
				filter_shape(filterShape),
				dtype(dtype),
				transform_size(transformSize)
		{
			assert(filterShape.back() == inputShape.back());
		}
		void WinogradTest::set(avConvolutionMode_t mode, const std::array<int, 3> &strides, const std::array<int, 3> &padding, int groups,
				const void *paddingValue)
		{
			std::array<int, 3> dilation = { 1, 1, 1 };
			config.set(mode, strides, padding, dilation, groups, paddingValue);
		}
		double WinogradTest::getDifferenceWeight() noexcept
		{
			TensorWrapper weights(filter_shape, dtype, getDevice());
			initForTest(weights, 0.0);
//			weights.setall(1.0f);

			const std::vector<int> matrices_shape = { square(filter_shape[1] + transform_size - 1), filter_shape.front(), filter_shape.back() };
			TensorWrapper matrices_baseline(matrices_shape, dtype, getDevice());
			TensorWrapper matrices_tested(matrices_shape, dtype, getDevice());

			refWinogradWeightTransform(0, config.getRefDescriptor(), transform_size, weights.getRefDescriptor(), weights.getRefMemory(),
					matrices_baseline.getRefDescriptor(), matrices_baseline.getRefMemory());

#if USE_CPU
			cpuWinogradWeightTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), weights.getMemory(),
					matrices_tested.getDescriptor(), matrices_tested.getMemory());
#elif USE_CUDA
			cudaWinogradWeightTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), weights.getMemory(),
					matrices_tested.getDescriptor(), matrices_tested.getMemory());
#elif USE_OPENCL
			openclWinogradWeightTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), weights.getMemory(),
					matrices_tested.getDescriptor(), matrices_tested.getMemory());
#endif
//			for (int i = 0; i < 4; i++)
//			{
//				for (int j = 0; j < 4; j++)
//					std::cout << matrices_baseline.get<float>( { i * 4 + j, 0, 0 }) << " ";
//				std::cout << '\n';
//			}
//			std::cout << "----------------------------------------------------------------\n";
//			for (int i = 0; i < 4; i++)
//			{
//				for (int j = 0; j < 4; j++)
//					std::cout << matrices_tested.get<float>( { i * 4 + j, 0, 0 }) << " ";
//				std::cout << '\n';
//			}
			getMasterContext().synchronize();
			return diffForTest(matrices_baseline, matrices_tested);
		}
		double WinogradTest::getDifferenceInput() noexcept
		{
			TensorWrapper weights(filter_shape, dtype, getDevice());
			TensorWrapper input(input_shape, dtype, getDevice());
			initForTest(input, 0.0);

			const std::vector<int> matrices_shape = winograd_matrices_shape(input_shape, filter_shape, transform_size);
			TensorWrapper matrices_baseline(matrices_shape, dtype, getDevice());
			TensorWrapper matrices_tested(matrices_shape, dtype, getDevice());

//			for (int i = 0; i < 6; i++)
//			{
//				for (int j = 0; j < 6; j++)
//					std::cout << input.get<float16>( { 0, i, j, 0 }) << " ";
//				std::cout << '\n';
//			}
//			std::cout << "----------------------------------------------------------------\n";

			refWinogradInputTransform(0, config.getRefDescriptor(), transform_size, weights.getRefDescriptor(), input.getRefDescriptor(),
					input.getRefMemory(), matrices_baseline.getRefDescriptor(), matrices_baseline.getRefMemory());

#if USE_CPU
			cpuWinogradInputTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), input.getDescriptor(),
					input.getMemory(), matrices_tested.getDescriptor(), matrices_tested.getMemory());
#elif USE_CUDA
			cudaWinogradInputTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), input.getDescriptor(),
					input.getMemory(), matrices_tested.getDescriptor(), matrices_tested.getMemory());
#elif USE_OPENCL
			openclWinogradInputTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), input.getDescriptor(), input.getMemory(),
					matrices_tested.getDescriptor(), matrices_tested.getMemory());
#endif
			getMasterContext().synchronize();
//			for (int a = 0; a < matrices_baseline.dimension(1); a++)
//				for (int b = 0; b < matrices_baseline.dimension(2); b++)
//				{
//					std::cout << a << ", " << b << '\n';
//					for (int i = 0; i < 4; i++)
//					{
//						for (int j = 0; j < 4; j++)
//							std::cout << matrices_baseline.get<float>( { i * 4 + j, a, b }) << " ";
//						std::cout << '\n';
//					}
//					std::cout << "----------------------------------------------------------------\n";
//					for (int i = 0; i < 4; i++)
//					{
//						for (int j = 0; j < 4; j++)
//							std::cout << matrices_tested.get<float>( { i * 4 + j, a, b }) << " ";
//						std::cout << '\n';
//					}
//					std::cout << "\n\n";
//				}
			return diffForTest(matrices_baseline, matrices_tested);
		}
		double WinogradTest::getDifferenceOutput(const void *alpha1, const void *alpha2, const void *beta, bool useBias, bool useExt) noexcept
		{
			TensorWrapper weights(filter_shape, dtype, getDevice());
			avDataType_t bias_dtype = std::max(AVOCADO_DTYPE_FLOAT32, dtype);
			TensorWrapper bias( { filter_shape.front() }, bias_dtype, getDevice());
			TensorWrapper input(input_shape, dtype, getDevice());
			initForTest(bias, 1.0);

			std::vector<int> matrices_shape = winograd_matrices_shape(input_shape, filter_shape, transform_size);
			const std::vector<int> output_shape = config.getOutputShape(input, weights);
			matrices_shape.back() = output_shape.back();

			TensorWrapper matrices(matrices_shape, dtype, getDevice());
			initForTest(matrices, 1.0);

			TensorWrapper ext(output_shape, dtype, getDevice());
			TensorWrapper output_baseline(output_shape, dtype, getDevice());
			TensorWrapper output_tested(output_shape, dtype, getDevice());

			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);
			initForTest(ext, 0.2);

			avMemoryDescriptor_t bias_mem = useBias ? bias.getRefMemory() : AVOCADO_NULL_DESCRIPTOR;
			avMemoryDescriptor_t ext_mem = useExt ? ext.getRefMemory() : AVOCADO_NULL_DESCRIPTOR;

			refWinogradOutputTransform(0, config.getRefDescriptor(), transform_size, weights.getRefDescriptor(), alpha1, matrices.getRefDescriptor(),
					matrices.getRefMemory(), output_baseline.getRefDescriptor(), output_baseline.getRefMemory(), bias.getRefDescriptor(), bias_mem,
					alpha2, ext.getRefDescriptor(), ext_mem, beta, AVOCADO_ACTIVATION_LINEAR);

			bias_mem = useBias ? bias.getMemory() : AVOCADO_NULL_DESCRIPTOR;
			ext_mem = useExt ? ext.getMemory() : AVOCADO_NULL_DESCRIPTOR;
#if USE_CPU
			cpuWinogradOutputTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), alpha1,
					matrices.getDescriptor(), matrices.getMemory(), output_tested.getDescriptor(), output_tested.getMemory(), bias.getDescriptor(),
					bias_mem, alpha2, ext.getDescriptor(), ext_mem, beta, AVOCADO_ACTIVATION_LINEAR);
#elif USE_CUDA
			cudaWinogradOutputTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), alpha1,
					matrices.getDescriptor(), matrices.getMemory(), output_tested.getDescriptor(), output_tested.getMemory(), bias.getDescriptor(),
					bias_mem, alpha2, ext.getDescriptor(), ext_mem, beta, AVOCADO_ACTIVATION_LINEAR);
#elif USE_OPENCL
			openclWinogradOutputTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), alpha1,
					matrices.getDescriptor(), matrices.getMemory(), output_tested.getDescriptor(), output_tested.getMemory(), bias.getDescriptor(),
					bias_mem, alpha2, ext.getDescriptor(), ext_mem, beta, AVOCADO_ACTIVATION_LINEAR);
#endif
//			for (int i = 0; i < 4; i++)
//			{
//				for (int j = 0; j < 4; j++)
//					std::cout << output_baseline.get<float>( { 0, i, j, 0 }) << " ";
//				std::cout << '\n';
//			}
//			std::cout << "----------------------------------------------------------------\n";
//			for (int i = 0; i < 4; i++)
//			{
//				for (int j = 0; j < 4; j++)
//					std::cout << output_tested.get<float>( { 0, i, j, 0 }) << " ";
//				std::cout << '\n';
//			}
			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		double WinogradTest::getDifferenceGradient() noexcept
		{
			TensorWrapper weights(filter_shape, dtype, getDevice());
			TensorWrapper gradient(input_shape, dtype, getDevice());
			initForTest(gradient, 0.0);
			gradient.setall(1.0f);

			const std::vector<int> matrices_shape = winograd_matrices_shape(input_shape, filter_shape, transform_size);
			TensorWrapper matrices_baseline(matrices_shape, dtype, getDevice());
			TensorWrapper matrices_tested(matrices_shape, dtype, getDevice());

			refWinogradGradientTransform(0, config.getRefDescriptor(), transform_size, weights.getRefDescriptor(), gradient.getRefDescriptor(),
					gradient.getRefMemory(), matrices_baseline.getRefDescriptor(), matrices_baseline.getRefMemory());

#if USE_CPU
			cpuWinogradGradientTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), gradient.getDescriptor(),
					gradient.getMemory(), matrices_tested.getDescriptor(), matrices_tested.getMemory());
#elif USE_CUDA
			cudaWinogradGradientTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(), gradient.getDescriptor(),
					gradient.getMemory(), matrices_tested.getDescriptor(), matrices_tested.getMemory());
#elif USE_OPENCL
			openclWinogradGradientTransform(getContextDesc(), config.getDescriptor(), transform_size, weights.getDescriptor(),
					gradient.getDescriptor(), gradient.getMemory(), matrices_tested.getDescriptor(), matrices_tested.getMemory());
#endif
			getMasterContext().synchronize();
//			for (int i = 0; i < 4; i++)
//			{
//				for (int j = 0; j < 4; j++)
//					std::cout << matrices_baseline.get<float>( { i * 4 + j, 0, 0 }) << " ";
//				std::cout << '\n';
//			}
//			std::cout << "----------------------------------------------------------------\n";
//			for (int i = 0; i < 4; i++)
//			{
//				for (int j = 0; j < 4; j++)
//					std::cout << matrices_tested.get<float>( { i * 4 + j, 0, 0 }) << " ";
//				std::cout << '\n';
//			}
			return diffForTest(matrices_baseline, matrices_tested);
		}
		double WinogradTest::getDifferenceUpdate(const void *alpha, const void *beta) noexcept
		{
			TensorWrapper weights(filter_shape, dtype, getDevice());
			initForTest(weights, 0.0);
//			weights.setall(1.0f);

			const std::vector<int> matrices_shape = { square(filter_shape[1] + transform_size - 1), filter_shape.front(), filter_shape.back() };
			TensorWrapper matrices(matrices_shape, dtype, getDevice());
			initForTest(matrices, 1.0);

			TensorWrapper update_baseline(filter_shape, dtype, getDevice());
			TensorWrapper update_tested(filter_shape, dtype, getDevice());

			refWinogradUpdateTransform(0, config.getRefDescriptor(), transform_size, alpha, matrices.getRefDescriptor(), matrices.getRefMemory(),
					beta, update_baseline.getRefDescriptor(), update_baseline.getRefMemory());

#if USE_CPU
			cpuWinogradUpdateTransform(getContextDesc(), config.getDescriptor(), transform_size, alpha, matrices.getDescriptor(),
					matrices.getMemory(), beta, update_tested.getDescriptor(), update_tested.getMemory());
#elif USE_CUDA
			cudaWinogradUpdateTransform(getContextDesc(), config.getDescriptor(), transform_size, alpha, matrices.getDescriptor(), matrices.getMemory(), beta,
					update_tested.getDescriptor(), update_tested.getMemory());
#elif USE_OPENCL
			openclWinogradUpdateTransform(getContextDesc(), config.getDescriptor(), transform_size, alpha, matrices.getDescriptor(), matrices.getMemory(),
					beta, update_tested.getDescriptor(), update_tested.getMemory());
#endif
//			for (int i = 0; i < 6; i++)
//			{
//				for (int j = 0; j < 6; j++)
//					std::cout << matrices_baseline.get<float>( { i * 6 + j, 0, 0 }) << " ";
//				std::cout << '\n';
//			}
//			std::cout << "----------------------------------------------------------------\n";
//			for (int i = 0; i < 6; i++)
//			{
//				for (int j = 0; j < 6; j++)
//					std::cout << matrices_tested.get<float>( { i * 6 + j, 0, 0 }) << " ";
//				std::cout << '\n';
//			}
			getMasterContext().synchronize();
			return diffForTest(update_baseline, update_tested);
		}

		ConvolutionTest::ConvolutionTest(std::vector<int> inputShape, std::vector<int> filterShape, avDataType_t dtype) :
				config(getDevice(), inputShape.size() - 2),
				input_shape(inputShape),
				filter_shape(filterShape),
				dtype(dtype)
		{
		}
		void ConvolutionTest::set(avConvolutionMode_t mode, const std::array<int, 3> &strides, const std::array<int, 3> &padding,
				const std::array<int, 3> &dilation, int groups, const void *paddingValue)
		{
			config.set(mode, strides, padding, dilation, groups, paddingValue);
		}
		double ConvolutionTest::getDifferenceInference(const void *alpha, const void *beta) noexcept
		{
			return 1.0;
		}
		double ConvolutionTest::getDifferenceForward(const void *alpha, const void *beta) noexcept
		{
			TensorWrapper input(input_shape, dtype, getDevice());
			TensorWrapper weights(filter_shape, dtype, getDevice());
			initForTest(input, 0.0);
			initForTest(weights, 1.0);

			std::vector<int> output_shape = config.getOutputShape(input, weights);
			TensorWrapper output_baseline(output_shape, dtype, getDevice());
			TensorWrapper output_tested(output_shape, dtype, getDevice());
			initForTest(output_baseline, 0.1);
			initForTest(output_tested, 0.1);

			av_int64 workspace_size = 0;
//			refGetConvolutionWorkspaceSize(config.getRefDescriptor(), input.getRefDescriptor(), weights.getRefDescriptor(), true, &workspace_size);
			TensorWrapper workspace_baseline( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, getDevice());

			refConvolutionForward(0, config.getRefDescriptor(), alpha, input.getRefDescriptor(), input.getRefMemory(), weights.getRefDescriptor(),
					weights.getRefMemory(), beta, output_baseline.getRefDescriptor(), output_baseline.getRefMemory(),
					workspace_baseline.getRefMemory());

			workspace_size = 0;
//#if USE_CPU
//			cpuGetConvolutionWorkspaceSize(config.getDescriptor(), input.getDescriptor(), weights.getDescriptor(), true, &workspace_size);
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			cpuConvolutionForward(cpuGetDefaultContext(), config.getDescriptor(), alpha, input.getDescriptor(), input.getMemory(),
//					weights.getDescriptor(), weights.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory(),
//					workspace_tested.getMemory());
//#elif USE_CUDA
//			cudaGetConvolutionWorkspaceSize(config.getDescriptor(), input.getDescriptor(), weights.getDescriptor(), &workspace_size);
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			cudaConvolutionForward(cudaGetDefaultContext(), config.getDescriptor(), alpha, input.getDescriptor(), input.getMemory(),
//					weights.getDescriptor(), weights.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory(),
//					workspace_tested.getMemory());
//#elif USE_OPENCL
//			cudaGetConvolutionWorkspaceSize(config.getDescriptor(), input.getDescriptor(), weights.getDescriptor(), &workspace_size);
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			openclConvolutionForward(openclGetDefaultContext(), config.getDescriptor(), alpha, input.getDescriptor(), input.getMemory(),
//					weights.getDescriptor(), weights.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory(),
//					workspace_tested.getMemory());
//#endif
//			getMasterContext().synchronize();
			return diffForTest(output_baseline, output_tested);
		}
		double ConvolutionTest::getDifferenceBackward(const void *alpha, const void *beta) noexcept
		{
			TensorWrapper gradient_prev_baseline(input_shape, dtype, getDevice());
			TensorWrapper gradient_prev_tested(input_shape, dtype, getDevice());
			initForTest(gradient_prev_baseline, 0.1);
			initForTest(gradient_prev_tested, 0.1);

			TensorWrapper weights(filter_shape, dtype, getDevice());
			initForTest(weights, 1.0);

			std::vector<int> gradient_next_shape = config.getOutputShape(gradient_prev_baseline, weights);
			TensorWrapper gradient_next(gradient_next_shape, dtype, getDevice());
			initForTest(gradient_next, 0.0);

			av_int64 workspace_size = 0;
//			refGetConvolutionWorkspaceSize(config.getRefDescriptor(), gradient_prev_baseline.getRefDescriptor(), weights.getRefDescriptor(), false,
//					&workspace_size);
			TensorWrapper workspace_baseline( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, getDevice());

			refConvolutionBackward(0, config.getRefDescriptor(), alpha, gradient_prev_baseline.getRefDescriptor(),
					gradient_prev_baseline.getRefMemory(), weights.getRefDescriptor(), weights.getRefMemory(), beta, gradient_next.getRefDescriptor(),
					gradient_next.getRefMemory(), workspace_baseline.getRefMemory());

			workspace_size = 0;
//#if USE_CPU
//			cpuGetConvolutionWorkspaceSize(config.getDescriptor(), gradient_prev_tested.getDescriptor(), weights.getDescriptor(), false,
//					&workspace_size);
//
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			cpuConvolutionBackward(cpuGetDefaultContext(), config.getDescriptor(), alpha, gradient_prev_tested.getDescriptor(),
//					gradient_prev_tested.getMemory(), weights.getDescriptor(), weights.getMemory(), beta, gradient_next.getDescriptor(),
//					gradient_next.getMemory(), workspace_tested.getMemory());
//#elif USE_CUDA
//			cudaGetConvolutionWorkspaceSize(config.getDescriptor(), input.getDescriptor(), weights.getDescriptor(), &workspace_size);
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			cudaConvolutionForward(cudaGetDefaultContext(), config.getDescriptor(), alpha, input.getDescriptor(), input.getMemory(),
//					weights.getDescriptor(), weights.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory(),
//					workspace_tested.getMemory());
//#elif USE_OPENCL
//			cudaGetConvolutionWorkspaceSize(config.getDescriptor(), input.getDescriptor(), weights.getDescriptor(), &workspace_size);
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			openclConvolutionForward(openclGetDefaultContext(), config.getDescriptor(), alpha, input.getDescriptor(), input.getMemory(),
//					weights.getDescriptor(), weights.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory(),
//					workspace_tested.getMemory());
//#endif
//			getMasterContext().synchronize();
			return diffForTest(gradient_prev_baseline, gradient_prev_tested);
		}
		double ConvolutionTest::getDifferenceUpdate(const void *alpha, const void *beta) noexcept
		{
			TensorWrapper update_baseline(filter_shape, dtype, getDevice());
			TensorWrapper update_tested(filter_shape, dtype, getDevice());
//			initForTest(update_baseline, 0.1);
//			initForTest(update_tested, 0.1);

			TensorWrapper input(input_shape, dtype, getDevice());
//			initForTest(input, 1.0);
			input.setall(1.0);

			std::vector<int> gradient_shape = config.getOutputShape(input, update_baseline);
			TensorWrapper gradient(gradient_shape, dtype, getDevice());
//			initForTest(gradient, 0.0);
			gradient.setall(1.0);

			av_int64 workspace_size = 0;
//			refGetConvolutionWorkspaceSize(config.getRefDescriptor(), input.getRefDescriptor(), update_baseline.getRefDescriptor(), false, &workspace_size);
			TensorWrapper workspace_baseline( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, getDevice());

			refConvolutionUpdate(0, config.getRefDescriptor(), alpha, input.getRefDescriptor(), input.getRefMemory(), gradient.getRefDescriptor(),
					gradient.getRefMemory(), beta, update_baseline.getRefDescriptor(), update_baseline.getRefMemory(),
					workspace_baseline.getRefMemory());

			workspace_size = 0;
//#if USE_CPU
//			cpuGetConvolutionWorkspaceSize(config.getDescriptor(), input.getDescriptor(), update_tested.getDescriptor(), false, &workspace_size);
//			std::cout << "cpu workspace = " << workspace_size << '\n';
//
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			cpuConvolutionUpdate(cpuGetDefaultContext(), config.getDescriptor(), alpha, input.getDescriptor(), input.getMemory(),
//					gradient.getDescriptor(), gradient.getMemory(), beta, update_tested.getDescriptor(), update_tested.getMemory(),
//					workspace_tested.getMemory());
//#elif USE_CUDA
//			cudaGetConvolutionWorkspaceSize(config.getDescriptor(), input.getDescriptor(), weights.getDescriptor(), &workspace_size);
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			cudaConvolutionForward(cudaGetDefaultContext(), config.getDescriptor(), alpha, input.getDescriptor(), input.getMemory(),
//					weights.getDescriptor(), weights.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory(),
//					workspace_tested.getMemory());
//#elif USE_OPENCL
//			cudaGetConvolutionWorkspaceSize(config.getDescriptor(), input.getDescriptor(), weights.getDescriptor(), &workspace_size);
//			TensorWrapper workspace_tested( { static_cast<int>(workspace_size) }, AVOCADO_DTYPE_INT8, device_index);
//
//			openclConvolutionForward(openclGetDefaultContext(), config.getDescriptor(), alpha, input.getDescriptor(), input.getMemory(),
//					weights.getDescriptor(), weights.getMemory(), beta, output_tested.getDescriptor(), output_tested.getMemory(),
//					workspace_tested.getMemory());
//#endif
//			getMasterContext().synchronize();
			printForTest<double>(update_baseline);
			printForTest<double>(update_tested);
			return diffForTest(update_baseline, update_tested);
		}

	} /* namespace backend */
} /* namespace avocado */

