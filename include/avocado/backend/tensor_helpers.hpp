/*
 * tensor_helpers.hpp
 *
 *  Created on: Jul 30, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_BACKEND_TENSOR_HELPERS_HPP_
#define AVOCADO_BACKEND_TENSOR_HELPERS_HPP_

#include <avocado/backend/backend_defs.h>
#include <type_traits>
#include <complex>
#include <cstring>
#include <cassert>

#if USE_OPENCL
#  include <CL/cl2.hpp>
#endif
#if USE_CUDA
#  include <cuda_fp16.h>
#endif

namespace avocado
{
	namespace backend
	{
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-function"
#endif

		/* placeholder structures instead of the real ones available in main Avocado library */
		struct float16;
		struct bfloat16;

		struct BroadcastedDimensions
		{
			avSize_t first;
			avSize_t last;
		};

		template<typename T>
		avDataType_t typeOf() noexcept
		{
			avDataType_t result = AVOCADO_DTYPE_UNKNOWN;
			if (std::is_same<T, uint8_t>::value)
				result = AVOCADO_DTYPE_UINT8;
			if (std::is_same<T, int8_t>::value)
				result = AVOCADO_DTYPE_INT8;
			if (std::is_same<T, int16_t>::value)
				result = AVOCADO_DTYPE_INT16;
			if (std::is_same<T, int32_t>::value)
				result = AVOCADO_DTYPE_INT32;
			if (std::is_same<T, int64_t>::value)
				result = AVOCADO_DTYPE_INT64;
#if USE_CUDA
			if (std::is_same<T, float16>::value or std::is_same<T, half>::value)
#else
			if (std::is_same<T, float16>::value)
#endif
				result = AVOCADO_DTYPE_FLOAT16;
			if (std::is_same<T, bfloat16>::value)
				result = AVOCADO_DTYPE_BFLOAT16;
			if (std::is_same<T, float>::value)
				result = AVOCADO_DTYPE_FLOAT32;
			if (std::is_same<T, double>::value)
				result = AVOCADO_DTYPE_FLOAT64;
			if (std::is_same<T, std::complex<float>>::value)
				result = AVOCADO_DTYPE_COMPLEX32;
			if (std::is_same<T, std::complex<double>>::value)
				result = AVOCADO_DTYPE_COMPLEX64;
			return result;
		}

		static int dataTypeSize(avDataType_t dtype) noexcept
		{
			switch (dtype)
			{
				default:
				case AVOCADO_DTYPE_UNKNOWN:
					return 0;
				case AVOCADO_DTYPE_UINT8:
					return 1;
				case AVOCADO_DTYPE_INT8:
					return 1;
				case AVOCADO_DTYPE_INT16:
					return 2;
				case AVOCADO_DTYPE_INT32:
					return 4;
				case AVOCADO_DTYPE_INT64:
					return 8;
				case AVOCADO_DTYPE_FLOAT16:
					return 2;
				case AVOCADO_DTYPE_BFLOAT16:
					return 2;
				case AVOCADO_DTYPE_FLOAT32:
					return 4;
				case AVOCADO_DTYPE_FLOAT64:
					return 8;
				case AVOCADO_DTYPE_COMPLEX32:
					return 8;
				case AVOCADO_DTYPE_COMPLEX64:
					return 16;
			}
		}

#if USE_OPENCL
		static cl::Buffer& data(void *ptr) noexcept
		{
			assert(ptr != nullptr);
			return *reinterpret_cast<cl::Buffer*>(ptr);
		}
		static const cl::Buffer& data(const void *ptr) noexcept
		{
			assert(ptr != nullptr);
			return *reinterpret_cast<const cl::Buffer*>(ptr);
		}
		static cl::Buffer& data(TensorDescriptor *desc) noexcept
		{
			assert(desc != nullptr);
			return data(desc->data);
		}
		static const cl::Buffer& data(const TensorDescriptor *desc) noexcept
		{
			assert(desc != nullptr);
			return constData(desc->data);
		}
#else
		template<typename T>
		T* data(void *ptr) noexcept
		{
			return reinterpret_cast<T*>(ptr);
		}
		template<typename T>
		const T* data(const void *ptr) noexcept
		{
			return reinterpret_cast<const T*>(ptr);
		}
		template<typename T>
		T* data(TensorDescriptor *desc) noexcept
		{
			assert(desc != nullptr);
			return data<T>(desc->data);
		}
		template<typename T>
		const T* data(const TensorDescriptor *desc) noexcept
		{
			assert(desc != nullptr);
			return data<T>(desc->data);
		}
#endif /* USE_OPENCL */

		/**
		 * \brief Returns number of dimensions of a tensor descriptor.
		 */
		static int numberOfDimensions(const TensorDescriptor *desc) noexcept
		{
			assert(desc != nullptr);
			return desc->shape.length;
		}
		/**
		 * \brief Returns specific dimension of a tensor descriptor.
		 */
		static int dimension(const TensorDescriptor *desc, int index) noexcept
		{
			assert(desc != nullptr);
			assert(index >= 0 && index < desc->shape.length);
			return desc->shape.dim[index];
		}

		/**
		 * \brief Creates scalar descriptor from given value.
		 */
		template<typename T>
		ScalarDescriptor createScalarDescriptor(T value) noexcept
		{
			ScalarDescriptor result;
			std::memcpy(result.data, &value, sizeof(T));
			result.dtype = typeOf<T>();
			return result;
		}
		/**
		 * \brief Returns value stored in scalar descriptor.
		 */
		template<typename T>
		T getScalarValue(const ScalarDescriptor *scalar) noexcept
		{
			assert(scalar != nullptr);
			assert(scalar->dtype == typeOf<T>());
			T result;
			std::memcpy(&result, scalar->data, sizeof(T));
			return result;
		}
		static double getDoubleValue(const ScalarDescriptor *scalar) noexcept
		{
			switch (scalar->dtype)
			{
				default:
				case AVOCADO_DTYPE_UNKNOWN:
					return 0.0;
				case AVOCADO_DTYPE_UINT8:
					return static_cast<double>(getScalarValue<uint8_t>(scalar));
				case AVOCADO_DTYPE_INT8:
					return static_cast<double>(getScalarValue<int8_t>(scalar));
				case AVOCADO_DTYPE_INT16:
					return static_cast<double>(getScalarValue<int16_t>(scalar));
				case AVOCADO_DTYPE_INT32:
					return static_cast<double>(getScalarValue<int32_t>(scalar));
				case AVOCADO_DTYPE_INT64:
					return static_cast<double>(getScalarValue<int64_t>(scalar));
				case AVOCADO_DTYPE_FLOAT16:
					return 0.0;
				case AVOCADO_DTYPE_BFLOAT16:
					return 0.0;
				case AVOCADO_DTYPE_FLOAT32:
					return getScalarValue<float>(scalar);
				case AVOCADO_DTYPE_FLOAT64:
					return getScalarValue<double>(scalar);
				case AVOCADO_DTYPE_COMPLEX32:
					return real(getScalarValue<std::complex<float>>(scalar));
				case AVOCADO_DTYPE_COMPLEX64:
					return real(getScalarValue<std::complex<double>>(scalar));
			}
		}
		template<typename T>
		void setScalarValue(ScalarDescriptor *scalar, T value) noexcept
		{
			assert(scalar != nullptr);
			std::memcpy(scalar->data, &value, sizeof(T));
			scalar->dtype = typeOf<T>();
		}

		template<typename T = float>
		T getAlphaValue(const ScalarDescriptor *alpha) noexcept
		{
			if (alpha == nullptr)
				return static_cast<T>(1);
			else
			{
				assert(alpha->dtype == typeOf<T>());
				return getDoubleValue(alpha);
			}
		}
		template<typename T = float>
		T getBetaValue(const ScalarDescriptor *beta) noexcept
		{
			if (beta == nullptr)
				return static_cast<T>(0);
			else
			{
				assert(beta->dtype == typeOf<T>());
				return getDoubleValue(beta);
			}
		}

		static ShapeDescriptor createShapeDescriptor(std::initializer_list<int> dimensions) noexcept
		{
			ShapeDescriptor result;
			std::memcpy(result.dim, dimensions.begin(), sizeof(int) * dimensions.size());
			result.length = dimensions.size();
			return result;
		}
		static bool areEqual(const ShapeDescriptor &lhs, const ShapeDescriptor &rhs)
		{
			if (lhs.length != rhs.length)
				return false;
			for (int i = 0; i < lhs.length; i++)
				if (lhs.dim[i] != rhs.dim[i])
					return false;
			return true;
		}
		static int firstDim(const ShapeDescriptor &shape) noexcept
		{
			if (shape.length == 0)
				return 0;
			else
				return shape.dim[0];
		}
		static int lastDim(const ShapeDescriptor &shape) noexcept
		{
			if (shape.length == 0)
				return 0;
			else
				return shape.dim[shape.length - 1];
		}
		static int volume(const ShapeDescriptor &shape) noexcept
		{
			if (shape.length == 0)
				return 0;
			else
			{
				int result = 1;
				for (int i = 0; i < shape.length; i++)
					result *= shape.dim[i];
				return result;
			}
		}
		static int volumeWithoutFirstDim(const ShapeDescriptor &shape) noexcept
		{
			if (shape.length == 0)
				return 0;
			else
			{
				int result = 1;
				for (int i = 1; i < shape.length; i++)
					result *= shape.dim[i];
				return result;
			}
		}
		static int volumeWithoutLastDim(const ShapeDescriptor &shape) noexcept
		{
			if (shape.length == 0)
				return 0;
			else
			{
				int result = 1;
				for (int i = 0; i < shape.length - 1; i++)
					result *= shape.dim[i];
				return result;
			}
		}

		static int firstDim(const TensorDescriptor *tensor) noexcept
		{
			assert(tensor != nullptr);
			return firstDim(tensor->shape);
		}
		static int lastDim(const TensorDescriptor *tensor) noexcept
		{
			assert(tensor != nullptr);
			return lastDim(tensor->shape);
		}
		static int volume(const TensorDescriptor *tensor) noexcept
		{
			assert(tensor != nullptr);
			return volume(tensor->shape);
		}
		static int volumeWithoutFirstDim(const TensorDescriptor *tensor) noexcept
		{
			assert(tensor != nullptr);
			return volumeWithoutFirstDim(tensor->shape);
		}
		static int volumeWithoutLastDim(const TensorDescriptor *tensor) noexcept
		{
			assert(tensor != nullptr);
			return volumeWithoutLastDim(tensor->shape);
		}

		/**
		 * Only the right hand side (rhs) operand can be broadcasted into the left hand side (lhs).
		 * The number of dimensions of the rhs tensor must be lower or equal to the lhs tensor.
		 * All k dimensions of the rhs must match the last k dimensions of the lhs.
		 *
		 */
		static bool isBroadcastPossible(const ShapeDescriptor &lhs, const ShapeDescriptor &rhs) noexcept
		{
			if (lhs.length >= rhs.length)
				return false;
			else
			{
				for (int i = 0, k = lhs.length - rhs.length; i < rhs.length; i++, k++)
					if (rhs.dim[i] != lhs.dim[k])
						return false;
				return true;
			}
		}
		static int volume(const BroadcastedDimensions &dims) noexcept
		{
			return dims.first * dims.last;
		}
		static BroadcastedDimensions getBroadcastDimensions(const ShapeDescriptor &lhs, const ShapeDescriptor &rhs) noexcept
		{
			assert(isBroadcastPossible(lhs, rhs));
			avSize_t lhs_volume = volume(lhs);
			avSize_t rhs_volume = volume(rhs);
			assert(lhs_volume > 0 && rhs_volume > 0);
			BroadcastedDimensions result { lhs_volume / rhs_volume, rhs_volume };
//			for (int i = 0; i < lhs.length - rhs.length; i++)
//				result.first *= lhs.dim[i];
//			for (int i = lhs.length - rhs.length; i < lhs.length; i++)
//				result.last *= lhs.dim[i];
			return result;
		}
		static BroadcastedDimensions getBroadcastDimensions(const TensorDescriptor *lhs, const TensorDescriptor *rhs) noexcept
		{
			return getBroadcastDimensions(lhs->shape, rhs->shape);
		}

		template<class T, class U>
		bool same_type(const T *lhs, const U *rhs)
		{
			return lhs->dtype == rhs->dtype;
		}
		template<class T, class U, class ... ARGS>
		bool same_type(const T *lhs, const U *rhs, const ARGS *... args)
		{
			if (lhs->dtype == rhs->dtype)
				return same_type(lhs, args...);
			else
				return false;
		}

		template<class T, class U>
		bool same_shape(const T *lhs, const U *rhs)
		{
			return areEqual(lhs->shape, rhs->shape);
		}
		template<class T, class U, class ... ARGS>
		bool same_shape(const T *lhs, const U *rhs, const ARGS *... args)
		{
			if (areEqual(lhs->shape, rhs->shape))
				return same_shape(lhs, args...);
			else
				return false;
		}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_TENSOR_HELPERS_HPP_ */

