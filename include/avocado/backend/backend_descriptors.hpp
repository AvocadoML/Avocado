/*
 * backend_descriptors.hpp
 *
 *  Created on: Dec 5, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_
#define AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_

#include <avocado/backend/backend_defs_v2.h>

#include <type_traits>
#include <array>
#include <vector>
#include <stack>
#include <algorithm>
#include <complex>
#include <cstring>
#include <cassert>

#if USE_CPU
#  include <memory>
#elif USE_CUDA
#  include <cuda_runtime_api.h>
#  include <cuda_fp16.h>
#  include <cublas_v2.h>
#elif USE_OPENCL
#  include <CL/cl2.hpp>
#else
#  include <memory>
#endif

namespace avocado
{
	namespace backend
	{
#if USE_CUDA
#  define CHECK_CUDA_ERROR(x) if (x != cudaSuccess) throw std::runtime_error("");
#  define CHECK_CUBLAS_STATUS(x) if (x != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("");
#endif

//#ifdef __GNUC__
//#  pragma GCC diagnostic push
//#  pragma GCC diagnostic ignored "-Wunused-function"
//#endif

		template<typename T>
		class DescriptorPool
		{
				std::vector<std::unique_ptr<T>> m_pool;
				std::vector<int> m_available_descriptors;
			public:
				DescriptorPool(size_t initialSize = 10)
				{
					m_pool.reserve(initialSize);
					m_available_descriptors.reserve(initialSize);
				}
				DescriptorPool(const DescriptorPool<T> &other) = delete;
				DescriptorPool(DescriptorPool<T> &&other) = default;
				DescriptorPool& operator=(const DescriptorPool<T> &other) = delete;
				DescriptorPool& operator=(DescriptorPool<T> &&other) = default;
				~DescriptorPool()
				{
					try
					{
						for (size_t i = 0; i < m_pool.size(); i++)
							m_pool[i]->destroy();
					} catch (std::exception &e)
					{
						exit(-1);
					}
				}

				/**
				 * \brief Checks if the passed descriptor is valid.
				 * The descriptor is valid if and only if its index is within the size of m_pool vector and is not in the list of available descriptors.
				 */
				bool isValid(int index) const noexcept
				{
					if (index < 0 or index > static_cast<int>(m_pool.size()))
						return false;
					else
						return std::find(m_available_descriptors.begin(), m_available_descriptors.end(), index) == m_available_descriptors.end();
				}

				T& get(int index) noexcept
				{
					assert(m_pool[index] != nullptr);
					return *(m_pool[index]);
				}
				const T& get(int index) const noexcept
				{
					assert(m_pool[index] != nullptr);
					return *(m_pool[index]);
				}

				template<typename ... Args>
				int create(Args &&... args)
				{
					int result = -1;
					if (m_available_descriptors.size() > 0)
					{
						result = m_available_descriptors.back();
						m_available_descriptors.pop_back();
					}
					else
					{
						m_pool.push_back(std::make_unique<T>());
						result = m_pool.size() - 1;
					}
					m_pool[result]->create(std::forward<Args>(args)...);
					return result;
				}
				void destroy(int index)
				{
					m_pool[index]->destroy();
					m_available_descriptors.push_back(index);
				}
		};

		class MemoryDescriptor
		{
#if USE_CPU
				std::unique_ptr<uint8_t[]> m_data;
#elif USE_CUDA
				void *m_data = nullptr;
#elif USE_OPENCL

#else
				std::unique_ptr<uint8_t[]> m_data;
#endif
				avDeviceIndex_t m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
			public:
				MemoryDescriptor() = default;
				MemoryDescriptor(const MemoryDescriptor &other) = delete;
				MemoryDescriptor(MemoryDescriptor &&other) :
						m_device_index(other.m_device_index)
				{
#if USE_CUDA
					this->m_data = other.m_data;
					other.m_workspace = nullptr;
#endif
					other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				}
				MemoryDescriptor& operator=(const MemoryDescriptor &other) = delete;
				MemoryDescriptor& operator=(MemoryDescriptor &&other)
				{
#if USE_CUDA
					std::swap(this->m_data, other.m_data);
#endif
					std::swap(this->m_device_index, other.m_device_index);
					return *this;
				}
				~MemoryDescriptor()
				{
					try
					{
						destroy();
					} catch (std::exception &e)
					{
						exit(-1);
					}
				}
				explicit operator bool() const noexcept
				{
#if USE_CPU
					return static_cast<bool>(m_data);
#elif USE_CUDA
					return m_data != nullptr;
#elif USE_OPENCL

#else
					return static_cast<bool>(m_data);
#endif
				}
				/**
				 * \brief This method allocates new memory block and sets up the descriptor.
				 */
				void create(avSize_t sizeInBytes, avDeviceIndex_t index = 0)
				{
#if USE_CPU
					m_data = std::make_unique<uint8_t[]>(sizeInBytes);
#elif USE_CUDA
					cudaError_t err = cudaSetDevice(index);
					CHECK_CUDA_ERROR(err)
					err = cudaMalloc(&m_data, sizeInBytes);
					CHECK_CUDA_ERROR(err)
#elif USE_OPENCL

#else
					m_data = std::make_unique<uint8_t[]>(sizeInBytes);
#endif
					m_device_index = index;
				}
				/**
				 * \brief This method deallocates underlying memory and resets the descriptor.
				 * Calling this method on an already destroyed descriptor has no effect.
				 */
				void destroy()
				{
#if USE_CPU
					m_data = nullptr;
#elif USE_CUDA
					if (m_data != nullptr)
					{
						cudaError_t err = cudaFree(m_data);
						CHECK_CUDA_ERROR(err)
						m_data = nullptr;
					}
#elif USE_OPENCL

#else
					m_data = nullptr;
#endif
					m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				}
#if USE_OPENCL
//				cl::Buffer& data(void *ptr) noexcept
//				{
//					return *reinterpret_cast<cl::Buffer*>(ptr);
//				}
//				const cl::Buffer& data(const void *ptr) const noexcept
//				{
//					return *reinterpret_cast<const cl::Buffer*>(ptr);
//				}
#elif USE_CUDA
				template<typename T = void>
				T* data() noexcept
				{
					return reinterpret_cast<T*>(m_data);
				}
				template<typename T = void>
				const T* data() const noexcept
				{
					return reinterpret_cast<const T*>(m_data);
				}
#else
				template<typename T = void>
				T* data() noexcept
				{
					return reinterpret_cast<T*>(m_data.get());
				}
				template<typename T = void>
				const T* data() const noexcept
				{
					return reinterpret_cast<const T*>(m_data.get());
				}
#endif
		};

		class ContextDescriptor
		{
#if USE_CPU
#elif USE_CUDA
				cudaStream_t m_stream = nullptr;
				cublasHandle_t m_handle = nullptr;
#elif USE_OPENCL
#else
#endif
				avDeviceIndex_t m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				MemoryDescriptor m_workspace;
				avSize_t m_workspace_size = 0;
			public:
				ContextDescriptor() = default;
				ContextDescriptor(const ContextDescriptor &other) = delete;
				ContextDescriptor(ContextDescriptor &&other) :
#if USE_CUDA
						m_stream(other.m_stream),
						m_handle(other.m_handle),
#elif USE_OPENCL
#endif
						m_device_index(other.m_device_index),
						m_workspace(std::move(other.m_workspace)),
						m_workspace_size(other.m_workspace_size)
				{
#if USE_CUDA
					other.m_stream = nullptr;
					other.m_handle = nullptr;
#elif USE_OPENCL
#endif
					other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
					other.m_workspace_size = 0;

				}
				ContextDescriptor& operator=(const ContextDescriptor &other) = delete;
				ContextDescriptor& operator=(ContextDescriptor &&other)
				{
#if USE_CPU
#elif USE_CUDA
					std::swap(this->m_stream, other.m_stream);
					std::swap(this->m_handle, other.m_handle);
#elif USE_OPENCL
#endif
					std::swap(this->m_device_index, other.m_device_index);
					std::swap(this->m_workspace, other.m_workspace);
					std::swap(this->m_workspace_size, other.m_workspace_size);
					return *this;
				}
				~ContextDescriptor()
				{
					try
					{
						destroy();
					} catch (std::exception &e)
					{
						exit(-1);
					}
				}

				/**
				 * \brief This method initializes context descriptor.
				 */
#if USE_CPU
				void create()
				{
					m_device_index = 0;
				}

#elif USE_CUDA
				void create(avDeviceIndex_t index, bool useDefaultStream)
				{
					cudaError_t err = cudaSetDevice(index);
					CHECK_CUDA_ERROR(err)
					if(useDefaultStream)
						m_stream = nullptr;
					else
					{
						err = cudaStreamCreate(&m_stream);
						CHECK_CUDA_ERROR(err)
					}

					cublasStatus_t status = cublasCreate_v2(&m_handle);
					CHECK_CUBLAS_STATUS(status)
					status = cublasSetStream_v2(m_handle, m_stream);
					CHECK_CUBLAS_STATUS(status)
					m_device_index = index;
				}
#elif USE_OPENCL
				void create(avDeviceIndex_t index, bool useDefaultCommandQueue)
				{
					m_device_index = index;
				}
#else
				void create()
				{
					m_device_index = 0;
				}
#endif
				/**
				 * \brief This method destroys context and all its resources.
				 * Calling this method on an already destroyed descriptor has no effect.
				 */
				void destroy()
				{
#if USE_CPU
#elif USE_CUDA
					if(m_handle != nullptr)
					{
						cublasStatus_t status = cublasDestroy_v2(m_handle);
						CHECK_CUBLAS_STATUS(status)
						m_handle = nullptr;
					}

					if (m_stream != nullptr)
					{
						cudaError_t err = cudaStreamDestroy(m_stream);
						CHECK_CUDA_ERROR(err)
						m_stream = nullptr;
					}
#elif USE_OPENCL
#else
#endif
					m_workspace.destroy();
					m_workspace_size = 0;
					m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				}
				MemoryDescriptor& getWorkspace()
				{
					if (static_cast<bool>(m_workspace) == false)
					{
						m_workspace_size = 1 << 22;
						m_workspace.create(m_workspace_size, m_device_index); // lazy allocation of 4MB workspace
					}
					return m_workspace;
				}

#if USE_CUDA
				void setDevice() const
				{
					cudaError_t err = cudaSetDevice(m_device_index);
					CHECK_CUDA_ERROR(err)
				}
				cudaStream_t getStream() const noexcept
				{
					return m_stream;
				}
				cublasHandle_t getHandle() const noexcept
				{
					return m_handle;
				}
#endif
		};

		class ShapeDescriptor
		{
				std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> m_dimensions;
				int m_length = 0;
			public:
				ShapeDescriptor() = default;
				ShapeDescriptor(std::initializer_list<int> dimensions) :
						m_length(dimensions.size())
				{
					std::memcpy(m_dimensions.data(), dimensions.begin(), sizeof(int) * dimensions.size());
				}

				void create() noexcept
				{
					m_dimensions.fill(0);
					m_length = 0;
				}
				void destroy() const noexcept
				{
				}
				void set(int nbDims, const int dimensions[])
				{
					if (dimensions == nullptr or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS)
						throw std::invalid_argument("");
					std::memcpy(m_dimensions.data(), dimensions, sizeof(int) * nbDims);
					m_length = nbDims;
				}
				void get(int *nbDims, int dimensions[]) const
				{
					if (dimensions == nullptr or nbDims == nullptr)
						throw std::invalid_argument("");
					std::memcpy(dimensions, m_dimensions.data(), sizeof(int) * m_length);
					nbDims[0] = m_length;
				}
				int firstDim() const noexcept
				{
					if (m_length == 0)
						return 0;
					else
						return m_dimensions[0];
				}
				int lastDim() const noexcept
				{
					if (m_length == 0)
						return 0;
					else
						return m_dimensions[m_length - 1];
				}
				int volume() const noexcept
				{
					if (m_length == 0)
						return 0;
					else
					{
						int result = 1;
						for (int i = 0; i < m_length; i++)
							result *= m_dimensions[i];
						return result;
					}
				}
				int volumeWithoutFirstDim() const noexcept
				{
					if (m_length == 0)
						return 0;
					else
					{
						int result = 1;
						for (int i = 1; i < m_length; i++)
							result *= m_dimensions[i];
						return result;
					}
				}
				int volumeWithoutLastDim() const noexcept
				{
					if (m_length == 0)
						return 0;
					else
					{
						int result = 1;
						for (int i = 0; i < m_length - 1; i++)
							result *= m_dimensions[i];
						return result;
					}
				}

				friend bool operator==(const ShapeDescriptor &lhs, const ShapeDescriptor &rhs) noexcept
				{
					if (lhs.m_length != rhs.m_length)
						return false;
					for (int i = 0; i < lhs.m_length; i++)
						if (lhs.m_dimensions[i] != rhs.m_dimensions[i])
							return false;
					return true;
				}
		};

		class TensorDescriptor
		{
				ShapeDescriptor m_shape;
				avDataType_t m_dtype = AVOCADO_DTYPE_UNKNOWN;
			public:
				TensorDescriptor() = default;
				void create() noexcept
				{
					m_shape.create();
					m_dtype = AVOCADO_DTYPE_UNKNOWN;
				}
				void destroy() const noexcept
				{
					m_shape.destroy();
				}
				void set(avDataType_t dtype, int nbDims, const int dimensions[])
				{
					m_shape.set(nbDims, dimensions);
					m_dtype = dtype;
				}
				void get(avDataType_t *dtype, int *nbDims, int dimensions[]) const
				{
					if (dtype == nullptr)
						throw std::invalid_argument("");
					m_shape.get(nbDims, dimensions);
					dtype[0] = m_dtype;
				}
				int firstDim() const noexcept
				{
					return m_shape.firstDim();
				}
				int lastDim() const noexcept
				{
					return m_shape.lastDim();
				}
				int volume() const noexcept
				{
					return m_shape.volume();
				}
				int volumeWithoutFirstDim() const noexcept
				{
					return m_shape.volumeWithoutFirstDim();
				}
				int volumeWithoutLastDim() const noexcept
				{
					return m_shape.volumeWithoutLastDim();
				}
				avDataType_t dtype() const noexcept
				{
					return m_dtype;
				}
		};

		class ConvolutionDescriptor
		{
			public:
				ConvolutionDescriptor() = default;
		};

		class PoolingDescriptor
		{
			public:
				PoolingDescriptor() = default;
		};

		class OptimizerDescriptor
		{
			public:
				OptimizerDescriptor() = default;
		};

		class DropoutDescriptor
		{
			public:
				DropoutDescriptor() = default;
		};

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

//#if USE_OPENCL
//		static cl::Buffer& data(void *ptr) noexcept
//		{
//			assert(ptr != nullptr);
//			return *reinterpret_cast<cl::Buffer*>(ptr);
//		}
//		static const cl::Buffer& data(const void *ptr) noexcept
//		{
//			assert(ptr != nullptr);
//			return *reinterpret_cast<const cl::Buffer*>(ptr);
//		}
//		static cl::Buffer& data(TensorDescriptor *desc) noexcept
//		{
//			assert(desc != nullptr);
//			return data(desc->data);
//		}
//		static const cl::Buffer& data(const TensorDescriptor *desc) noexcept
//		{
//			assert(desc != nullptr);
//			return constData(desc->data);
//		}
//#else
//		template<typename T>
//		T* data(void *ptr) noexcept
//		{
//			return reinterpret_cast<T*>(ptr);
//		}
//		template<typename T>
//		const T* data(const void *ptr) noexcept
//		{
//			return reinterpret_cast<const T*>(ptr);
//		}
//		template<typename T>
//		T* data(TensorDescriptor *desc) noexcept
//		{
//			assert(desc != nullptr);
//			return data<T>(desc->data);
//		}
//		template<typename T>
//		const T* data(const TensorDescriptor *desc) noexcept
//		{
//			assert(desc != nullptr);
//			return data<T>(desc->data);
//		}
//#endif /* USE_OPENCL */

//		/**
//		 * \brief Returns number of dimensions of a tensor descriptor.
//		 */
//		static int numberOfDimensions(const TensorDescriptor *desc) noexcept
//		{
//			assert(desc != nullptr);
//			return desc->shape.length;
//		}
//		/**
//		 * \brief Returns specific dimension of a tensor descriptor.
//		 */
//		static int dimension(const TensorDescriptor *desc, int index) noexcept
//		{
//			assert(desc != nullptr);
//			assert(index >= 0 && index < desc->shape.length);
//			return desc->shape.dim[index];
//		}
//
//		/**
//		 * \brief Creates scalar descriptor from given value.
//		 */
//		template<typename T>
//		ScalarDescriptor createScalarDescriptor(T value) noexcept
//		{
//			ScalarDescriptor result;
//			std::memcpy(result.data, &value, sizeof(T));
//			result.dtype = typeOf<T>();
//			return result;
//		}
//		/**
//		 * \brief Returns value stored in scalar descriptor.
//		 */
//		template<typename T>
//		T getScalarValue(const ScalarDescriptor *scalar) noexcept
//		{
//			assert(scalar != nullptr);
//			assert(scalar->dtype == typeOf<T>());
//			T result;
//			std::memcpy(&result, scalar->data, sizeof(T));
//			return result;
//		}
//		static double getDoubleValue(const ScalarDescriptor *scalar) noexcept
//		{
//			switch (scalar->dtype)
//			{
//				default:
//				case AVOCADO_DTYPE_UNKNOWN:
//					return 0.0;
//				case AVOCADO_DTYPE_UINT8:
//					return static_cast<double>(getScalarValue<uint8_t>(scalar));
//				case AVOCADO_DTYPE_INT8:
//					return static_cast<double>(getScalarValue<int8_t>(scalar));
//				case AVOCADO_DTYPE_INT16:
//					return static_cast<double>(getScalarValue<int16_t>(scalar));
//				case AVOCADO_DTYPE_INT32:
//					return static_cast<double>(getScalarValue<int32_t>(scalar));
//				case AVOCADO_DTYPE_INT64:
//					return static_cast<double>(getScalarValue<int64_t>(scalar));
//				case AVOCADO_DTYPE_FLOAT16:
//					return 0.0;
//				case AVOCADO_DTYPE_BFLOAT16:
//					return 0.0;
//				case AVOCADO_DTYPE_FLOAT32:
//					return getScalarValue<float>(scalar);
//				case AVOCADO_DTYPE_FLOAT64:
//					return getScalarValue<double>(scalar);
//				case AVOCADO_DTYPE_COMPLEX32:
//					return real(getScalarValue<std::complex<float>>(scalar));
//				case AVOCADO_DTYPE_COMPLEX64:
//					return real(getScalarValue<std::complex<double>>(scalar));
//			}
//		}
//		template<typename T>
//		void setScalarValue(ScalarDescriptor *scalar, T value) noexcept
//		{
//			assert(scalar != nullptr);
//			std::memcpy(scalar->data, &value, sizeof(T));
//			scalar->dtype = typeOf<T>();
//		}
//
//		template<typename T = float>
//		T getAlphaValue(const ScalarDescriptor *alpha) noexcept
//		{
//			if (alpha == nullptr)
//				return static_cast<T>(1);
//			else
//			{
//				assert(alpha->dtype == typeOf<T>());
//				return getDoubleValue(alpha);
//			}
//		}
//		template<typename T = float>
//		T getBetaValue(const ScalarDescriptor *beta) noexcept
//		{
//			if (beta == nullptr)
//				return static_cast<T>(0);
//			else
//			{
//				assert(beta->dtype == typeOf<T>());
//				return getDoubleValue(beta);
//			}
//		}
//
//		static ShapeDescriptor createShapeDescriptor(std::initializer_list<int> dimensions) noexcept
//		{
//			ShapeDescriptor result;
//			std::memcpy(result.dim, dimensions.begin(), sizeof(int) * dimensions.size());
//			result.length = dimensions.size();
//			return result;
//		}
//		static bool areEqual(const ShapeDescriptor &lhs, const ShapeDescriptor &rhs)
//		{
//			if (lhs.length != rhs.length)
//				return false;
//			for (int i = 0; i < lhs.length; i++)
//				if (lhs.dim[i] != rhs.dim[i])
//					return false;
//			return true;
//		}
//		static int firstDim(const ShapeDescriptor &shape) noexcept
//		{
//			if (shape.length == 0)
//				return 0;
//			else
//				return shape.dim[0];
//		}
//		static int lastDim(const ShapeDescriptor &shape) noexcept
//		{
//			if (shape.length == 0)
//				return 0;
//			else
//				return shape.dim[shape.length - 1];
//		}
//		static int volume(const ShapeDescriptor &shape) noexcept
//		{
//			if (shape.length == 0)
//				return 0;
//			else
//			{
//				int result = 1;
//				for (int i = 0; i < shape.length; i++)
//					result *= shape.dim[i];
//				return result;
//			}
//		}
//		static int volumeWithoutFirstDim(const ShapeDescriptor &shape) noexcept
//		{
//			if (shape.length == 0)
//				return 0;
//			else
//			{
//				int result = 1;
//				for (int i = 1; i < shape.length; i++)
//					result *= shape.dim[i];
//				return result;
//			}
//		}
//		static int volumeWithoutLastDim(const ShapeDescriptor &shape) noexcept
//		{
//			if (shape.length == 0)
//				return 0;
//			else
//			{
//				int result = 1;
//				for (int i = 0; i < shape.length - 1; i++)
//					result *= shape.dim[i];
//				return result;
//			}
//		}
//
//		static int firstDim(const TensorDescriptor *tensor) noexcept
//		{
//			assert(tensor != nullptr);
//			return firstDim(tensor->shape);
//		}
//		static int lastDim(const TensorDescriptor *tensor) noexcept
//		{
//			assert(tensor != nullptr);
//			return lastDim(tensor->shape);
//		}
//		static int volume(const TensorDescriptor *tensor) noexcept
//		{
//			assert(tensor != nullptr);
//			return volume(tensor->shape);
//		}
//		static int volumeWithoutFirstDim(const TensorDescriptor *tensor) noexcept
//		{
//			assert(tensor != nullptr);
//			return volumeWithoutFirstDim(tensor->shape);
//		}
//		static int volumeWithoutLastDim(const TensorDescriptor *tensor) noexcept
//		{
//			assert(tensor != nullptr);
//			return volumeWithoutLastDim(tensor->shape);
//		}
//
//		/**
//		 * Only the right hand side (rhs) operand can be broadcasted into the left hand side (lhs).
//		 * The number of dimensions of the rhs tensor must be lower or equal to the lhs tensor.
//		 * All k dimensions of the rhs must match the last k dimensions of the lhs.
//		 *
//		 */
//		static bool isBroadcastPossible(const ShapeDescriptor &lhs, const ShapeDescriptor &rhs) noexcept
//		{
//			if (lhs.length >= rhs.length)
//				return false;
//			else
//			{
//				for (int i = 0, k = lhs.length - rhs.length; i < rhs.length; i++, k++)
//					if (rhs.dim[i] != lhs.dim[k])
//						return false;
//				return true;
//			}
//		}
//		static int volume(const BroadcastedDimensions &dims) noexcept
//		{
//			return dims.first * dims.last;
//		}
//		static BroadcastedDimensions getBroadcastDimensions(const ShapeDescriptor &lhs, const ShapeDescriptor &rhs) noexcept
//		{
//			assert(isBroadcastPossible(lhs, rhs));
//			avSize_t lhs_volume = volume(lhs);
//			avSize_t rhs_volume = volume(rhs);
//			assert(lhs_volume > 0 && rhs_volume > 0);
//			BroadcastedDimensions result { lhs_volume / rhs_volume, rhs_volume };
////			for (int i = 0; i < lhs.length - rhs.length; i++)
////				result.first *= lhs.dim[i];
////			for (int i = lhs.length - rhs.length; i < lhs.length; i++)
////				result.last *= lhs.dim[i];
//			return result;
//		}
//		static BroadcastedDimensions getBroadcastDimensions(const TensorDescriptor *lhs, const TensorDescriptor *rhs) noexcept
//		{
//			return getBroadcastDimensions(lhs->shape, rhs->shape);
//		}

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

//#ifdef __GNUC__
//#  pragma GCC diagnostic pop
//#endif

	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_ */
