/*
 * backend_descriptors.hpp
 *
 *  Created on: Dec 5, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_
#define AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_

#include <avocado/backend/backend_defs.h>
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
		/* placeholder structures instead of the real ones available in main Avocado library */
		struct float16;
		struct bfloat16;

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

		inline int dataTypeSize(avDataType_t dtype) noexcept
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

#if USE_CUDA
#  define CHECK_CUDA_ERROR(x) if (x != cudaSuccess) throw std::runtime_error("");
#  define CHECK_CUBLAS_STATUS(x) if (x != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("");
#endif

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
					return *(m_pool.at(index));
				}
				const T& get(int index) const noexcept
				{
					return *(m_pool.at(index));
				}

				template<typename ... Args>
				int create(Args &&... args)
				{
					int result;
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
					m_pool.at(result)->create(std::forward<Args>(args)...);
					return result;
				}
				void destroy(int index)
				{
					m_pool.at(index)->destroy();
					m_available_descriptors.push_back(index);
				}
		};

		class MemoryDescriptor
		{
#if USE_OPENCL
#else
				uint8_t *m_data = nullptr;
#endif
				avDeviceIndex_t m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				size_t m_size = 0;
				size_t m_offset = 0;
				bool m_is_owning = false;
			public:
				MemoryDescriptor() = default;
				MemoryDescriptor(const MemoryDescriptor &other) = delete;
				MemoryDescriptor(MemoryDescriptor &&other) :
						m_data(other.m_data),
						m_device_index(other.m_device_index),
						m_size(other.m_size),
						m_offset(other.m_offset),
						m_is_owning(other.m_is_owning)
				{
#if USE_OPENCL
#else
					other.m_data = nullptr;
#endif
					other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
					other.m_offset = 0;
					other.m_size = 0;
					other.m_is_owning = false;
				}
				MemoryDescriptor& operator=(const MemoryDescriptor &other) = delete;
				MemoryDescriptor& operator=(MemoryDescriptor &&other)
				{
#if USE_OPENCL
#endif
					std::swap(this->m_data, other.m_data);
					std::swap(this->m_device_index, other.m_device_index);
					std::swap(this->m_size, other.m_size);
					std::swap(this->m_offset, other.m_offset);
					std::swap(this->m_is_owning, other.m_is_owning);
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
#if USE_OPENCL
#else
					return m_data != nullptr;
#endif
				}
				avSize_t size() const noexcept
				{
					return static_cast<avSize_t>(m_size);
				}

				/**
				 * \brief This method allocates new memory block and sets up the descriptor.
				 */
				void create(avSize_t sizeInBytes, avDeviceIndex_t index = 0)
				{
#if USE_CUDA
					cudaError_t err = cudaSetDevice(index);
					CHECK_CUDA_ERROR(err)
					err = cudaMalloc(&m_data, sizeInBytes);
					CHECK_CUDA_ERROR(err)
#elif USE_OPENCL

#else
					m_data = new uint8_t[sizeInBytes];
#endif
					m_device_index = index;
					m_offset = 0;
					m_size = sizeInBytes;
					m_is_owning = true;
				}
				/**
				 * \brief Creates a non-owning view of another memory block.
				 */
				void create(const MemoryDescriptor &other, size_t size, size_t offset)
				{
					if (other.m_is_owning == false)
						throw std::logic_error("cannot create memory view from non-owning memory descriptor");
					if (other.m_size < offset + size)
						throw std::logic_error("the view would extend beyond the original tensor");
#if USE_OPENCL
#else
					m_data = other.m_data + offset;
#endif
					m_device_index = other.m_device_index;
					m_size = size;
					m_offset = offset;
					m_is_owning = false;
				}
				/**
				 * \brief This method deallocates underlying memory and resets the descriptor.
				 * Calling this method on an already destroyed descriptor has no effect.
				 */
				void destroy()
				{
#if USE_CUDA
					if (m_data != nullptr)
					{
						if (m_is_owning)
						{
							cudaError_t err = cudaFree(m_data);
							CHECK_CUDA_ERROR(err)
						}
						m_data = nullptr;
					}
#elif USE_OPENCL
#else
					if (m_is_owning)
						delete[] m_data;
					m_data = nullptr;
#endif
					m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
					m_size = 0;
					m_offset = 0;
					m_is_owning = false;
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
#else
				template<typename T = void>
				T* data() noexcept
				{
					return reinterpret_cast<T*>(m_data + m_offset);
				}
				template<typename T = void>
				const T* data() const noexcept
				{
					return reinterpret_cast<const T*>(m_data + m_offset);
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
					cudaError_t err = cudaDeviceSynchronize(m_device_index);
					CHECK_CUDA_ERROR(err)
					if(m_handle != nullptr)
					{
						cublasStatus_t status = cublasDestroy_v2(m_handle);
						CHECK_CUBLAS_STATUS(status)
						m_handle = nullptr;
					}

					if (m_stream != nullptr)
					{
						err = cudaStreamDestroy(m_stream);
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

		class TensorDescriptor
		{
				std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> m_dimensions;
				int m_number_of_dimensions = 0;
				avDataType_t m_dtype = AVOCADO_DTYPE_UNKNOWN;
			public:
				TensorDescriptor() = default;
				TensorDescriptor(std::initializer_list<int> dimensions, avDataType_t dtype) :
						m_number_of_dimensions(dimensions.size()),
						m_dtype(dtype)
				{
					m_dimensions.fill(0);
				}

				void create()
				{
				}
				void destroy()
				{
					m_dimensions.fill(0);
					m_number_of_dimensions = 0;
					m_dtype = AVOCADO_DTYPE_UNKNOWN;
				}
				void set(avDataType_t dtype, int nbDims, const int dimensions[])
				{
					if (dimensions == nullptr or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS)
						throw std::invalid_argument("");
					std::memcpy(m_dimensions.data(), dimensions, sizeof(int) * nbDims);
					m_number_of_dimensions = nbDims;
					m_dtype = dtype;
				}
				void get(avDataType_t *dtype, int *nbDims, int dimensions[]) const
				{
					if (dtype != nullptr)
						dtype[0] = m_dtype;
					if (nbDims != nullptr)
						nbDims[0] = m_number_of_dimensions;
					if (dimensions != nullptr)
						std::memcpy(dimensions, m_dimensions.data(), sizeof(int) * m_number_of_dimensions);
				}

				int dimension(int index) const
				{
					return m_dimensions[index];
				}
				int nbDims() const noexcept
				{
					return m_number_of_dimensions;
				}
				avSize_t sizeInBytes() const noexcept
				{
					return dataTypeSize(m_dtype) * this->volume();
				}
				int getIndex(std::initializer_list<int> indices) const noexcept
				{
					assert(nbDims() == static_cast<int>(indices.size()));
					int result = 0;
					int tmp = 1;
					for (int i = indices.size() - 1; i >= 0; i--)
					{
						result += indices.begin()[i] * tmp;
						tmp *= this->dimension(i);
					}
					return result;
				}
				int firstDim() const noexcept
				{
					if (m_number_of_dimensions == 0)
						return 0;
					else
						return m_dimensions[0];
				}
				int lastDim() const noexcept
				{
					if (m_number_of_dimensions == 0)
						return 0;
					else
						return m_dimensions[m_number_of_dimensions - 1];
				}
				int volume() const noexcept
				{
					if (m_number_of_dimensions == 0)
						return 0;
					else
					{
						int result = 1;
						for (int i = 0; i < m_number_of_dimensions; i++)
							result *= m_dimensions[i];
						return result;
					}
				}
				int volumeWithoutFirstDim() const noexcept
				{
					if (m_number_of_dimensions == 0)
						return 0;
					else
					{
						int result = 1;
						for (int i = 1; i < m_number_of_dimensions; i++)
							result *= m_dimensions[i];
						return result;
					}
				}
				int volumeWithoutLastDim() const noexcept
				{
					if (m_number_of_dimensions == 0)
						return 0;
					else
					{
						int result = 1;
						for (int i = 0; i < m_number_of_dimensions - 1; i++)
							result *= m_dimensions[i];
						return result;
					}
				}
				avDataType_t dtype() const noexcept
				{
					return m_dtype;
				}

				bool equalShape(const TensorDescriptor &other) noexcept
				{
					if (m_number_of_dimensions != other.m_number_of_dimensions)
						return false;
					for (int i = 0; i < m_number_of_dimensions; i++)
						if (m_dimensions[i] != other.m_dimensions[i])
							return false;
					return true;
				}
		};

		class ConvolutionDescriptor
		{
			public:
				avConvolutionAlgorithm_t algorithm = AVOCADO_CONVOLUTION_ALGORITHM_AUTO;
				avConvolutionMode_t mode = AVOCADO_CONVOLUTION_MODE;
				int dimensions = 2;
				std::array<int, 3> padding;
				std::array<int, 3> stride;
				std::array<int, 3> dilation;
				uint8_t padding_value[16];
				int groups = 1;

				ConvolutionDescriptor() = default;
				void create()
				{
					algorithm = AVOCADO_CONVOLUTION_ALGORITHM_AUTO;
					padding.fill(0);
					stride.fill(1);
					dilation.fill(1);
					std::memset(padding_value, 0, sizeof(padding_value));
					groups = 1;
				}
				void destroy()
				{
				}

				void set(avConvolutionMode_t mode, int nbDims, const int strides[], const int padding[], const int dilation[], int groups,
						const void *paddingValue)
				{
					if (nbDims < 0 or nbDims > 3)
						throw std::invalid_argument("");
					this->mode = mode;
					dimensions = nbDims;
					if (strides != nullptr)
						std::memcpy(this->stride.data(), strides, sizeof(int) * dimensions);
					if (padding != nullptr)
						std::memcpy(this->padding.data(), padding, sizeof(int) * dimensions);
					if (dilation != nullptr)
						std::memcpy(this->stride.data(), dilation, sizeof(int) * dimensions);

					this->groups = groups;
					if (paddingValue != nullptr)
						std::memcpy(this->padding_value, paddingValue, sizeof(padding_value));
				}
				void get(avConvolutionMode_t *mode, int *nbDims, int strides[], int padding[], int dilation[], int *groups, void *paddingValue) const
				{
					if (mode != nullptr)
						mode[0] = this->mode;
					if (nbDims != nullptr)
						nbDims[0] = dimensions;
					if (strides != nullptr)
						std::memcpy(strides, this->stride.data(), sizeof(int) * dimensions);
					if (padding != nullptr)
						std::memcpy(padding, this->padding.data(), sizeof(int) * dimensions);
					if (dilation != nullptr)
						std::memcpy(dilation, this->dilation.data(), sizeof(int) * dimensions);

					if (groups != nullptr)
						groups[0] = this->groups;
					if (paddingValue != nullptr)
						std::memcpy(paddingValue, this->padding_value, sizeof(padding_value));
				}
		};

		class PoolingDescriptor
		{
			public:
				avPoolingMode_t mode = AVOCADO_POOLING_MAX;
				std::array<int, 3> filter;
				std::array<int, 3> padding;
				std::array<int, 3> stride;

				PoolingDescriptor() = default;
				void create()
				{
					filter.fill(0);
					padding.fill(0);
					stride.fill(1);
				}
				void destroy()
				{
				}
		};

		class OptimizerDescriptor
		{
			public:
				avOptimizerType_t type = AVOCADO_OPTIMIZER_SGD;
				double learning_rate = 0.0;
				std::array<double, 4> coef;
				std::array<bool, 4> flags;

				OptimizerDescriptor() = default;
				void create()
				{
					type = AVOCADO_OPTIMIZER_SGD;
					learning_rate = 0.0;
					coef.fill(0);
					flags.fill(false);
				}
				void destroy()
				{
				}

				void set_sgd(double learningRate, bool useMomentum, bool useNesterov, double beta1)
				{
					this->type = AVOCADO_OPTIMIZER_SGD;
					this->learning_rate = learningRate;
					this->coef[0] = beta1;
					this->flags[0] = useMomentum;
					this->flags[1] = useNesterov;
				}
				void set_adam(double learningRate, double beta1, double beta2)
				{
					this->type = AVOCADO_OPTIMIZER_SGD;
					this->learning_rate = learningRate;
					this->coef[0] = beta1;
					this->coef[1] = beta2;
				}
				void get_type(avOptimizerType_t *type) const
				{
					if (type == nullptr)
						throw std::invalid_argument("");
					type[0] = this->type;
				}
				void get_sgd(double *learningRate, bool *useMomentum, bool *useNesterov, double *beta1) const
				{
					if (learningRate != nullptr)
						learningRate[0] = learning_rate;
					if (beta1 != nullptr)
						beta1[0] = this->coef[0];
					if (useMomentum != nullptr)
						useMomentum[0] = this->flags[0];
					if (useNesterov != nullptr)
						useNesterov[0] = this->flags[1];
				}
				void get_adam(double *learningRate, double *beta1, double *beta2) const
				{
					if (learningRate != nullptr)
						learningRate[0] = learning_rate;
					if (beta1 != nullptr)
						beta1[0] = this->coef[0];
					if (beta2 != nullptr)
						beta2[0] = this->coef[1];
				}
				void get_workspace_size(int *result, const TensorDescriptor &wDesc) const
				{
					if (result == nullptr)
						throw std::invalid_argument("");
					switch (type)
					{
						case AVOCADO_OPTIMIZER_SGD:
							if (flags[0] == true)
								result[0] = wDesc.volume() * dataTypeSize(wDesc.dtype());
							else
								result[0] = 0;
							break;
						case AVOCADO_OPTIMIZER_ADAM:
							result[0] = 2 * wDesc.volume() * dataTypeSize(wDesc.dtype());
							break;
						default:
							result[0] = 0;
					}
				}
		};

		class DropoutDescriptor
		{
			public:
				DropoutDescriptor() = default;
				void create()
				{
				}
				void destroy()
				{
				}
		};

		namespace internal
		{
			template<class T>
			inline DescriptorPool<T>& getPool()
			{
				thread_local DescriptorPool<T> result;
				return result;
			}
#if USE_CUDA

#elif USE_OPENCL

#else
			template<>
			inline DescriptorPool<ContextDescriptor>& getPool()
			{
				thread_local DescriptorPool<ContextDescriptor> result = []()
				{
					DescriptorPool<ContextDescriptor> tmp;
					tmp.create(); // reserve descriptor 0 for default context
					return tmp;
				}();
				return result;
			}
#endif
			template<typename T, typename ... Args>
			inline avStatus_t create(int *result, Args &&... args)
			{
				if (result == nullptr)
					return AVOCADO_STATUS_BAD_PARAM;
				try
				{
					result[0] = getPool<T>().create(std::forward<Args>(args)...);
				} catch (std::exception &e)
				{
					return AVOCADO_STATUS_INTERNAL_ERROR;
				}
				return AVOCADO_STATUS_SUCCESS;
			}
			template<typename T>
			inline avStatus_t destroy(int desc)
			{
				try
				{
					getPool<T>().destroy(desc);
				} catch (std::exception &e)
				{
					return AVOCADO_STATUS_FREE_FAILED;
				}
				return AVOCADO_STATUS_SUCCESS;
			}
		} /* namespace internal */

		inline MemoryDescriptor& getMemory(avMemoryDescriptor_t desc)
		{
			return internal::getPool<MemoryDescriptor>().get(desc);
		}
		inline ContextDescriptor& getContext(avContextDescriptor_t desc)
		{
			return internal::getPool<ContextDescriptor>().get(desc);
		}
		inline TensorDescriptor& getTensor(avTensorDescriptor_t desc)
		{
			return internal::getPool<TensorDescriptor>().get(desc);
		}
		inline ConvolutionDescriptor& getConvolution(avConvolutionDescriptor_t desc)
		{
			return internal::getPool<ConvolutionDescriptor>().get(desc);
		}
		inline PoolingDescriptor& getPooling(avPoolingDescriptor_t desc)
		{
			return internal::getPool<PoolingDescriptor>().get(desc);
		}
		inline OptimizerDescriptor& getOptimizer(avOptimizerDescriptor_t desc)
		{
			return internal::getPool<OptimizerDescriptor>().get(desc);
		}
		inline DropoutDescriptor& getDropout(avDropoutDescriptor_t desc)
		{
			return internal::getPool<DropoutDescriptor>().get(desc);
		}

		template<typename T = void>
		T* getPointer(avMemoryDescriptor_t desc)
		{
			try
			{
				return getMemory(desc).data<T>();
			} catch (std::exception &e)
			{
				return nullptr;
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

		template<typename T>
		void setScalarValue(void *scalar, T x) noexcept
		{
			assert(scalar != nullptr);
			reinterpret_cast<T*>(scalar)[0] = x;
		}

		template<typename T>
		T getScalarValue(const void *scalar) noexcept
		{
			assert(scalar != nullptr);
			return reinterpret_cast<const T*>(scalar)[0];
		}

		template<typename T = float>
		T getAlphaValue(const void *alpha) noexcept
		{
			if (alpha == nullptr)
				return static_cast<T>(1);
			else
				return reinterpret_cast<const T*>(alpha)[0];
		}
		template<typename T = float>
		T getBetaValue(const void *beta) noexcept
		{
			if (beta == nullptr)
				return static_cast<T>(0);
			else
				return reinterpret_cast<const T*>(beta)[0];
		}

		struct BroadcastedDimensions
		{
				avSize_t first;
				avSize_t last;
		};

		/**
		 * Only the right hand side (rhs) operand can be broadcasted into the left hand side (lhs).
		 * The number of dimensions of the rhs tensor must be lower or equal to the lhs tensor.
		 * All k dimensions of the rhs must match the last k dimensions of the lhs.
		 *
		 */
		inline bool isBroadcastPossible(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept
		{
			if (lhs.nbDims() >= rhs.nbDims())
				return false;
			else
			{
				for (int i = 0, k = lhs.nbDims() - rhs.nbDims(); i < rhs.nbDims(); i++, k++)
					if (rhs.dimension(i) != lhs.dimension(k))
						return false;
				return true;
			}
		}
		inline int volume(const BroadcastedDimensions &dims) noexcept
		{
			return dims.first * dims.last;
		}
		inline BroadcastedDimensions getBroadcastDimensions(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept
		{
			assert(isBroadcastPossible(lhs, rhs));
			avSize_t lhs_volume = lhs.volume();
			avSize_t rhs_volume = rhs.volume();
			assert(lhs_volume > 0 && rhs_volume > 0);
			BroadcastedDimensions result { lhs_volume / rhs_volume, rhs_volume };
//			for (int i = 0; i < lhs.length - rhs.length; i++)
//				result.first *= lhs.dim[i];
//			for (int i = lhs.length - rhs.length; i < lhs.length; i++)
//				result.last *= lhs.dim[i];
			return result;
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

	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_ */
