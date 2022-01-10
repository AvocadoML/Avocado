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
#include <memory>
#include <cstring>
#include <cassert>

#if USE_CPU
#elif USE_CUDA
#  include <cuda_runtime_api.h>
#  include <cuda_fp16.h>
#  include <cublas_v2.h>
#elif USE_OPENCL
#  include <CL/cl2.hpp>
#else
#endif

namespace avocado
{
	namespace backend
	{
		inline avDeviceType_t get_device_type(av_int64 descriptor) noexcept
		{
			const av_int64 device_type_mask = 0xFF00000000000000ull;
			return static_cast<avDeviceType_t>((descriptor & device_type_mask) >> 56ull);
		}
		inline int get_descriptor_type(av_int64 descriptor) noexcept
		{
			const av_int64 descriptor_type_mask = 0x00FF000000000000ull;
			return static_cast<int>((descriptor & descriptor_type_mask) >> 48ull);
		}
		inline avDeviceIndex_t get_device_index(av_int64 descriptor) noexcept
		{
			const av_int64 device_index_mask = 0x0000FFFF00000000ull;
			return static_cast<avDeviceIndex_t>((descriptor & device_index_mask) >> 32ull);
		}
		inline int get_descriptor_index(av_int64 descriptor) noexcept
		{
			const av_int64 descriptor_index_mask = 0x00000000FFFFFFFFull;
			return static_cast<int>(descriptor & descriptor_index_mask);
		}

		inline avDeviceType_t get_current_device_type() noexcept
		{
#if USE_CUDA
			return AVOCADO_DEVICE_CUDA;
#elif USE_OPENCL
			return AVOCADO_DEVICE_OPENCL;
#else
			return AVOCADO_DEVICE_CPU;
#endif
		}
		inline avDeviceIndex_t get_current_device_index() noexcept
		{
#if USE_CUDA
			return AVOCADO_DEVICE_CUDA;
#elif USE_OPENCL
			return AVOCADO_DEVICE_OPENCL;
#else
			return AVOCADO_DEVICE_CPU;
#endif
		}

		template<class T>
		constexpr av_int64 create_descriptor_of_type(int index)
		{
			return (static_cast<av_int64>(get_current_device_type()) << 56ull) | (T::descriptor_type << 32ull) | static_cast<av_int64>(index);
		}

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
						if (not m_pool.empty())
							std::cout << "There are " << m_pool.size() << " not destroyed instances of " << T::className() << '\n';
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
				bool isValid(int64_t index) const noexcept
				{
					if (get_current_device_type() != get_device_type(index))
						return false;
					if (T::descriptor_type != get_descriptor_type(index))
						return false;

					int object_index = get_descriptor_index(index);
					if (object_index < 0 or object_index > static_cast<int>(m_pool.size()))
						return false;
					return std::find(m_available_descriptors.begin(), m_available_descriptors.end(), object_index) == m_available_descriptors.end();
				}

				T& get(av_int64 index)
				{
					if (not isValid(index))
						throw std::logic_error("invalid descriptor");
					return *(m_pool.at(get_descriptor_index(index)));
				}
				const T& get(int64_t index) const
				{
					if (not isValid(index))
						throw std::logic_error("invalid descriptor");
					return *(m_pool.at(get_descriptor_index(index)));
				}

				template<typename ... Args>
				av_int64 create(Args &&... args)
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
					return create_descriptor_of_type<T>(result);
				}
				void destroy(av_int64 index)
				{
					if (not isValid(index))
						throw std::logic_error("invalid descriptor");

					int object_index = get_descriptor_index(index);
					m_pool.at(object_index)->destroy();
					m_available_descriptors.push_back(object_index);
				}
		};

		class MemoryDescriptor
		{
#if USE_OPENCL
#else
				uint8_t *m_data = nullptr;
#endif
				avDeviceIndex_t m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				int64_t m_size = 0;
				int64_t m_offset = 0;
				bool m_is_owning = false;
			public:
				static constexpr av_int64 descriptor_type = 1;

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
				int64_t size() const noexcept
				{
					return m_size;
				}
				avDeviceIndex_t device() const noexcept
				{
					return m_device_index;
				}
				static std::string className()
				{
					return "MemoryDescriptor";
				}

				/**
				 * \brief This method allocates new memory block and sets up the descriptor.
				 */
#if USE_CUDA or USE_OPENCL
				void create(avDeviceIndex_t index, avSize_t sizeInBytes)
				{
#  if USE_CUDA
					cudaError_t err = cudaSetDevice(index);
					CHECK_CUDA_ERROR(err)
					err = cudaMalloc(reinterpret_cast<void**>(&m_data), sizeInBytes);
					CHECK_CUDA_ERROR(err)
#  else /* USE_OPENCL */

#  endif
					m_device_index = index;
					m_offset = 0;
					m_size = sizeInBytes;
					m_is_owning = true;
				}
#else
				void create(int64_t sizeInBytes)
				{
					m_data = new uint8_t[sizeInBytes];
					m_device_index = 0;
					m_offset = 0;
					m_size = sizeInBytes;
					m_is_owning = true;
				}
#endif
				/**
				 * \brief Creates a non-owning view of another memory block.
				 */
				void create(const MemoryDescriptor &other, int64_t size, int64_t offset)
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
//			cl::Buffer& data(void *ptr) noexcept
//			{
//				return *reinterpret_cast<cl::Buffer*>(ptr);
//			}
//			const cl::Buffer& data(const void *ptr) const noexcept
//			{
//				return *reinterpret_cast<const cl::Buffer*>(ptr);
//			}
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
				static constexpr av_int64 descriptor_type = 2;

				ContextDescriptor() = default;
				ContextDescriptor(const ContextDescriptor &other) = delete;
				ContextDescriptor(ContextDescriptor &&other) :
#if USE_CUDA
							m_stream(other.m_stream), m_handle(other.m_handle),
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
				static std::string className()
				{
					return "ContextDescriptor";
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
				void create(avDeviceIndex_t index, bool useDefaultStream = false)
				{
					cudaError_t err = cudaSetDevice(index);
					CHECK_CUDA_ERROR(err)
					if (useDefaultStream)
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
				cudaError_t err = cudaDeviceSynchronize();
				CHECK_CUDA_ERROR(err)
				if (m_handle != nullptr)
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
						m_workspace_size = 1 << 23;
#if USE_CUDA or USE_OPENCL
					m_workspace.create(m_workspace_size, m_device_index); // lazy allocation of 8MB workspace
#else
						m_workspace.create(m_workspace_size); // lazy allocation of 8MB workspace
#endif
					}
					return m_workspace;
				}

#if USE_CUDA
				void setDevice() const
				{
					cudaError_t err = cudaSetDevice(m_device_index);
					CHECK_CUDA_ERROR(err)
				}
				avDeviceIndex_t getDevice() const noexcept
				{
					return m_device_index;
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
				std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> m_strides;
				int m_number_of_dimensions = 0;
				avDataType_t m_dtype = AVOCADO_DTYPE_UNKNOWN;
			public:
				static constexpr av_int64 descriptor_type = 3;

				TensorDescriptor() = default;
				TensorDescriptor(std::initializer_list<int> dimensions, avDataType_t dtype) :
						m_number_of_dimensions(dimensions.size()),
						m_dtype(dtype)
				{
					m_dimensions.fill(0);
					m_strides.fill(0);
				}
				static std::string className()
				{
					return "TensorDescriptor";
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
					setup_stride();
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
					for (int i = 0; i < m_number_of_dimensions; i++)
						result += indices.begin()[i] * m_strides[i];
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
			private:
				void setup_stride()
				{
					int tmp = 1;
					for (int i = m_number_of_dimensions - 1; i >= 0; i--)
					{
						m_strides[i] = tmp;
						tmp *= this->dimension(i);
					}
				}
		};

		class ConvolutionDescriptor
		{
			public:
				static constexpr av_int64 descriptor_type = 4;

				avConvolutionAlgorithm_t algorithm = AVOCADO_CONVOLUTION_ALGORITHM_AUTO;
				avConvolutionMode_t mode = AVOCADO_CONVOLUTION_MODE;
				int dimensions = 2;
				std::array<int, 3> padding;
				std::array<int, 3> stride;
				std::array<int, 3> dilation;
				std::array<uint8_t, 16> padding_value;
				int groups = 1;

				ConvolutionDescriptor() = default;
				void create()
				{
					algorithm = AVOCADO_CONVOLUTION_ALGORITHM_AUTO;
					padding.fill(0);
					stride.fill(1);
					dilation.fill(1);
					padding_value.fill(0u);
					groups = 1;
				}
				void destroy()
				{
				}
				static std::string className()
				{
					return "ConvolutionDescriptor";
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
						std::memcpy(this->padding_value.data(), paddingValue, sizeof(int8_t) * padding_value.size());
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
						std::memcpy(paddingValue, this->padding_value.data(), sizeof(int8_t) * padding_value.size());
				}
				bool paddingWithZeros() const noexcept
				{
					return std::all_of(padding_value.begin(), padding_value.end(), [](uint8_t x)
					{	return x == 0u;});
				}
		};

		class PoolingDescriptor
		{
			public:
				static constexpr av_int64 descriptor_type = 5;

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
				static std::string className()
				{
					return "PoolingDescriptor";
				}
		};

		class OptimizerDescriptor
		{
			public:
				static constexpr av_int64 descriptor_type = 6;

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
				static std::string className()
				{
					return "OptimizerDescriptor";
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
				static constexpr av_int64 descriptor_type = 7;

				DropoutDescriptor() = default;
				void create()
				{
				}
				void destroy()
				{
				}
				static std::string className()
				{
					return "DropoutDescriptor";
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
			template<>
			inline DescriptorPool<ContextDescriptor>& getPool()
			{
				thread_local DescriptorPool<ContextDescriptor> result = []()
				{
					int nb_devices = 0;
					cudaError_t status = cudaGetDeviceCount(&nb_devices);
					if(status != cudaSuccess)
					nb_devices = 0;
					DescriptorPool<ContextDescriptor> tmp;
					for(int i = 0; i < nb_devices; i++)
					tmp.create(i, true); // reserve descriptors for default contexts
						return tmp;
					}();
				return result;
			}
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
			inline avStatus_t create(av_int64 *result, Args &&... args)
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
			inline avStatus_t destroy(av_int64 desc)
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

		inline constexpr bool is_transpose(avGemmOperation_t op) noexcept
		{
			return op == AVOCADO_GEMM_OPERATION_T;
		}

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

#if USE_CUDA
		template<>
		inline float2 getAlphaValue<float2>(const void *alpha) noexcept
		{
			if (alpha == nullptr)
				return float2 { 1.0f, 0.0f };
			else
				return reinterpret_cast<const float2*>(alpha)[0];
		}
		template<>
		inline float2 getBetaValue<float2>(const void *beta) noexcept
		{
			if (beta == nullptr)
				return float2 { 0.0f, 0.0f };
			else
				return reinterpret_cast<const float2*>(beta)[0];
		}
		template<>
		inline double2 getAlphaValue<double2>(const void *alpha) noexcept
		{
			if (alpha == nullptr)
				return double2 { 1.0, 0.0 };
			else
				return reinterpret_cast<const double2*>(alpha)[0];
		}
		template<>
		inline double2 getBetaValue<double2>(const void *beta) noexcept
		{
			if (beta == nullptr)
				return double2 { 0.0, 0.0 };
			else
				return reinterpret_cast<const double2*>(beta)[0];
		}
#endif

		struct BroadcastedDimensions
		{
				int first;
				int last;
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
			int lhs_volume = lhs.volume();
			int rhs_volume = rhs.volume();
			assert(lhs_volume > 0 && rhs_volume > 0);
			BroadcastedDimensions result { lhs_volume / rhs_volume, rhs_volume };
//			for (int i = 0; i < lhs.length - rhs.length; i++)
//				result.first *= lhs.dim[i];
//			for (int i = lhs.length - rhs.length; i < lhs.length; i++)
//				result.last *= lhs.dim[i];
			return result;
		}

		inline bool is_logical(avBinaryOp_t op) noexcept
		{
			return (op == AVOCADO_BINARY_OP_LOGICAL_AND) or (op == AVOCADO_BINARY_OP_LOGICAL_OR) or (op == AVOCADO_BINARY_OP_LOGICAL_OR);
		}
		inline bool is_logical(avUnaryOp_t op) noexcept
		{
			return op == AVOCADO_UNARY_OP_LOGICAL_NOT;
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

		inline TensorDescriptor getConvolutionOutputShape(const ConvolutionDescriptor &config, const TensorDescriptor &inputDesc,
				const TensorDescriptor &filterDesc)
		{
			std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> shape;
			shape[0] = inputDesc.firstDim(); // batch size
			for (int i = 0; i < inputDesc.nbDims() - 2; i++)
				shape[1 + i] = 1
						+ (inputDesc.dimension(1 + i) - 2 * config.padding[i] - (((filterDesc.dimension(1 + i) - 1) * config.dilation[i]) + 1))
								/ config.stride[i];
			shape[inputDesc.nbDims() - 1] = filterDesc.firstDim(); // output filters

			TensorDescriptor result;
			result.set(inputDesc.dtype(), inputDesc.nbDims(), shape.data());
			return result;
		}

	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_ */
