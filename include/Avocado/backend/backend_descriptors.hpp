/*
 * backend_descriptors.hpp
 *
 *  Created on: Dec 5, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_
#define AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_

#include "backend_defs.h"
#include <type_traits>
#include <array>
#include <vector>
#include <stack>
#include <algorithm>
#include <complex>
#include <memory>
#include <cstring>
#include <cassert>
#include <mutex>
#include <iostream>

#if USE_CPU
#elif USE_CUDA
#  include <cuda_runtime_api.h>
#  include <cuda_fp16.h>
#  include <cublas_v2.h>
#elif USE_OPENCL
#  include <CL/cl.hpp>
#else
#endif

namespace avocado
{
	namespace backend
	{
#if USE_CPU
		namespace cpu
		{
#elif USE_CUDA
		namespace cuda
		{
#elif USE_OPENCL
		namespace opencl
		{
#else
		namespace reference
		{
#endif
			int get_number_of_devices();
			avDeviceType_t get_device_type(av_int64 descriptor) noexcept;
			int get_descriptor_type(av_int64 descriptor) noexcept;
			avDeviceIndex_t get_device_index(av_int64 descriptor) noexcept;
			int get_descriptor_index(av_int64 descriptor) noexcept;

			av_int64 get_current_device_type() noexcept;
			av_int64 get_current_device_index() noexcept;

			av_int64 create_descriptor(int index, av_int64 type);

			int dataTypeSize(avDataType_t dtype) noexcept;

			class MemoryDescriptor
			{
#if USE_OPENCL
					int8_t *m_data = nullptr;
#else
					int8_t *m_data = nullptr;
#endif
					avDeviceIndex_t m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
					av_int64 m_size = 0;
					av_int64 m_offset = 0;
					bool m_is_owning = false;
				public:
					static constexpr av_int64 descriptor_type = 1;

					MemoryDescriptor() = default;
#if USE_CUDA or USE_OPENCL
					MemoryDescriptor(avDeviceIndex_t index, av_int64 sizeInBytes);
#else
					MemoryDescriptor(av_int64 sizeInBytes);
#endif
					MemoryDescriptor(const MemoryDescriptor &other, av_int64 size, av_int64 offset);
					MemoryDescriptor(const MemoryDescriptor &other) = delete;
					MemoryDescriptor(MemoryDescriptor &&other);
					MemoryDescriptor& operator=(const MemoryDescriptor &other) = delete;
					MemoryDescriptor& operator=(MemoryDescriptor &&other);
					~MemoryDescriptor();
					bool isNull() const noexcept;
					av_int64 size() const noexcept;
					avDeviceIndex_t device() const noexcept;
					static std::string className();

					/**
					 * \brief This method allocates new memory block and sets up the descriptor.
					 */
#if USE_CUDA or USE_OPENCL
					void create(avDeviceIndex_t index, av_int64 sizeInBytes);
#else
					void create(av_int64 sizeInBytes);
#endif
					/**
					 * \brief Creates a non-owning view of another memory block.
					 */
					void create(const MemoryDescriptor &other, av_int64 size, av_int64 offset);

					/**
					 * \brief This method deallocates underlying memory and resets the descriptor.
					 * Calling this method on an already destroyed descriptor has no effect.
					 */
					void destroy();

#if USE_OPENCL
//					cl::Buffer& data(void *ptr) noexcept;
//					const cl::Buffer& data(const void *ptr) const noexcept;

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

#else
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
					mutable MemoryDescriptor m_workspace;
					mutable av_int64 m_workspace_size = 0;
				public:
					static constexpr av_int64 descriptor_type = 2;

					ContextDescriptor() = default;
					ContextDescriptor(const ContextDescriptor &other) = delete;
					ContextDescriptor(ContextDescriptor &&other);
					ContextDescriptor& operator=(const ContextDescriptor &other) = delete;
					ContextDescriptor& operator=(ContextDescriptor &&other);
					~ContextDescriptor();
					static std::string className();

					/**
					 * \brief This method initializes context descriptor.
					 */
#if USE_CPU
					void create();

#elif USE_CUDA
					void create(avDeviceIndex_t index, bool useDefaultStream = false);
#elif USE_OPENCL
				void create(avDeviceIndex_t index, bool useDefaultCommandQueue);
#else
					void create();
#endif
					/**
					 * \brief This method destroys context and all its resources.
					 * Calling this method on an already destroyed descriptor has no effect.
					 */
					void destroy();
					MemoryDescriptor& getWorkspace() const;
#if USE_CUDA
					void setDevice() const;
					avDeviceIndex_t getDevice() const noexcept;
					cudaStream_t getStream() const noexcept;
					cublasHandle_t getHandle() const noexcept;
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
					TensorDescriptor(std::initializer_list<int> dimensions, avDataType_t dtype);
					static std::string className();

					void create();
					void destroy();
					void set(avDataType_t dtype, int nbDims, const int dimensions[]);
					void get(avDataType_t *dtype, int *nbDims, int dimensions[]) const;

					int& operator[](int index);
					int operator[](int index) const;
					int dimension(int index) const;
					int nbDims() const noexcept;
					av_int64 sizeInBytes() const noexcept;
					int getIndex(std::initializer_list<int> indices) const noexcept;
					int firstDim() const noexcept;
					int lastDim() const noexcept;
					int volume() const noexcept;
					int volumeWithoutFirstDim() const noexcept;
					int volumeWithoutLastDim() const noexcept;
					avDataType_t dtype() const noexcept;

					bool equalShape(const TensorDescriptor &other) noexcept;
					std::string toString() const;
				private:
					void setup_stride();
			};

			class ConvolutionDescriptor
			{
				public:
					static constexpr av_int64 descriptor_type = 4;

					avConvolutionMode_t mode = AVOCADO_CONVOLUTION_MODE;
					int dimensions = 2;
					std::array<int, 3> padding;
					std::array<int, 3> stride;
					std::array<int, 3> dilation;
					std::array<uint8_t, 16> padding_value;
					int groups = 1;

					ConvolutionDescriptor() = default;
					void create();
					void destroy();
					static std::string className();

					void set(avConvolutionMode_t mode, int nbDims, const int padding[], const int strides[], const int dilation[], int groups,
							const void *paddingValue);
					void get(avConvolutionMode_t *mode, int *nbDims, int padding[], int strides[], int dilation[], int *groups,
							void *paddingValue) const;
					template<typename T>
					T getPaddingValue() const noexcept
					{
						static_assert(sizeof(T) <= sizeof(padding_value), "");
						T result;
						std::memcpy(&result, padding_value.data(), sizeof(T));
						return result;
					}
					bool paddingWithZeros() const noexcept;
					TensorDescriptor getOutputShape(const TensorDescriptor &xDesc, const TensorDescriptor &wDesc) const;
					bool isStrided() const noexcept;
					bool isDilated() const noexcept;
					std::string toString() const;
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
					void create();
					void destroy();
					static std::string className();
			};

			class OptimizerDescriptor
			{
				public:
					static constexpr av_int64 descriptor_type = 6;

					avOptimizerType_t type = AVOCADO_OPTIMIZER_SGD;
					int64_t steps = 0;
					double learning_rate = 0.0;
					std::array<double, 4> coef;
					std::array<bool, 4> flags;

					OptimizerDescriptor() = default;
					void create();
					void destroy();
					static std::string className();

					void set(avOptimizerType_t optimizerType, av_int64 steps, double learningRate, const double coefficients[], const bool flags[]);
					void get(avOptimizerType_t *optimizerType, av_int64 *steps, double *learningRate, double coefficients[], bool flags[]);
					void get_workspace_size(av_int64 *result, const TensorDescriptor &wDesc) const;
			};

			class DropoutDescriptor
			{
				public:
					static constexpr av_int64 descriptor_type = 7;

					DropoutDescriptor() = default;
					void create();
					void destroy();
					static std::string className();
			};

			/*
			 * DescriptorPool
			 */
			template<typename T>
			class DescriptorPool
			{
					T m_null_descriptor;
					std::vector<std::unique_ptr<T>> m_pool;
					std::vector<int> m_available_descriptors;
					std::mutex m_pool_mutex;
				public:
					DescriptorPool(size_t initialSize = 10, int numRestricted = 0)
					{
						m_pool.reserve(initialSize + numRestricted);
						m_available_descriptors.reserve(initialSize);
						for (int i = 0; i < numRestricted; i++)
							m_pool.push_back(nullptr); // reserve few descriptors values for default objects
					}
					DescriptorPool(const DescriptorPool<T> &other) = delete;
					DescriptorPool(DescriptorPool<T> &&other) :
							m_pool(std::move(other.m_pool)),
							m_available_descriptors(std::move(other.m_available_descriptors))
					{
					}
					DescriptorPool& operator=(const DescriptorPool<T> &other) = delete;
					DescriptorPool& operator=(DescriptorPool<T> &&other)
					{
						std::swap(this->m_pool, other.m_pool);
						std::swap(this->m_available_descriptors, other.m_available_descriptors);
						return *this;
					}
					~DescriptorPool() = default;

					/**
					 * \brief Checks if the passed descriptor is valid.
					 * The descriptor is valid if and only if its index is within the size of m_pool vector and is not in the list of available descriptors.
					 */
					bool isValid(int64_t desc) const noexcept
					{
//						std::cout << __FUNCTION__ << " " << __LINE__ << " : index = " << desc << '\n';
//						std::cout << __FUNCTION__ << " " << __LINE__ << " : device type = " << get_current_device_type() << '\n';
						if (get_current_device_type() != get_device_type(desc))
						{
//							std::cout << __FUNCTION__ << " " << __LINE__ << " : device type mismatch : " << get_current_device_type() << " vs "
//									<< get_device_type(desc) << std::endl;
							return false;
						}
						if (T::descriptor_type != get_descriptor_type(desc))
						{
//							std::cout << __FUNCTION__ << " " << __LINE__ << " : type mismatch : " << T::descriptor_type << " vs "
//									<< get_descriptor_type(desc) << std::endl;
							return false;
						}

						int index = get_descriptor_index(desc);
//						std::cout << __FUNCTION__ << " " << __LINE__ << " object index = " << index << '\n';
						if (index < 0 or index > static_cast<int>(m_pool.size()))
						{
//							std::cout << __FUNCTION__ << " " << __LINE__ << " : out of bounds : " << index << " vs 0:" << m_pool.size() << std::endl;
							return false;
						}
						bool asdf = std::find(m_available_descriptors.begin(), m_available_descriptors.end(), index) == m_available_descriptors.end();
						if (asdf == false)
						{
//							std::cout << "not in available" << std::endl;
						}
						return asdf;
					}

					T& get(av_int64 desc)
					{
//						std::cout << __FUNCTION__ << " " << __LINE__ << " : " << T::className() << "\n";
//						std::cout << __FUNCTION__ << " " << __LINE__ << " : " << index << '\n';
						if (desc == AVOCADO_NULL_DESCRIPTOR)
							return m_null_descriptor;
						else
						{
							if (isValid(desc))
								return *(m_pool.at(get_descriptor_index(desc)));
							else
								throw std::logic_error("invalid descriptor " + std::to_string(desc) + " for pool type '" + T::className() + "'");
						}
					}

					template<typename ... Args>
					av_int64 create(Args &&... args)
					{
						std::lock_guard<std::mutex> lock(m_pool_mutex);
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
						av_int64 tmp = create_descriptor(result, T::descriptor_type);
//						std::cout << __FUNCTION__ << " " << __LINE__ << " : " << T::className() << " = " << tmp << ", " << m_pool.size() << std::endl;
						return tmp;
//						return create_descriptor(result, T::descriptor_type);
					}
					void destroy(av_int64 desc)
					{
						std::lock_guard<std::mutex> lock(m_pool_mutex);
//						std::cout << __FUNCTION__ << " " << __LINE__ << " : " << T::className() << " = " << desc << std::endl;
						if (not isValid(desc))
							throw std::logic_error("invalid descriptor " + std::to_string(desc) + " of type '" + T::className() + "'");
						int index = get_descriptor_index(desc);
						m_pool.at(index)->destroy();
						m_available_descriptors.push_back(index);
					}
			};

			template<class T>
			DescriptorPool<T>& getPool()
			{
				static DescriptorPool<T> result;
				return result;
			}
			template<>
			DescriptorPool<ContextDescriptor>& getPool();

			template<typename T, typename ... Args>
			avStatus_t create(av_int64 *result, Args &&... args)
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
			avStatus_t destroy(av_int64 desc)
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

			bool isDefault(avContextDescriptor_t desc);

			MemoryDescriptor& getMemory(avMemoryDescriptor_t desc);
			ContextDescriptor& getContext(avContextDescriptor_t desc);
			TensorDescriptor& getTensor(avTensorDescriptor_t desc);
			ConvolutionDescriptor& getConvolution(avConvolutionDescriptor_t desc);
			PoolingDescriptor& getPooling(avPoolingDescriptor_t desc);
			OptimizerDescriptor& getOptimizer(avOptimizerDescriptor_t desc);
			DropoutDescriptor& getDropout(avDropoutDescriptor_t desc);

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
			float2 getAlphaValue<float2>(const void *alpha) noexcept;
			template<>
			float2 getBetaValue<float2>(const void *beta) noexcept;
			template<>
			double2 getAlphaValue<double2>(const void *alpha) noexcept;
			template<>
			double2 getBetaValue<double2>(const void *beta) noexcept;
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
			bool isBroadcastPossible(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept;
			int volume(const BroadcastedDimensions &dims) noexcept;
			BroadcastedDimensions getBroadcastDimensions(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept;

			bool is_transpose(avGemmOperation_t op) noexcept;
			bool is_logical(avBinaryOp_t op) noexcept;
			bool is_logical(avUnaryOp_t op) noexcept;
			bool is_logical(avReduceOp_t op) noexcept;

			template<typename T, typename U>
			bool same_device_type(T lhs, U rhs)
			{
				return get_device_type(lhs) == get_device_type(rhs);
			}
			template<typename T, typename U, typename ... ARGS>
			bool same_device_type(T lhs, U rhs, ARGS ... args)
			{
				if (get_device_type(lhs) == get_device_type(rhs))
					return same_device_type(lhs, args...);
				else
					return false;
			}

		} /* namespace cpu/cuda/opencl/reference */
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_BACKEND_BACKEND_DESCRIPTORS_HPP_ */
