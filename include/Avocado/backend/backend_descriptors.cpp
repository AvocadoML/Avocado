/*
 * backend_descriptors.cpp
 *
 *  Created on: Jan 17, 2022
 *      Author: Maciej Kozarzewski
 */

#include "backend_descriptors.hpp"
#include <type_traits>
#include <array>
#include <vector>
#include <stack>
#include <algorithm>
#include <complex>
#include <memory>
#include <cstring>
#include <cassert>

namespace
{
	std::string dtype_to_string(avocado::backend::avDataType_t dtype)
	{
		switch (dtype)
		{
			default:
			case avocado::backend::AVOCADO_DTYPE_UNKNOWN:
				return "UNKNOWN";
			case avocado::backend::AVOCADO_DTYPE_UINT8:
				return "UINT8";
			case avocado::backend::AVOCADO_DTYPE_INT8:
				return "INT8";
			case avocado::backend::AVOCADO_DTYPE_INT16:
				return "INT16";
			case avocado::backend::AVOCADO_DTYPE_INT32:
				return "INT32";
			case avocado::backend::AVOCADO_DTYPE_INT64:
				return "INT64";
			case avocado::backend::AVOCADO_DTYPE_FLOAT16:
				return "FLOAT16";
			case avocado::backend::AVOCADO_DTYPE_BFLOAT16:
				return "BFLOAT16";
			case avocado::backend::AVOCADO_DTYPE_FLOAT32:
				return "FLOAT32";
			case avocado::backend::AVOCADO_DTYPE_FLOAT64:
				return "FLOAT64";
			case avocado::backend::AVOCADO_DTYPE_COMPLEX32:
				return "COMPLEX32";
			case avocado::backend::AVOCADO_DTYPE_COMPLEX64:
				return "COMPLEX64";
		}
	}

}

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

			int get_number_of_devices()
			{
#if USE_CUDA
				static const int result = []()
				{
					int tmp = 0;
					cudaError_t status = cudaGetDeviceCount(&tmp);
					if (status != cudaSuccess)
					return 0;
					else
					return tmp;
				}();
#elif USE_OPENCL

#else
				static const int result = 1;
#endif
				return result;
			}

			avDeviceType_t get_device_type(av_int64 descriptor) noexcept
			{
				const av_int64 device_type_mask = 0xFF00000000000000ull;
				return static_cast<avDeviceType_t>((descriptor & device_type_mask) >> 56ull);
			}
			int get_descriptor_type(av_int64 descriptor) noexcept
			{
				const av_int64 descriptor_type_mask = 0x00FF000000000000ull;
				return static_cast<int>((descriptor & descriptor_type_mask) >> 48ull);
			}
			avDeviceIndex_t get_device_index(av_int64 descriptor) noexcept
			{
				const av_int64 device_index_mask = 0x0000FFFF00000000ull;
				return static_cast<avDeviceIndex_t>((descriptor & device_index_mask) >> 32ull);
			}
			int get_descriptor_index(av_int64 descriptor) noexcept
			{
				const av_int64 descriptor_index_mask = 0x00000000FFFFFFFFull;
				return static_cast<int>(descriptor & descriptor_index_mask);
			}

			av_int64 get_current_device_type() noexcept
			{
#if USE_CUDA
				return static_cast<av_int64>(AVOCADO_DEVICE_CUDA);
#elif USE_OPENCL
				return static_cast<av_int64>(AVOCADO_DEVICE_OPENCL);
#else
				return static_cast<av_int64>(AVOCADO_DEVICE_CPU);
#endif
			}
			av_int64 get_current_device_index() noexcept
			{
				return 0;
			}

			av_int64 create_descriptor(int index, av_int64 type)
			{
				return (static_cast<av_int64>(get_current_device_type()) << 56ull) | (type << 48ull)
						| (static_cast<av_int64>(get_current_device_index()) << 32ull) | static_cast<av_int64>(index);
			}

			int dataTypeSize(avDataType_t dtype) noexcept
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
#  define CHECK_CUDA_ERROR(x, msg) if ((x) != cudaSuccess) throw std::runtime_error(msg);
#  define CHECK_CUBLAS_STATUS(x, msg) if ((x) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(msg);
#endif

			/*
			 * MemoryDescriptor
			 */

#if USE_CUDA or USE_OPENCL
			MemoryDescriptor::MemoryDescriptor(avDeviceIndex_t index, avSize_t sizeInBytes)
			{
				create(index, sizeInBytes);
			}
#else
			MemoryDescriptor::MemoryDescriptor(avSize_t sizeInBytes)
			{
				create(sizeInBytes);
			}
#endif
			MemoryDescriptor::MemoryDescriptor(const MemoryDescriptor &other, avSize_t size, avSize_t offset)
			{
				create(other, size, offset);
			}
			MemoryDescriptor::MemoryDescriptor(MemoryDescriptor &&other) :
					m_data(other.m_data), m_device_index(other.m_device_index), m_size(other.m_size), m_offset(other.m_offset), m_is_owning(other.m_is_owning)
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
			MemoryDescriptor& MemoryDescriptor::operator=(MemoryDescriptor &&other)
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
			MemoryDescriptor::~MemoryDescriptor()
			{
				try
				{
					destroy();
				} catch (std::exception &e)
				{
					exit(-1);
				}
			}
			MemoryDescriptor::operator bool() const noexcept
			{
#if USE_OPENCL
				return m_data != nullptr;
#else
				return m_data != nullptr;
#endif
			}
			avSize_t MemoryDescriptor::size() const noexcept
			{
				return m_size;
			}
			avDeviceIndex_t MemoryDescriptor::device() const noexcept
			{
				return m_device_index;
			}
			std::string MemoryDescriptor::className()
			{
				return "MemoryDescriptor";
			}
#if USE_CUDA or USE_OPENCL
			void MemoryDescriptor::create(avDeviceIndex_t index, avSize_t sizeInBytes)
			{
				if (sizeInBytes > 0)
				{
#  if USE_CUDA
					cudaError_t err = cudaSetDevice(index);
					CHECK_CUDA_ERROR(err, "MemoryDescriptor::create() : cudaSetDevice()");
					err = cudaMalloc(reinterpret_cast<void**>(&m_data), sizeInBytes);
					CHECK_CUDA_ERROR(err, "MemoryDescriptor::create() : cudaMalloc()");
#  else /* USE_OPENCL */

#  endif
				}
				else
					m_data = nullptr;
				m_device_index = index;
				m_offset = 0;
				m_size = sizeInBytes;
				m_is_owning = true;
			}
#else
			void MemoryDescriptor::create(avSize_t sizeInBytes)
			{
				if (sizeInBytes > 0)
					m_data = new int8_t[sizeInBytes];
				else
					m_data = nullptr;
				m_device_index = 0;
				m_offset = 0;
				m_size = sizeInBytes;
				m_is_owning = true;
			}
#endif
			void MemoryDescriptor::create(const MemoryDescriptor &other, avSize_t size, avSize_t offset)
			{
				if (other.m_is_owning == false)
					throw std::logic_error("cannot create memory view from non-owning memory descriptor");
				if (other.m_size < offset + size)
					throw std::logic_error(
							"the view would extend beyond the original tensor : " + std::to_string(other.m_size) + " < " + std::to_string(offset) + "+"
									+ std::to_string(size));
#if USE_OPENCL
#else
				m_data = other.m_data + offset;
#endif
				m_device_index = other.m_device_index;
				m_size = size;
				m_offset = offset; // offset is added twice in data<T>()
				m_is_owning = false;
			}
			void MemoryDescriptor::destroy()
			{
#if USE_CUDA
				if (m_data != nullptr)
				{
					if (m_is_owning)
					{
						cudaError_t err = cudaFree(m_data);
						CHECK_CUDA_ERROR(err, "MemoryDescriptor::destroy() : cudaFree()");
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
//			cl::Buffer& MemoryDescriptor::data(void *ptr) noexcept
//			{
//				return *reinterpret_cast<cl::Buffer*>(ptr);
//			}
//			const cl::Buffer& MemoryDescriptor::data(const void *ptr) const noexcept
//			{
//				return *reinterpret_cast<const cl::Buffer*>(ptr);
//			}
#endif

			/*
			 * ContextDescriptor
			 */
			ContextDescriptor::ContextDescriptor(ContextDescriptor &&other) :
#if USE_CUDA
							m_stream(other.m_stream), m_handle(other.m_handle),
#elif USE_OPENCL
#endif
							m_device_index(other.m_device_index), m_workspace(std::move(other.m_workspace)), m_workspace_size(other.m_workspace_size)
			{
#if USE_CUDA
				other.m_stream = nullptr;
				other.m_handle = nullptr;
#elif USE_OPENCL
#endif
				other.m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
				other.m_workspace_size = 0;
			}
			ContextDescriptor& ContextDescriptor::operator=(ContextDescriptor &&other)
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
			ContextDescriptor::~ContextDescriptor()
			{
				try
				{
					destroy();
				} catch (std::exception &e)
				{
					exit(-1);
				}
			}
			std::string ContextDescriptor::className()
			{
				return "ContextDescriptor";
			}
#if USE_CPU
			void ContextDescriptor::create()
			{
				m_device_index = 0;
			}

#elif USE_CUDA
			void ContextDescriptor::create(avDeviceIndex_t index, bool useDefaultStream)
			{
				cudaError_t err = cudaSetDevice(index);
				CHECK_CUDA_ERROR(err, "ContextDescriptor::create() : cudaSetDevice()");
				if (useDefaultStream)
					m_stream = nullptr;
				else
				{
					err = cudaStreamCreate(&m_stream);
					CHECK_CUDA_ERROR(err, "ContextDescriptor::create() : cudaStreamCreate()");
				}

				cublasStatus_t status = cublasCreate_v2(&m_handle);
				CHECK_CUBLAS_STATUS(status, "ContextDescriptor::create() : cublasCreate_v2()");
				status = cublasSetStream_v2(m_handle, m_stream);
				CHECK_CUBLAS_STATUS(status, "ContextDescriptor::create() : cublasSetStream_v2()");
				m_device_index = index;
			}
#elif USE_OPENCL
			void ContextDescriptor::create(avDeviceIndex_t index, bool useDefaultCommandQueue)
			{
				m_device_index = index;
			}
#else
			void ContextDescriptor::create()
			{
				m_device_index = 0;
			}
#endif
			void ContextDescriptor::destroy()
			{
#if USE_CPU
#elif USE_CUDA
				cudaError_t err = cudaDeviceSynchronize();
				CHECK_CUDA_ERROR(err, "ContextDescriptor::destroy() : cudaDeviceSynchronize()");
				if (m_handle != nullptr)
				{
					cublasStatus_t status = cublasDestroy_v2(m_handle);
					CHECK_CUBLAS_STATUS(status, "ContextDescriptor::destroy() : cublasDestroy_v2()");
					m_handle = nullptr;
				}

				if (m_stream != nullptr)
				{
					err = cudaStreamDestroy(m_stream);
					CHECK_CUDA_ERROR(err, "ContextDescriptor::destroy() : cudaStreamDestroy()");
					m_stream = nullptr;
				}
#elif USE_OPENCL
#else
#endif
				m_workspace.destroy();
				m_workspace_size = 0;
				m_device_index = AVOCADO_INVALID_DEVICE_INDEX;
			}
			MemoryDescriptor& ContextDescriptor::getWorkspace() const
			{
				if (static_cast<bool>(m_workspace) == false)
				{
					m_workspace_size = 1 << 23;
#if USE_CUDA or USE_OPENCL
					m_workspace.create(m_device_index, m_workspace_size); // lazy allocation of 8MB workspace
#else
					m_workspace.create(m_workspace_size); // lazy allocation of 8MB workspace
#endif
				}
				return m_workspace;
			}

#if USE_CUDA
			void ContextDescriptor::setDevice() const
			{
				cudaError_t err = cudaSetDevice(m_device_index);
				CHECK_CUDA_ERROR(err, "ContextDescriptor::setDevice() : cudaSetDevice()");
			}
			avDeviceIndex_t ContextDescriptor::getDevice() const noexcept
			{
				return m_device_index;
			}
			cudaStream_t ContextDescriptor::getStream() const noexcept
			{
				return m_stream;
			}
			cublasHandle_t ContextDescriptor::getHandle() const noexcept
			{
				return m_handle;
			}
#endif

			/*
			 * TensorDescriptor
			 */
			TensorDescriptor::TensorDescriptor(std::initializer_list<int> dimensions, avDataType_t dtype) :
					m_number_of_dimensions(dimensions.size()), m_dtype(dtype)
			{
				m_dimensions.fill(0);
				m_strides.fill(0);
				std::memcpy(m_dimensions.data(), dimensions.begin(), sizeof(int) * dimensions.size());
				setup_stride();
			}
			std::string TensorDescriptor::className()
			{
				return "TensorDescriptor";
			}
			void TensorDescriptor::create()
			{
				m_dimensions.fill(0);
				m_strides.fill(0);
			}
			void TensorDescriptor::destroy()
			{
				m_dimensions.fill(0);
				m_number_of_dimensions = 0;
				m_dtype = AVOCADO_DTYPE_UNKNOWN;
			}
			void TensorDescriptor::set(avDataType_t dtype, int nbDims, const int dimensions[])
			{
				if (dimensions == nullptr or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS)
					throw std::invalid_argument("");
				std::memcpy(m_dimensions.data(), dimensions, sizeof(int) * nbDims);
				m_number_of_dimensions = nbDims;
				m_dtype = dtype;
				setup_stride();
			}
			void TensorDescriptor::get(avDataType_t *dtype, int *nbDims, int dimensions[]) const
			{
				if (dtype != nullptr)
					dtype[0] = m_dtype;
				if (nbDims != nullptr)
					nbDims[0] = m_number_of_dimensions;
				if (dimensions != nullptr)
					std::memcpy(dimensions, m_dimensions.data(), sizeof(int) * m_number_of_dimensions);
			}
			int& TensorDescriptor::operator[](int index)
			{
				return m_dimensions[index];
			}
			int TensorDescriptor::operator[](int index) const
			{
				return m_dimensions[index];
			}
			int TensorDescriptor::dimension(int index) const
			{
				return m_dimensions[index];
			}
			int TensorDescriptor::nbDims() const noexcept
			{
				return m_number_of_dimensions;
			}
			avSize_t TensorDescriptor::sizeInBytes() const noexcept
			{
				return dataTypeSize(m_dtype) * this->volume();
			}
			int TensorDescriptor::getIndex(std::initializer_list<int> indices) const noexcept
			{
				assert(nbDims() == static_cast<int>(indices.size()));
				int result = 0;
				for (int i = 0; i < m_number_of_dimensions; i++)
				{
					const int idx = indices.begin()[i];
					assert(idx >= 0 && idx < m_dimensions[i]);
					result += idx * m_strides[i];
				}
				return result;
			}
			int TensorDescriptor::firstDim() const noexcept
			{
				if (m_number_of_dimensions == 0)
					return 0;
				else
					return m_dimensions[0];
			}
			int TensorDescriptor::lastDim() const noexcept
			{
				if (m_number_of_dimensions == 0)
					return 0;
				else
					return m_dimensions[m_number_of_dimensions - 1];
			}
			int TensorDescriptor::volume() const noexcept
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
			int TensorDescriptor::volumeWithoutFirstDim() const noexcept
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
			int TensorDescriptor::volumeWithoutLastDim() const noexcept
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
			avDataType_t TensorDescriptor::dtype() const noexcept
			{
				return m_dtype;
			}
			bool TensorDescriptor::equalShape(const TensorDescriptor &other) noexcept
			{
				if (m_number_of_dimensions != other.m_number_of_dimensions)
					return false;
				for (int i = 0; i < m_number_of_dimensions; i++)
					if (m_dimensions[i] != other.m_dimensions[i])
						return false;
				return true;
			}
			std::string TensorDescriptor::toString() const
			{
				std::string result = std::string("Tensor<") + dtype_to_string(m_dtype) + ">[";
				for (int i = 0; i < m_number_of_dimensions; i++)
				{
					if (i > 0)
						result += ", ";
					result += std::to_string(m_dimensions[i]);
				}
				result += "] on ";
#if USE_CPU
				result += "CPU";
#elif USE_CUDA
				result += "CUDA";
#elif USE_OPENCL
				result += "OPENCL";
#else
				result += "reference";
#endif
				return result;
			}
			//private
			void TensorDescriptor::setup_stride()
			{
				int tmp = 1;
				for (int i = m_number_of_dimensions - 1; i >= 0; i--)
				{
					m_strides[i] = tmp;
					tmp *= this->dimension(i);
				}
			}

			/*
			 * ConvolutionDescriptor
			 */
			void ConvolutionDescriptor::create()
			{
				padding.fill(0);
				stride.fill(1);
				dilation.fill(1);
				padding_value.fill(0u);
				groups = 1;
			}
			void ConvolutionDescriptor::destroy()
			{
			}
			std::string ConvolutionDescriptor::className()
			{
				return "ConvolutionDescriptor";
			}
			void ConvolutionDescriptor::set(avConvolutionMode_t mode, int nbDims, const int padding[], const int strides[], const int dilation[], int groups,
					const void *paddingValue)
			{
				if (nbDims < 0 or nbDims > 3)
					throw std::invalid_argument("");
				this->mode = mode;
				this->dimensions = nbDims;
				if (strides != nullptr)
					std::memcpy(this->stride.data(), strides, sizeof(int) * dimensions);
				if (padding != nullptr)
					std::memcpy(this->padding.data(), padding, sizeof(int) * dimensions);
				if (dilation != nullptr)
					std::memcpy(this->dilation.data(), dilation, sizeof(int) * dimensions);

				this->groups = groups;
				if (paddingValue != nullptr)
					std::memcpy(this->padding_value.data(), paddingValue, sizeof(int8_t) * padding_value.size());
			}
			void ConvolutionDescriptor::get(avConvolutionMode_t *mode, int *nbDims, int padding[], int strides[], int dilation[], int *groups,
					void *paddingValue) const
			{
				if (mode != nullptr)
					mode[0] = this->mode;
				if (nbDims != nullptr)
					nbDims[0] = this->dimensions;
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
			bool ConvolutionDescriptor::paddingWithZeros() const noexcept
			{
				return std::all_of(padding_value.begin(), padding_value.end(), [](uint8_t x)
				{	return x == 0u;});
			}
			TensorDescriptor ConvolutionDescriptor::getOutputShape(const TensorDescriptor &xDesc, const TensorDescriptor &wDesc) const
			{
				std::array<int, AVOCADO_MAX_TENSOR_DIMENSIONS> shape;
				shape[0] = xDesc.firstDim(); // batch size
				for (int i = 0; i < dimensions; i++)
					shape[1 + i] = 1 + (xDesc.dimension(1 + i) - 2 * padding[i] - (((wDesc.dimension(1 + i) - 1) * dilation[i]) + 1)) / stride[i];
				shape[xDesc.nbDims() - 1] = wDesc.firstDim(); // output filters

				TensorDescriptor result;
				result.set(xDesc.dtype(), xDesc.nbDims(), shape.data());
				return result;
			}
			bool ConvolutionDescriptor::isStrided() const noexcept
			{
				for (int i = 0; i < dimensions; i++)
					if (stride[i] > 1)
						return true;
				return false;
			}
			bool ConvolutionDescriptor::isDilated() const noexcept
			{
				for (int i = 0; i < dimensions; i++)
					if (dilation[i] > 1)
						return true;
				return false;
			}
			std::string ConvolutionDescriptor::toString() const
			{
				std::string result;
				if (mode == AVOCADO_CONVOLUTION_MODE)
					result += "convolution : ";
				else
					result += "cross-correlation : ";
				result += "padding = [";
				for (int i = 0; i < dimensions; i++)
				{
					if (i > 0)
						result += ", ";
					result += std::to_string(padding[i]);
				}
				result += "], strides = [";
				for (int i = 0; i < dimensions; i++)
				{
					if (i > 0)
						result += ", ";
					result += std::to_string(stride[i]);
				}
				result += "], dilation = [";
				for (int i = 0; i < dimensions; i++)
				{
					if (i > 0)
						result += ", ";
					result += std::to_string(dilation[i]);
				}
				result += "], groups = " + std::to_string(groups);
				return result;
			}

			/*
			 * PoolingDescriptor
			 */
			void PoolingDescriptor::create()
			{
				filter.fill(0);
				padding.fill(0);
				stride.fill(1);
			}
			void PoolingDescriptor::destroy()
			{
			}
			std::string PoolingDescriptor::className()
			{
				return "PoolingDescriptor";
			}

			/*
			 * OptimizerDescriptor
			 */
			void OptimizerDescriptor::create()
			{
				type = AVOCADO_OPTIMIZER_SGD;
				learning_rate = 0.0;
				coef.fill(0);
				flags.fill(false);
			}
			void OptimizerDescriptor::destroy()
			{
			}
			std::string OptimizerDescriptor::className()
			{
				return "OptimizerDescriptor";
			}
			void OptimizerDescriptor::set(avOptimizerType_t optimizerType, double learningRate, const double coefficients[], const bool flags[])
			{
				if (coefficients == nullptr)
					throw std::invalid_argument("");
				if (flags == nullptr)
					throw std::invalid_argument("");

				this->type = optimizerType;
				this->learning_rate = learningRate;
				std::memcpy(this->coef.data(), coefficients, sizeof(this->coef));
				std::memcpy(this->flags.data(), flags, sizeof(this->flags));
			}
			void OptimizerDescriptor::get(avOptimizerType_t *optimizerType, double *learningRate, double coefficients[], bool flags[])
			{
				if (optimizerType != nullptr)
					optimizerType[0] = this->type;
				if (learningRate != nullptr)
					learningRate[0] = this->learning_rate;
				if (coefficients != nullptr)
					std::memcpy(coefficients, this->coef.data(), sizeof(this->coef));
				if (flags != nullptr)
					std::memcpy(flags, this->flags.data(), sizeof(this->flags));
			}
			void OptimizerDescriptor::get_workspace_size(avSize_t *result, const TensorDescriptor &wDesc) const
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

			/*
			 * DropoutDescriptor
			 */
			void DropoutDescriptor::create()
			{
			}
			void DropoutDescriptor::destroy()
			{
			}
			std::string DropoutDescriptor::className()
			{
				return "DropoutDescriptor";
			}

			/*
			 * GetPool
			 */
			template<>
			DescriptorPool<ContextDescriptor>& getPool()
			{
				static DescriptorPool<ContextDescriptor> result(10, get_number_of_devices());
				return result;
			}

			bool isDefault(avContextDescriptor_t desc)
			{
				const int idx = get_descriptor_index(desc);
				return 0 <= idx and idx < get_number_of_devices();
			}
#if USE_CUDA
			thread_local DescriptorPool<ContextDescriptor> default_context_pool = []()
			{
				try
				{
					DescriptorPool<ContextDescriptor> tmp;
					for(int i = 0; i < get_number_of_devices(); i++)
					tmp.create(i, true);
					return tmp;
				}
				catch (std::exception &e)
				{
					return DescriptorPool<ContextDescriptor>();
				}
			}();
#elif USE_OPENCL
#else
			thread_local DescriptorPool<ContextDescriptor> default_context_pool = []()
			{
				try
				{
					DescriptorPool<ContextDescriptor> tmp;
					tmp.create();
					return tmp;
				}
				catch (std::exception &e)
				{
					return DescriptorPool<ContextDescriptor>();
				}
			}();
#endif

			MemoryDescriptor& getMemory(avMemoryDescriptor_t desc)
			{
				return getPool<MemoryDescriptor>().get(desc);
			}
			ContextDescriptor& getContext(avContextDescriptor_t desc)
			{
				if (isDefault(desc))
					return default_context_pool.get(desc);
				else
					return getPool<ContextDescriptor>().get(desc);
			}
			TensorDescriptor& getTensor(avTensorDescriptor_t desc)
			{
				return getPool<TensorDescriptor>().get(desc);
			}
			ConvolutionDescriptor& getConvolution(avConvolutionDescriptor_t desc)
			{
				return getPool<ConvolutionDescriptor>().get(desc);
			}
			PoolingDescriptor& getPooling(avPoolingDescriptor_t desc)
			{
				return getPool<PoolingDescriptor>().get(desc);
			}
			OptimizerDescriptor& getOptimizer(avOptimizerDescriptor_t desc)
			{
				return getPool<OptimizerDescriptor>().get(desc);
			}
			DropoutDescriptor& getDropout(avDropoutDescriptor_t desc)
			{
				return getPool<DropoutDescriptor>().get(desc);
			}

			const MemoryDescriptor& const_getMemory(avMemoryDescriptor_t desc)
			{
				return getPool<MemoryDescriptor>().const_get(desc);
			}
			const ContextDescriptor& const_getContext(avContextDescriptor_t desc)
			{
				if (isDefault(desc))
					return default_context_pool.const_get(desc);
				else
					return getPool<ContextDescriptor>().const_get(desc);
			}
			const TensorDescriptor& const_getTensor(avTensorDescriptor_t desc)
			{
				return getPool<TensorDescriptor>().const_get(desc);
			}
			const ConvolutionDescriptor& const_getConvolution(avConvolutionDescriptor_t desc)
			{
				return getPool<ConvolutionDescriptor>().const_get(desc);
			}
			const PoolingDescriptor& const_getPooling(avPoolingDescriptor_t desc)
			{
				return getPool<PoolingDescriptor>().const_get(desc);
			}
			const OptimizerDescriptor& const_getOptimizer(avOptimizerDescriptor_t desc)
			{
				return getPool<OptimizerDescriptor>().const_get(desc);
			}
			const DropoutDescriptor& const_getDropout(avDropoutDescriptor_t desc)
			{
				return getPool<DropoutDescriptor>().const_get(desc);
			}

			bool is_transpose(avGemmOperation_t op) noexcept
			{
				return op == AVOCADO_GEMM_OPERATION_T;
			}

#if USE_CUDA
			template<>
			float2 getAlphaValue<float2>(const void *alpha) noexcept
			{
				if (alpha == nullptr)
					return float2 { 1.0f, 0.0f };
				else
					return reinterpret_cast<const float2*>(alpha)[0];
			}
			template<>
			float2 getBetaValue<float2>(const void *beta) noexcept
			{
				if (beta == nullptr)
					return float2 { 0.0f, 0.0f };
				else
					return reinterpret_cast<const float2*>(beta)[0];
			}
			template<>
			double2 getAlphaValue<double2>(const void *alpha) noexcept
			{
				if (alpha == nullptr)
					return double2 { 1.0, 0.0 };
				else
					return reinterpret_cast<const double2*>(alpha)[0];
			}
			template<>
			double2 getBetaValue<double2>(const void *beta) noexcept
			{
				if (beta == nullptr)
					return double2 { 0.0, 0.0 };
				else
					return reinterpret_cast<const double2*>(beta)[0];
			}
#endif

			/**
			 * Only the right hand side (rhs) operand can be broadcasted into the left hand side (lhs).
			 * The number of dimensions of the rhs tensor must be lower or equal to the lhs tensor.
			 * All k dimensions of the rhs must match the last k dimensions of the lhs.
			 *
			 */
			bool isBroadcastPossible(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept
			{
				if (lhs.nbDims() < rhs.nbDims())
					return false;
				else
				{
					for (int i = 0, k = lhs.nbDims() - rhs.nbDims(); i < rhs.nbDims(); i++, k++)
						if (lhs.dimension(k) != rhs.dimension(i) and rhs.dimension(i) != 1)
							return false;
					return true;
				}
			}
			int volume(const BroadcastedDimensions &dims) noexcept
			{
				return dims.first * dims.last;
			}
			BroadcastedDimensions getBroadcastDimensions(const TensorDescriptor &lhs, const TensorDescriptor &rhs) noexcept
			{
				assert(isBroadcastPossible(lhs, rhs));
				int lhs_volume = lhs.volume();
				int rhs_volume = rhs.volume();
				assert(lhs_volume > 0 && rhs_volume > 0);
				BroadcastedDimensions result { lhs_volume / rhs_volume, rhs_volume };
				return result;
			}

			bool is_logical(avBinaryOp_t op) noexcept
			{
				return (op == AVOCADO_BINARY_OP_LOGICAL_AND) or (op == AVOCADO_BINARY_OP_LOGICAL_OR) or (op == AVOCADO_BINARY_OP_LOGICAL_OR);
			}
			bool is_logical(avUnaryOp_t op) noexcept
			{
				return op == AVOCADO_UNARY_OP_LOGICAL_NOT;
			}
			bool is_logical(avReduceOp_t op) noexcept
			{
				return (op == AVOCADO_REDUCE_LOGICAL_AND) or (op == AVOCADO_REDUCE_LOGICAL_OR);
			}

		} /* namespace cpu/cuda/opencl/reference */
	} /* namespace backend */
} /* namespace avocado */

