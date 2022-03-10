/*
 * convolution.hpp
 *
 *  Created on: Oct 22, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_CONVOLUTIONS_HPP_
#define AVOCADO_MATH_CONVOLUTIONS_HPP_

#include <Avocado/math/activations.hpp>
#include <Avocado/math/descriptor_wrappers.hpp>

#include <array>
#include <cstring>

namespace avocado
{
	class Context;
	class Shape;
	class Tensor;
	enum class NonlinearityType
	;
}

namespace avocado
{
	enum class ConvPadding
	{
		SAME,
		VALID
	};

	enum class ConvMode
	{
		CONVOLUTION,
		CROSS_CORRELATION
	};

	enum class ConvAlgorithm
	{
		AUTO,
		EXPLICIT_GEMM,
		IMPLICIT_GEMM,
		WINOGRAD_FUSED,
		WINOGRAD_NON_FUSED
	};

	struct ConvConfig
	{
		private:
			avocado::internal::ConvolutionDescWrapper m_descriptor;

			ConvAlgorithm m_algorithm = ConvAlgorithm::AUTO;
			ConvMode m_mode = ConvMode::CONVOLUTION;
			int m_groups = 1;
			int m_dimensions = 2;
			std::array<int, 3> m_padding = { 0, 0, 0 };
			std::array<int, 3> m_stride = { 1, 1, 1 };
			std::array<int, 3> m_dilation = { 1, 1, 1 };
			std::array<uint8_t, 16> m_padding_value;
		public:
			ConvConfig()
			{
				std::fill(m_padding_value.begin(), m_padding_value.end(), 0);
			}
			void setAlgorithm(ConvAlgorithm algorithm)
			{
				m_algorithm = algorithm;
			}
			void setMode(ConvMode mode)
			{
				m_mode = mode;
				m_descriptor.set(m_mode, m_dimensions, m_padding, m_stride, m_dilation, m_groups, m_padding_value);
			}
			void setGroups(int groups)
			{
				m_groups = groups;
				m_descriptor.set(m_mode, m_dimensions, m_padding, m_stride, m_dilation, m_groups, m_padding_value);
			}
			void setDimensions(int dimensions)
			{
				m_dimensions = dimensions;
				m_descriptor.set(m_mode, m_dimensions, m_padding, m_stride, m_dilation, m_groups, m_padding_value);
			}
			void setPadding(const std::array<int, 3> &padding)
			{
				m_padding = padding;
				m_descriptor.set(m_mode, m_dimensions, m_padding, m_stride, m_dilation, m_groups, m_padding_value);
			}
			void setStride(const std::array<int, 3> &stride)
			{
				m_stride = stride;
				m_descriptor.set(m_mode, m_dimensions, m_padding, m_stride, m_dilation, m_groups, m_padding_value);
			}
			void setDilation(const std::array<int, 3> &dilation)
			{
				m_dilation = dilation;
				m_descriptor.set(m_mode, m_dimensions, m_padding, m_stride, m_dilation, m_groups, m_padding_value);
			}
			template<typename T>
			void setPaddingValue(T value)
			{
				std::memcpy(m_padding_value.data(), &value, sizeof(T));
				m_descriptor.set(m_mode, m_dimensions, m_padding, m_stride, m_dilation, m_groups, m_padding_value);
			}

			ConvAlgorithm getAlgorithm() const noexcept
			{
				return m_algorithm;
			}
			ConvMode getMode() const noexcept
			{
				return m_mode;
			}
			int getGroups() const noexcept
			{
				return m_groups;
			}
			int getDimensions() const noexcept
			{
				return m_dimensions;
			}
			const std::array<int, 3>& getPadding() const noexcept
			{
				return m_padding;
			}
			const std::array<int, 3>& getStride() const noexcept
			{
				return m_stride;
			}
			const std::array<int, 3>& getDilation() const noexcept
			{
				return m_dilation;
			}
			template<typename T>
			T getPaddingValue() const noexcept
			{
				T result;
				std::memcpy(&result, m_padding_value.data(), sizeof(T));
				return result;
			}
			operator backend::avConvolutionDescriptor_t() const noexcept
			{
				return static_cast<backend::avConvolutionDescriptor_t>(m_descriptor);
			}
	};

	namespace math
	{
		std::array<int, 3> getConvolutionPadding(const ConvConfig &config, const Shape &inputShape, const Shape &weightShape);
		Shape getConvolutionOutputShape(const ConvConfig &config, const Shape &inputShape, const Shape &weightShape);

		void imToRow(const Context &context, const ConvConfig &config, const Shape &weightShape, const Tensor &input, Tensor &output,
				bool invertKernel = false);

		void winogradWeightTransform(const Context &context, const ConvConfig &config, int transformSize, const Tensor &filter, Tensor &matrices);
		void winogradInputTransform(const Context &context, const ConvConfig &config, int transformSize, const Tensor &filter, const Tensor &input,
				Tensor &matrices);
		void winogradOutputTransform(const Context &context, const ConvConfig &config, int transformSize, const Tensor &filter, Scalar alpha1,
				const Tensor &matrices, Tensor &output, const Tensor &bias, Scalar alpha2, const Tensor &ext, Scalar beta,
				NonlinearityType activation);
		void winogradGradientTransform(const Context &context, const ConvConfig &config, int transformSize, const Tensor &filter,
				const Tensor &gradientNext, Tensor &matrices);
		void winogradUpdateTransform(const Context &context, const ConvConfig &config, int transformSize, Scalar alpha, const Tensor &matricesDesc,
				Scalar beta, Tensor &dwDesc);

		/**
		 *  @brief Calculates output = activation( (input * weights) + bias)
		 */
		void convolutionForward(const Context &context, const ConvConfig &config, const Tensor &input, Tensor &output, const Tensor &weights,
				const Tensor &bias, const Tensor &ext, Scalar alpha1, Scalar alpha2, Scalar beta, NonlinearityType activation);
		void convolutionBackward(const Context &context, const ConvConfig &config, Tensor &gradientPrev, Tensor &gradientNext, const Tensor &input,
				const Tensor &output, const Tensor &weights);
		void convolutionUpdate(const Context &context, const ConvConfig &config, const Tensor &gradientNext, const Tensor &input,
				Tensor &weightUpdate, Tensor &biasUpdate);

	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_CONVOLUTIONS_HPP_ */
