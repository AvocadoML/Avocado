/*
 * Conv2D.hpp
 *
 *  Created on: Feb 26, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_CONV_CONV2D_HPP_
#define AVOCADO_LAYERS_CONV_CONV2D_HPP_

#include <Avocado/layers/Layer.hpp>
#include <Avocado/math/convolutions.hpp>
#include <initializer_list>

namespace avocado
{

	class Conv2D: public Layer
	{
		private:
			int m_output_filters = 0;
			ConvConfig m_config;

			bool m_use_bias = true;
			ConvPadding m_padding = ConvPadding::VALID;

		public:
			Conv2D(int filters, int kernelSize, const std::string &activation = "linear", bool useBias = true);
			Conv2D(int filters, std::initializer_list<int> kernelSize, const std::string &activation = "linear", bool useBias = true);

			Layer& useBias(bool b) noexcept;
			Layer& setPadding(ConvPadding padding) noexcept;
			Layer& setStride(int stride) noexcept;
			Layer& setStride(std::initializer_list<int> stride) noexcept;
			Layer& setDilation(int dilation) noexcept;
			Layer& setDilation(std::initializer_list<int> dilation) noexcept;
			Layer& setGroups(int groups) noexcept;
			bool isUsingBias() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			Conv2D* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha,
					Scalar beta);
	};

} /* namespace avocado */

#endif /* AVOCADO_LAYERS_CONV_CONV2D_HPP_ */
