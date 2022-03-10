/*
 * Dense.hpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_DENSE_DENSE_HPP_
#define AVOCADO_LAYERS_DENSE_DENSE_HPP_

#include <Avocado/layers/Layer.hpp>

namespace avocado
{

	class Dense: public Layer
	{
			int m_neurons = 0;
			bool m_use_bias = true;
		public:
			Dense(int neurons, const std::string &activation = "linear", bool useBias = true);

			Layer& useBias(bool b) noexcept;
			bool isUsingBias() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			Dense* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar beta);
	};

} /* namespace avocado */

#endif /* AVOCADO_LAYERS_DENSE_DENSE_HPP_ */
