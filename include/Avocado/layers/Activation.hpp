/*
 * Activation.hpp
 *
 *  Created on: Feb 10, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_ACTIVATIONS_ACTIVATION_HPP_
#define AVOCADO_LAYERS_ACTIVATIONS_ACTIVATION_HPP_

#include <Avocado/layers/Layer.hpp>

namespace avocado
{
	class Activation: public Layer
	{
		public:
			Activation(const std::string &activation = "linear");
			Activation(NonlinearityType activation);

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;

			Activation* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};
} /* namespace avocado */

#endif /* AVOCADO_LAYERS_ACTIVATIONS_ACTIVATION_HPP_ */
