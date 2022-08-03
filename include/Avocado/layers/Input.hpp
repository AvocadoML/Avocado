/*
 * Input.hpp
 *
 *  Created on: Feb 22, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_DENSE_INPUT_HPP_
#define AVOCADO_LAYERS_DENSE_INPUT_HPP_

#include <Avocado/layers/Layer.hpp>

namespace avocado
{
	class Input: public Layer
	{
		public:
			Input(const Shape &input_shape = Shape());

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			Input* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha, Scalar beta);
	};
} /* namespace avocado */

#endif /* AVOCADO_LAYERS_DENSE_INPUT_HPP_ */
