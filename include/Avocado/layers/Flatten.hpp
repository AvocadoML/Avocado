/*
 * Flatten.hpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_DENSE_FLATTEN_HPP_
#define AVOCADO_LAYERS_DENSE_FLATTEN_HPP_

#include <Avocado/layers/Layer.hpp>

namespace avocado
{
	class Flatten: public Layer
	{
		public:
			Flatten() = default;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;

			Flatten* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar beta);
	};
} /* namespace avocado */

#endif /* AVOCADO_LAYERS_DENSE_FLATTEN_HPP_ */
