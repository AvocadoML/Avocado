/*
 * Concatenate.hpp
 *
 *  Created on: Feb 24, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_MERGE_CONCATENATE_HPP_
#define AVOCADO_LAYERS_MERGE_CONCATENATE_HPP_

#include <Avocado/layers/Layer.hpp>

namespace avocado
{
	class Concatenate: public Layer
	{
		public:
			Concatenate() = default;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;

			Concatenate* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradient_prev, Tensor &gradient_next);
	};
} /* namespace avocado */

#endif /* AVOCADO_LAYERS_MERGE_CONCATENATE_HPP_ */
