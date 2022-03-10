/*
 * Add.hpp
 *
 *  Created on: Feb 24, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_MERGE_ADD_HPP_
#define AVOCADO_LAYERS_MERGE_ADD_HPP_

#include <Avocado/layers/Layer.hpp>

namespace avocado
{
	class Add: public Layer
	{
		public:
			Add(const std::string &activation = "linear");

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;

			Add* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha,
					Scalar beta);
	};
} /* namespace avocado */

#endif /* AVOCADO_LAYERS_MERGE_ADD_HPP_ */
