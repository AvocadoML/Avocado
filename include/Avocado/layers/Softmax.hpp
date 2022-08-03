/*
 * Softmax.hpp
 *
 *  Created on: Feb 10, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_ACTIVATIONS_SOFTMAX_HPP_
#define AVOCADO_LAYERS_ACTIVATIONS_SOFTMAX_HPP_

#include <Avocado/layers/Layer.hpp>
#include <Avocado/math/activations.hpp>

#include <vector>

namespace avocado
{
	class Softmax: public Layer
	{
			SoftmaxMode m_mode;
		public:
			Softmax(SoftmaxMode mode = SoftmaxMode::PER_CHANNEL);

			void setNonlinearity(NonlinearityType act) noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;

			std::string name() const;
			Json getConfig() const;

			Softmax* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha,
					Scalar beta);
	};
} /* namespace avocado */

#endif /* AVOCADO_LAYERS_ACTIVATIONS_SOFTMAX_HPP_ */
