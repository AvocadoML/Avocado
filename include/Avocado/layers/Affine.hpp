/*
 * Affine.hpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_DENSE_AFFINE_HPP_
#define AVOCADO_LAYERS_DENSE_AFFINE_HPP_

#include <Avocado/layers/Layer.hpp>

namespace avocado
{

	class Affine: public Layer
	{
			bool m_use_weights;
			bool m_use_bias;
		public:
			Affine(const std::string &activation = "linear", bool useWeights = true, bool useBias = true);

			Layer& useWeights(bool b) noexcept;
			bool isUsingWeights() const noexcept;
			Layer& useBias(bool b) noexcept;
			bool isUsingBias() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			Affine* clone(const Json &config) const;

			void forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha,
					Scalar beta);

			void learn();
	};

} /* namespace avocado */

#endif /* AVOCADO_LAYERS_DENSE_AFFINE_HPP_ */
