/*
 * BatchNormalization.hpp
 *
 *  Created on: Feb 24, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_NORM_BATCHNORMALIZATION_HPP_
#define AVOCADO_LAYERS_NORM_BATCHNORMALIZATION_HPP_

#include <Avocado/layers/Layer.hpp>

namespace avocado
{

	class BatchNormalization: public Layer
	{
			Tensor m_running_mean;
			Tensor m_running_variance;
			double m_epsilon = 1.0e-3;
			int m_running_id = 0;
			int m_total_steps = 0;
			int m_history_size = 50;
			bool m_use_gamma = true;
			bool m_use_beta = true;

		public:
			BatchNormalization(const std::string &activation = "linear", bool useGamma = true, bool useBeta = true, int historySize = 64);

			BatchNormalization& useGamma(bool b) noexcept;
			BatchNormalization& useBeta(bool b) noexcept;
			BatchNormalization& history(int historySize) noexcept;
			BatchNormalization& epsilon(double epsilon) noexcept;

			double getEpsilon() const noexcept;

			void setInputShape(const std::vector<Shape> &shapes);
			Shape getOutputShape() const;
			Shape getWeightShape() const;
			Shape getBiasShape() const;

			std::string name() const;
			Json getConfig() const;

			void changeContext(Context &context);

			BatchNormalization* clone(const Json &config) const;

			void init();
			void forward(const std::vector<Tensor> &input, Tensor &output);
			void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar beta);
			void learn();
	};

} /* namespace avocado */

#endif /* AVOCADO_LAYERS_NORM_BATCHNORMALIZATION_HPP_ */
