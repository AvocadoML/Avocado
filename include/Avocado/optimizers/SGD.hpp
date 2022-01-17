/*
 * SGD.hpp
 *
 *  Created on: Sep 29, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_OPTIMIZERS_SGD_HPP_
#define AVOCADO_OPTIMIZERS_SGD_HPP_

#include <Avocado/optimizers/Optimizer.hpp>
#include <Avocado/core/Tensor.hpp>

#include <memory>

namespace avocado
{
	class SGD: public Optimizer
	{
		private:
			std::unique_ptr<Tensor> m_momentum;

			float m_learning_rate = 0.01f;
			float m_beta = 0.0f;
			int m_steps = 0;

			bool m_use_nesterov = false;

		public:
			SGD() = default;
			/**
			 * For details of nesterov see http://proceedings.mlr.press/v28/sutskever13.pdf
			 * @param learningRate
			 * @param momentum
			 * @param nesterov
			 */
			SGD(float learningRate, float momentum = 0.0f, bool useNesterov = false);

			float getLearningRate() const noexcept;
			void setLearningRate(float lr) noexcept;
			int getSteps() const noexcept;

			void restart() noexcept;
			void moveTo(Device newDevice);
			void learn(const Context &context, Parameter &param);

			std::string name() const;
			SGD* clone() const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace avocado */

#endif /* AVOCADO_OPTIMIZERS_SGD_HPP_ */
