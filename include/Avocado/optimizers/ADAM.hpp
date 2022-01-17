/*
 * ADAM.hpp
 *
 *  Created on: Feb 24, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_OPTIMIZERS_ADAM_HPP_
#define AVOCADO_OPTIMIZERS_ADAM_HPP_

#include <Avocado/optimizers/Optimizer.hpp>
#include <Avocado/core/Tensor.hpp>

#include <memory>

namespace avocado
{
	class ADAM: public Optimizer
	{
		private:
			std::unique_ptr<Tensor> m_momentum;
			std::unique_ptr<Tensor> m_variance;

			float m_learning_rate = 0.001f;
			float m_beta1 = 0.9f;
			float m_beta2 = 0.999f;
			int m_steps = 0;

			bool m_use_amsgrad = false;

		public:
			ADAM() = default;
			ADAM(float learningRate, bool useAMSGrad = false);

			ADAM& setBeta1(float b);
			ADAM& setBeta2(float b);

			float getLearningRate() const noexcept;
			void setLearningRate(float lr) noexcept;
			int getSteps() const noexcept;

			void restart() noexcept;
			void moveTo(Device newDevice);
			void learn(const Context &context, Parameter &param);

			std::string name() const;
			ADAM* clone() const;
			Json serialize(SerializedObject &binary_data) const;
			void unserialize(const Json &json, const SerializedObject &binary_data);
	};

} /* namespace avocado */

#endif /* AVOCADO_OPTIMIZERS_ADAM_HPP_ */
