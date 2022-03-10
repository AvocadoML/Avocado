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
#include <Avocado/math/training.hpp>

#include <memory>

namespace avocado
{
	class ADAM: public Optimizer
	{
		private:
			std::unique_ptr<Tensor> m_workspace;
			OptimizerConfig m_config;

		public:
			ADAM() = default;
			ADAM(double learningRate, bool useAMSGrad = false);

			ADAM& setBeta1(double beta1);
			ADAM& setBeta2(double beta2);

			float getLearningRate() const noexcept;
			void setLearningRate(double learningRate) noexcept;
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
