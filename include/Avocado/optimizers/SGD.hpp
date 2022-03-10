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
#include <Avocado/math/training.hpp>

#include <memory>

namespace avocado
{
	class SGD: public Optimizer
	{
		private:
			std::unique_ptr<Tensor> m_workspace;
			OptimizerConfig m_config;
		public:
			SGD() = default;
			/**
			 * For details of nesterov see http://proceedings.mlr.press/v28/sutskever13.pdf
			 * @param learningRate
			 * @param momentum
			 * @param nesterov
			 */
			SGD(double learningRate, double beta = 0.0, bool useNesterov = false);

			float getLearningRate() const noexcept;
			void setLearningRate(double lr) noexcept;
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
