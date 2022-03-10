/*
 * training.hpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_TRAINING_HPP_
#define AVOCADO_MATH_TRAINING_HPP_

#include <Avocado/math/descriptor_wrappers.hpp>

#include <array>

namespace avocado
{
	class Context;
	class Scalar;
	class Tensor;
}

namespace avocado
{
	enum class MetricType
	{
		ACCURACY
	};

	enum class LossType
	{
		MEAN_SQUARE_LOSS,
		CROSS_ENTROPY_LOSS,
		KL_DIVERGECE_LOSS
	};

	enum class OptimizerType
	{
		SGD,
		ADAM
	};
	class OptimizerConfig
	{
		private:
			avocado::internal::OptimizerDescWrapper m_descriptor;
			OptimizerType m_type;
			double m_learning_rate;
			std::array<double, 4> m_coeficients;
			std::array<bool, 4> m_flags;
		public:

			void setType(OptimizerType type)
			{
				m_type = type;
				m_descriptor.set(m_type, m_learning_rate, m_coeficients, m_flags);
			}
			void setLearningRate(double learningRate)
			{
				m_learning_rate = learningRate;
				m_descriptor.set(m_type, m_learning_rate, m_coeficients, m_flags);
			}
			void setCoefficients(const std::array<double, 4> &coefficients)
			{
				m_coeficients = coefficients;
				m_descriptor.set(m_type, m_learning_rate, m_coeficients, m_flags);
			}
			void setFlags(const std::array<bool, 4> &flags)
			{
				m_flags = flags;
				m_descriptor.set(m_type, m_learning_rate, m_coeficients, m_flags);
			}

			OptimizerType getType() const noexcept
			{
				return m_type;
			}
			double getLearningRate() const noexcept
			{
				return m_learning_rate;
			}
			const std::array<double, 4>& getCoefficients() const noexcept
			{
				return m_coeficients;
			}
			const std::array<bool, 4>& getFlags() const noexcept
			{
				return m_flags;
			}
			operator backend::avOptimizerDescriptor_t() const noexcept
			{
				return static_cast<backend::avOptimizerDescriptor_t>(m_descriptor);
			}
	};

	namespace math
	{

		Scalar calcMetricFunction(const Context &context, MetricType metricType, const Tensor &output, const Tensor &target);

		Scalar calcLossFunction(const Context &context, LossType lossType, const Tensor &output, const Tensor &target);
		void calcLossGradient(const Context &context, LossType lossType, Scalar alpha, Scalar beta, Tensor &gradient, const Tensor &output,
				const Tensor &target, bool isFused);

		void calcOptimizerLearn(const Context &context, const OptimizerConfig &config, Scalar alpha, Scalar beta, Tensor &weight,
				const Tensor &update, Tensor &workspace);

		Scalar applyRegularizerL2(const Context &context, Tensor &gradient, const Tensor &weight, Tensor &update, Scalar scale, Scalar offset,
				bool calcLoss);

	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_TRAINING_HPP_ */
