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
			OptimizerType m_type = static_cast<OptimizerType>(-1);
			int64_t m_steps = 0;
			double m_learning_rate = 0.0;
			std::array<double, 4> m_coeficients = { 0.0, 0.0, 0.0, 0.0 };
			std::array<bool, 4> m_flags = { false, false, false, false };
		public:
			OptimizerConfig() = default;
			OptimizerConfig(Device device);
			OptimizerConfig(const OptimizerConfig &other);
			OptimizerConfig(OptimizerConfig &&other) = default;
			OptimizerConfig& operator=(const OptimizerConfig &other);
			OptimizerConfig& operator=(OptimizerConfig &&other) = default;

			Device device() const noexcept;
			void moveTo(Device newDevice);
			void setType(OptimizerType type);
			void setSteps(int64_t steps);
			void setLearningRate(double learningRate);
			void setCoefficients(const std::array<double, 4> &coefficients);
			void setFlags(const std::array<bool, 4> &flags);
			OptimizerType getType() const noexcept;
			double getLearningRate() const noexcept;
			int64_t getSteps() const noexcept;
			const std::array<double, 4>& getCoefficients() const noexcept;
			const std::array<bool, 4>& getFlags() const noexcept;
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

		void optimizerLearn(const Context &context, OptimizerConfig &config, Scalar alpha, Scalar beta, Tensor &weight, const Tensor &update,
				Tensor &workspace);

		Scalar applyRegularizerL2(const Context &context, Tensor &gradient, const Tensor &weight, Tensor &update, Scalar scale, Scalar offset,
				bool calcLoss);

	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_TRAINING_HPP_ */
