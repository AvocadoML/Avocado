/*
 * training.hpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_MATH_TRAINING_HPP_
#define AVOCADO_MATH_TRAINING_HPP_

namespace avocado
{
	class Context;
	class Scalar;
	class Tensor;
}

namespace avocado
{
	namespace math
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
		struct OptimizerConfig
		{
				OptimizerType type;
				double learning_rate;
				double coeficients[4];
				bool flags[4];
		};

		Scalar calcMetricFunction(const Context &context, MetricType metricType, const Tensor &output, const Tensor &target);

		Scalar calcLossFunction(const Context &context, LossType lossType, const Tensor &output, const Tensor &target);
		void calcLossGradient(const Context &context, LossType lossType, const Scalar alpha, const Scalar beta, Tensor &gradient,
				const Tensor &output, const Tensor &target, bool isFused);

		void calcOptimizerLearn(const Context &context, const OptimizerConfig &config, const Scalar alpha, const Scalar beta, Tensor &weight,
				const Tensor &update, Tensor &workspace1, Tensor &workspace2);

		Scalar applyRegularizerL2(const Context &context, Tensor &gradient, const Tensor &weight, Tensor &update, const Scalar coefficient,
				const Scalar offset);

	} /* namespace math */
} /* namespace avocado */

#endif /* AVOCADO_MATH_TRAINING_HPP_ */
