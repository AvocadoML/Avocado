/*
 * Metric.hpp
 *
 *  Created on: Jul 2, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_METRICS_METRIC_HPP_
#define AVOCADO_METRICS_METRIC_HPP_

#include <memory>
#include <string>
#include <stdexcept>

namespace avocado /* forward declarations */
{
	class Json;
	class SerializedObject;
	class Scalar;
	class Tensor;
	class Context;
	class Layer;
}

namespace avocado
{

	class Metric
	{
		public:
			Metric() = default;
			Metric(const Metric &other) = delete;
			Metric(Metric &&other) = delete;
			Metric& operator=(const Metric &other) = delete;
			Metric& operator=(Metric &&other) = delete;
			virtual ~Metric() = default;

			virtual Scalar getMetric(const Context &context, const Tensor &output, const Tensor &target, Tensor &workspace) const = 0;

			virtual std::string name() const = 0;
			virtual Metric* clone() const = 0;
			virtual Json serialize(SerializedObject &binary_data) const;
			virtual void unserialize(const Json &json, const SerializedObject &binary_data);
	};

	void registerMetric(const Metric &metric);
	std::unique_ptr<Metric> loadMetric(const Json &json, const SerializedObject &binary_data);

} /* namespace avocado */

#endif /* AVOCADO_METRICS_METRIC_HPP_ */
