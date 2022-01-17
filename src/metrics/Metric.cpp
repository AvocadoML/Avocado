/*
 * Metric.cpp
 *
 *  Created on: Nov 6, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/metrics/Metric.hpp>
#include <Avocado/layers/Layer.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/core/error_handling.hpp>

#include <unordered_map>

namespace
{
	std::unordered_map<std::string, std::unique_ptr<avocado::Metric>>& registered_metric()
	{
		static std::unordered_map<std::string, std::unique_ptr<avocado::Metric>> result;
		return result;
	}
}

namespace avocado
{
	Json Metric::serialize(SerializedObject &binary_data) const
	{
		return Json( { { "name", this->name() } });
	}
	void Metric::unserialize(const Json &json, const SerializedObject &binary_data)
	{
	}

	void registerMetric(const Metric &metric)
	{
		if (registered_metric().find(metric.name()) == registered_metric().end())
			registered_metric()[metric.name()] = std::unique_ptr<Metric>(metric.clone());
		else
			throw LogicError(METHOD_NAME, "loss function '" + metric.name() + "' has already been registered");
	}
	std::unique_ptr<Metric> loadMetric(const Json &json, const SerializedObject &binary_data)
	{
		auto opt = registered_metric().find(json["name"]);
		if (opt == registered_metric().end())
			throw LogicError(METHOD_NAME, "unknown loss function '" + static_cast<std::string>(json["name"]) + "'");

		std::unique_ptr<Metric> result(opt->second->clone());
		result->unserialize(json, binary_data);
		return result;
	}

} /* namespace avocado */


