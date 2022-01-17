/*
 * Optimizer.cpp
 *
 *  Created on: Sep 29, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/optimizers/Optimizer.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/core/error_handling.hpp>

#include <unordered_map>

namespace
{
	std::unordered_map<std::string, std::unique_ptr<avocado::Optimizer>>& registered_optimizers()
	{
		static std::unordered_map<std::string, std::unique_ptr<avocado::Optimizer>> result;
		return result;
	}
}

namespace avocado
{
	void registerOptimizer(const Optimizer &opt)
	{
		if (registered_optimizers().find(opt.name()) == registered_optimizers().end())
			registered_optimizers()[opt.name()] = std::unique_ptr<Optimizer>(opt.clone());
		else
			throw LogicError(METHOD_NAME, "optimizer '" + opt.name() + "' has already been registered");
	}
	std::unique_ptr<Optimizer> loadOptimizer(const Json &json, const SerializedObject &binary_data)
	{
		auto opt = registered_optimizers().find(json["name"]);
		if (opt == registered_optimizers().end())
			throw LogicError(METHOD_NAME, "unknown optimizer '" + static_cast<std::string>(json["name"]) + "'");

		std::unique_ptr<Optimizer> result(opt->second->clone());
		result->unserialize(json, binary_data);
		return result;
	}

} /* namespace avocado */

