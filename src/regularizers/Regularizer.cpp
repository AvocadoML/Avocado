/*
 * Regularizer.cpp
 *
 *  Created on: Oct 13, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/regularizers/Regularizer.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/core/error_handling.hpp>

#include <unordered_map>

namespace
{
	std::unordered_map<std::string, std::unique_ptr<avocado::Regularizer>>& registered_regularizers()
	{
		static std::unordered_map<std::string, std::unique_ptr<avocado::Regularizer>> result;
		return result;
	}
}

namespace avocado
{
	void registerRegularizer(const Regularizer &reg)
	{
		if (registered_regularizers().find(reg.name()) == registered_regularizers().end())
			registered_regularizers()[reg.name()] = std::unique_ptr<Regularizer>(reg.clone());
		else
			throw LogicError(METHOD_NAME, "regularizer '" + reg.name() + "' has already been registered");
	}
	std::unique_ptr<Regularizer> loadRegularizer(const Json &json, const SerializedObject &binary_data)
	{
		auto opt = registered_regularizers().find(json["name"]);
		if (opt == registered_regularizers().end())
			throw LogicError(METHOD_NAME, "unknown regularizer '" + static_cast<std::string>(json["name"]) + "'");

		std::unique_ptr<Regularizer> result(opt->second->clone());
		result->unserialize(json, binary_data);
		return result;
	}

} /* namespace avocado */

