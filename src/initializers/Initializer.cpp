/*
 * Initializer.cpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/initializers/Initializer.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/core/error_handling.hpp>

#include <unordered_map>

namespace
{
	std::unordered_map<std::string, std::unique_ptr<avocado::Initializer>>& registered_initializers()
	{
		static std::unordered_map<std::string, std::unique_ptr<avocado::Initializer>> result;
		return result;
	}
}

namespace avocado
{
	void registerInitializer(const Initializer &reg)
	{
		if (registered_initializers().find(reg.name()) == registered_initializers().end())
			registered_initializers()[reg.name()] = std::unique_ptr<Initializer>(reg.clone());
		else
			throw LogicError(METHOD_NAME, "initializer '" + reg.name() + "' has already been registered");
	}
	std::unique_ptr<Initializer> loadInitializer(const Json &json, const SerializedObject &binary_data)
	{
		auto opt = registered_initializers().find(json["name"]);
		if (opt == registered_initializers().end())
			throw LogicError(METHOD_NAME, "unknown initializer '" + static_cast<std::string>(json["name"]) + "'");

		std::unique_ptr<Initializer> result(opt->second->clone());
		result->unserialize(json, binary_data);
		return result;
	}

} /* namespace avocado */

