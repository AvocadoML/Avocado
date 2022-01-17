/*
 * LossFunction.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/losses/LossFunction.hpp>
#include <Avocado/layers/Layer.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/core/error_handling.hpp>

#include <unordered_map>

namespace
{
	std::unordered_map<std::string, std::unique_ptr<avocado::LossFunction>>& registered_loss_functions()
	{
		static std::unordered_map<std::string, std::unique_ptr<avocado::LossFunction>> result;
		return result;
	}
}

namespace avocado
{
	bool LossFunction::tryCombineWith(const Layer &layer) noexcept
	{
		return false;
	}

	Json LossFunction::serialize(SerializedObject &binary_data) const
	{
		return Json( { { "name", this->name() } });
	}
	void LossFunction::unserialize(const Json &json, const SerializedObject &binary_data)
	{
	}

	void registerLossFunction(const LossFunction &loss)
	{
		if (registered_loss_functions().find(loss.name()) == registered_loss_functions().end())
			registered_loss_functions()[loss.name()] = std::unique_ptr<LossFunction>(loss.clone());
		else
			throw LogicError(METHOD_NAME, "loss function '" + loss.name() + "' has already been registered");
	}
	std::unique_ptr<LossFunction> loadLossFunction(const Json &json, const SerializedObject &binary_data)
	{
		auto opt = registered_loss_functions().find(json["name"]);
		if (opt == registered_loss_functions().end())
			throw LogicError(METHOD_NAME, "unknown loss function '" + static_cast<std::string>(json["name"]) + "'");

		std::unique_ptr<LossFunction> result(opt->second->clone());
		result->unserialize(json, binary_data);
		return result;
	}

} /* namespace avocado */

