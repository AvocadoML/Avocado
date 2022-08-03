/*
 * SGD.cpp
 *
 *  Created on: Sep 29, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/optimizers/SGD.hpp>
#include <Avocado/layers/Parameter.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/core/Device.hpp>
#include <Avocado/utils/static_block.hpp>

#include <array>

namespace avocado
{
	static_block
	{
		registerOptimizer(SGD());
	}

	SGD::SGD(double learningRate, double beta, bool useNesterov)
	{
		m_config.setType(OptimizerType::SGD);
		m_config.setLearningRate(learningRate);
		m_config.setCoefficients(std::array<double, 4>( { beta, 0.0, 0.0, 0.0 }));
		m_config.setFlags(std::array<bool, 4>( { useNesterov, false, false, false }));
	}

	float SGD::getLearningRate() const noexcept
	{
		return m_config.getLearningRate();
	}
	void SGD::setLearningRate(double learningRate) noexcept
	{
		m_config.setLearningRate(learningRate);
	}
	int SGD::getSteps() const noexcept
	{
		return m_config.getSteps();
	}

	void SGD::restart() noexcept
	{
		m_config.setSteps(0);
		if (m_workspace != nullptr)
			m_workspace->zeroall();
	}
	void SGD::moveTo(Device newDevice)
	{
		m_config.moveTo(newDevice);
		if (m_workspace != nullptr)
			m_workspace->moveTo(newDevice);
	}
	void SGD::learn(const Context &context, Parameter &param)
	{
		assert(same_device(context, param));
		size_t length = param.getParam().volume();
		if (length == 0)
			return;

		if (m_workspace == nullptr)
		{
			if (m_config.getCoefficients()[0] != 0.0)
				m_workspace = std::make_unique<Tensor>(param.shape(), param.dtype(), param.device());
			else
				m_workspace = std::make_unique<Tensor>(Shape(), param.dtype(), param.device());
		}

		math::optimizerLearn(context, m_config, 1, 1, param.getParam(), param.getUpdate(), *m_workspace);
		param.getUpdate().zeroall();
	}

	std::string SGD::name() const
	{
		return "SGD";
	}
	SGD* SGD::clone() const
	{
		std::unique_ptr<SGD> result = std::make_unique<SGD>();
		result->m_config = this->m_config;
		if (this->m_workspace != nullptr)
			result->m_workspace = std::make_unique<Tensor>(*m_workspace);
		return result.release();
	}
	Json SGD::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = name();
		result["steps"] = m_config.getSteps();
		result["learning rate"] = m_config.getLearningRate();
		result["beta"] = m_config.getCoefficients()[0];
		result["use_nesterov"] = m_config.getFlags()[0];
		result["workspace"] = (m_workspace == nullptr) ? Json() : m_workspace->serialize(binary_data);
		return result;
	}
	void SGD::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_config.setLearningRate(json["learning rate"].getDouble());
		m_config.setSteps(json["steps"].getLong());
		std::array<double, 4> coef = { json["beta"].getDouble(), 0.0, 0.0, 0.0 };
		std::array<bool, 4> flags = { json["use_nesterov"].getBool(), false, false, false };
		m_config.setCoefficients(coef);
		m_config.setFlags(flags);
		m_workspace = json["workspace"].isNull() ? nullptr : std::make_unique<Tensor>(json["workspace"], binary_data);
	}

} /* namespace avocado */

