/*
 * ADAM.cpp
 *
 *  Created on: Feb 24, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/optimizers/ADAM.hpp>
#include <Avocado/layers/Parameter.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
		registerOptimizer(ADAM());
	}

	ADAM::ADAM(double learningRate, bool useAMSGrad)
	{
		m_config.setType(OptimizerType::ADAM);
		m_config.setLearningRate(learningRate);
		m_config.setFlags(std::array<bool, 4>( { useAMSGrad, false, false, false }));
	}

	ADAM& ADAM::setBeta1(double beta1)
	{
		assert(beta1 >= 0.0f && beta1 <= 1.0f);
		std::array<double, 4> tmp = m_config.getCoefficients();
		tmp[0] = beta1;
		m_config.setCoefficients(tmp);
		return *this;
	}
	ADAM& ADAM::setBeta2(double beta2)
	{
		assert(beta2 >= 0.0f && beta2 <= 1.0f);
		std::array<double, 4> tmp = m_config.getCoefficients();
		tmp[1] = beta2;
		m_config.setCoefficients(tmp);
		return *this;
	}

	float ADAM::getLearningRate() const noexcept
	{
		return m_config.getLearningRate();
	}
	void ADAM::setLearningRate(double learningRate) noexcept
	{
		m_config.setLearningRate(learningRate);
	}
	int ADAM::getSteps() const noexcept
	{
		return m_config.getSteps();
	}

	void ADAM::restart() noexcept
	{
		m_config.setSteps(0);
		if (m_workspace != nullptr)
			m_workspace->zeroall();
	}
	void ADAM::moveTo(Device newDevice)
	{
		m_config.moveTo(newDevice);
		if (m_workspace != nullptr)
			m_workspace->moveTo(newDevice);
	}
	void ADAM::learn(const Context &context, Parameter &param)
	{
		assert(same_device(context, param));
		size_t length = param.getParam().volume();
		if (length == 0)
			return;

		if (m_workspace == nullptr)
			m_workspace = std::make_unique<Tensor>(Shape( { 2 * param.shape().volume() }), param.dtype(), param.device());

		math::optimizerLearn(context, m_config, 1, 1, param.getParam(), param.getUpdate(), *m_workspace);
		param.getUpdate().zeroall();
	}

	std::string ADAM::name() const
	{
		return "ADAM";
	}
	ADAM* ADAM::clone() const
	{
		std::unique_ptr<ADAM> result = std::make_unique<ADAM>();
		result->m_config = this->m_config;
		if (this->m_workspace != nullptr)
			result->m_workspace = std::make_unique<Tensor>(*m_workspace);
		return result.release();
	}
	Json ADAM::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = name();
		result["steps"] = m_config.getSteps();
		result["learning rate"] = m_config.getLearningRate();
		result["beta1"] = m_config.getCoefficients()[0];
		result["beta2"] = m_config.getCoefficients()[1];
		result["use_amsgrad"] = m_config.getFlags()[0];
		result["workspace"] = (m_workspace == nullptr) ? Json() : m_workspace->serialize(binary_data);

		return result;
	}
	void ADAM::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_config.setLearningRate(json["learning rate"].getDouble());
		m_config.setSteps(json["steps"].getLong());
		std::array<double, 4> coef = { json["beta1"].getDouble(), json["beta2"].getDouble(), 0.0, 0.0 };
		std::array<bool, 4> flags = { json["use_amsgrad"].getBool(), false, false, false };
		m_config.setCoefficients(coef);
		m_config.setFlags(flags);
		m_workspace = json["workspace"].isNull() ? nullptr : std::make_unique<Tensor>(json["workspace"], binary_data);
	}

} /* namespace avocado */

