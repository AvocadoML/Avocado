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
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerOptimizer(SGD());
	}

	SGD::SGD(float learningRate, float momentum, bool useNesterov) :
			m_learning_rate(learningRate),
			m_beta(momentum),
			m_use_nesterov(useNesterov)
	{
	}

	float SGD::getLearningRate() const noexcept
	{
		return m_learning_rate;
	}
	void SGD::setLearningRate(float lr) noexcept
	{
		this->m_learning_rate = lr;
	}
	int SGD::getSteps() const noexcept
	{
		return m_steps;
	}

	void SGD::restart() noexcept
	{
		m_steps = 0;
		if (m_momentum != nullptr)
			m_momentum->zeroall();
	}
	void SGD::moveTo(Device newDevice)
	{
		if (m_momentum != nullptr)
			m_momentum->moveTo(newDevice);
	}
	void SGD::learn(const Context &context, Parameter &param)
	{
		assert(same_device(context, param));
		if (m_beta != 0.0f && m_momentum == nullptr)
			m_momentum = std::make_unique<Tensor>(param.shape(), param.dtype(), param.device());

		size_t length = param.getParam().volume();
		if (length == 0)
			return;
//		float *ptr_weight = param.getParam().data<float>();
//		float *ptr_update = param.getUpdate().data<float>();
//		float *ptr_momentum = (m_momentum == nullptr) ? nullptr : m_momentum->data<float>();
//		float learning_rate = m_learning_rate;

		switch (param.device().type())
		{
			case DeviceType::CPU:
			{
//				internal::cpu_learn_sgd(ptr_weight, ptr_update, ptr_momentum, length, learning_rate, m_beta, m_use_nesterov);
				break;
			}
			case DeviceType::CUDA:
			{
//				internal::cuda_learn_sgd(context.getCudaStream(), ptr_weight, ptr_update, ptr_momentum, length, learning_rate, m_beta, m_use_nesterov);
				break;
			}
		}
		m_steps++;
	}

	std::string SGD::name() const
	{
		return "SGD";
	}
	SGD* SGD::clone() const
	{
		std::unique_ptr<SGD> result = std::make_unique<SGD>();
		result->m_learning_rate = this->m_learning_rate;
		result->m_beta = this->m_beta;
		result->m_steps = this->m_steps;
		result->m_use_nesterov = this->m_use_nesterov;
		if (this->m_momentum != nullptr)
			result->m_momentum = std::make_unique<Tensor>(*m_momentum);
		return result.release();
	}
	Json SGD::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = name();
		result["learning rate"] = m_learning_rate;
		result["beta"] = m_beta;
		result["steps"] = m_steps;
		result["use_nesterov"] = m_use_nesterov;
		result["momentum"] = (m_momentum == nullptr) ? Json() : m_momentum->serialize(binary_data);

		return result;
	}
	void SGD::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_learning_rate = json["learning rate"];
		m_beta = json["beta"];
		m_steps = json["steps"];
		m_use_nesterov = json["use_nesterov"];
		m_momentum = json["momentum"].isNull() ? nullptr : std::make_unique<Tensor>(json["momentum"], binary_data);
	}

} /* namespace avocado */

