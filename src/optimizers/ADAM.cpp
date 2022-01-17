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
#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerOptimizer(ADAM());
	}

	ADAM::ADAM(float learningRate, bool useAMSGrad) :
			m_learning_rate(learningRate),
			m_use_amsgrad(useAMSGrad)
	{
	}

	ADAM& ADAM::setBeta1(float b)
	{
		assert(b >= 0.0f && b <= 1.0f);
		m_beta1 = b;
		return *this;
	}
	ADAM& ADAM::setBeta2(float b)
	{
		assert(b >= 0.0f && b <= 1.0f);
		m_beta2 = b;
		return *this;
	}

	float ADAM::getLearningRate() const noexcept
	{
		return m_learning_rate;
	}
	void ADAM::setLearningRate(float lr) noexcept
	{
		this->m_learning_rate = lr;
	}
	int ADAM::getSteps() const noexcept
	{
		return m_steps;
	}

	void ADAM::restart() noexcept
	{
		m_steps = 0;
		if (m_momentum != nullptr)
			m_momentum->zeroall();
		if (m_variance != nullptr)
			m_variance->zeroall();
	}
	void ADAM::moveTo(Device newDevice)
	{
		if (m_momentum != nullptr)
			m_momentum->moveTo(newDevice);
		if (m_variance != nullptr)
			m_variance->moveTo(newDevice);
	}
	void ADAM::learn(const Context &context, Parameter &param)
	{
		assert(same_device(context, param));
		size_t length = param.getParam().volume();
		if (length == 0)
			return;

		if (m_momentum == nullptr)
			m_momentum = std::make_unique<Tensor>(param.shape(), param.dtype(), param.device());
		if (m_variance == nullptr)
			m_variance = std::make_unique<Tensor>(param.shape(), param.dtype(), param.device());

		m_steps++;
//		float *ptr_weight = param.getParam().data<float>();
//		float *ptr_update = param.getUpdate().data<float>();
//		float *ptr_momentum = m_momentum->data<float>();
//		float *ptr_variance = m_variance->data<float>();
		float learning_rate = m_learning_rate;
		if (m_steps < 10000)
			learning_rate *= sqrt(1.0f - pow(m_beta2, m_steps)) / (1.0f - pow(m_beta1, m_steps));

		switch (param.device().type())
		{
			case DeviceType::CPU:
			{
//				internal::cpu_learn_adam(ptr_weight, ptr_update, ptr_momentum, ptr_variance, length, learning_rate, m_beta1, m_beta2);
				break;
			}
			case DeviceType::CUDA:
			{
//				int status = internal::cuda_learn_adam(context.getCudaStream(), ptr_weight, ptr_update, ptr_momentum, ptr_variance, length,
//						learning_rate, m_beta1, m_beta2);
//				if (status != 0)
//					throw CudaRuntimeError(METHOD_NAME, static_cast<cudaError_t>(status));
				break;
			}
		}

	}

	std::string ADAM::name() const
	{
		return "ADAM";
	}
	ADAM* ADAM::clone() const
	{
		std::unique_ptr<ADAM> result = std::make_unique<ADAM>();
		result->m_learning_rate = this->m_learning_rate;
		result->m_beta1 = this->m_beta1;
		result->m_beta2 = this->m_beta2;
		result->m_steps = this->m_steps;
		result->m_use_amsgrad = this->m_use_amsgrad;
		if (this->m_momentum != nullptr)
			result->m_momentum = std::make_unique<Tensor>(*m_momentum);
		if (this->m_variance != nullptr)
			result->m_variance = std::make_unique<Tensor>(*m_variance);
		return result.release();
	}
	Json ADAM::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["name"] = name();
		result["learning rate"] = m_learning_rate;
		result["beta1"] = m_beta1;
		result["beta2"] = m_beta2;
		result["steps"] = m_steps;
		result["use_amsgrad"] = m_use_amsgrad;
		result["momentum"] = (m_momentum == nullptr) ? Json() : m_momentum->serialize(binary_data);
		result["variance"] = (m_variance == nullptr) ? Json() : m_variance->serialize(binary_data);

		return result;
	}
	void ADAM::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_learning_rate = json["learning rate"];
		m_beta1 = json["beta1"];
		m_beta2 = json["beta2"];
		m_steps = json["steps"];
		m_use_amsgrad = json["use_amsgrad"];
		m_momentum = json["momentum"].isNull() ? nullptr : std::make_unique<Tensor>(json["momentum"], binary_data);
		m_variance = json["variance"].isNull() ? nullptr : std::make_unique<Tensor>(json["variance"], binary_data);
	}

} /* namespace avocado */

