/*
 * Parameter.cpp
 *
 *  Created on: Oct 12, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Parameter.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>

#include <Avocado/initializers/RandomNormal.hpp>

namespace avocado
{

	Parameter::Parameter(const Parameter &other) :
			m_param(other.m_param),
			m_update((other.m_update == nullptr) ? nullptr : std::make_unique<Tensor>(*other.m_update)),
			m_optimizer((other.m_optimizer == nullptr) ? nullptr : other.m_optimizer->clone()),
			m_regularizer((other.m_regularizer == nullptr) ? nullptr : other.m_regularizer->clone()),
			m_initializer(other.m_initializer->clone()),
			m_accumulated_updates(other.m_accumulated_updates),
			m_is_trainable(other.m_is_trainable)
	{
	}
	Parameter& Parameter::operator=(const Parameter &other)
	{
		if (this != &other)
		{
			m_param = other.m_param;
			m_update = (other.m_update == nullptr) ? nullptr : std::make_unique<Tensor>(*other.m_update);
			m_optimizer = (other.m_optimizer == nullptr) ? nullptr : std::unique_ptr<Optimizer>(other.m_optimizer->clone());
			m_regularizer = (other.m_regularizer == nullptr) ? nullptr : std::unique_ptr<Regularizer>(other.m_regularizer->clone());
			m_initializer = std::unique_ptr<Initializer>(other.m_initializer->clone());
			this->m_accumulated_updates = other.m_accumulated_updates;
			this->m_is_trainable = other.m_is_trainable;
		}
		return *this;
	}

	Parameter::Parameter(const Json &json, const SerializedObject &binary_data) :
			m_param(json["param"], binary_data),
			m_accumulated_updates(json["accumulated updates"]),
			m_is_trainable(json["is trainable"])
	{
		if (!json["update"].isNull())
			m_update = std::make_unique<Tensor>(json["update"], binary_data);
		if (!json["optimizer"].isNull())
			m_optimizer = loadOptimizer(json["optimizer"], binary_data);
		if (!json["regularizer"].isNull())
			m_regularizer = loadRegularizer(json["regularizer"], binary_data);
		m_initializer = loadInitializer(json["initializer"], binary_data);
	}
	Parameter::Parameter(const Shape &shape, DataType dtype, Device device, bool trainable) :
			m_param(shape, dtype, device),
			m_initializer(RandomNormal().clone()),
			m_is_trainable(trainable)
	{
	}

	void Parameter::setTrainable(bool t)
	{
		m_is_trainable = t;
		if (t == false)
		{
			m_update = nullptr;
			m_optimizer = nullptr;
			m_regularizer = nullptr;
			m_initializer = nullptr;
		}
	}
	bool Parameter::isTrainable() const noexcept
	{
		return m_is_trainable;
	}

	void Parameter::setOptimizer(const Optimizer &optimizer) noexcept
	{
		this->m_optimizer = std::unique_ptr<Optimizer>(optimizer.clone());
	}
	Optimizer& Parameter::getOptimizer() const
	{
		if (m_optimizer == nullptr)
			throw UninitializedObject(METHOD_NAME, "optimizer has not been set");
		return *m_optimizer;
	}

	void Parameter::setRegularizer(const Regularizer &regularizer) noexcept
	{
		this->m_regularizer = std::unique_ptr<Regularizer>(regularizer.clone());
	}
	Regularizer& Parameter::getRegularizer() const
	{
		if (m_regularizer == nullptr)
			throw UninitializedObject(METHOD_NAME, "regularizer has not been set");
		return *m_regularizer;
	}

	void Parameter::setInitializer(const Initializer &initializer) noexcept
	{
		this->m_initializer = std::unique_ptr<Initializer>(initializer.clone());
	}
	Initializer& Parameter::getInitializer() const
	{
		return *m_initializer;
	}

	Shape Parameter::shape() const noexcept
	{
		return getParam().shape();
	}
	DataType Parameter::dtype() const noexcept
	{
		return getParam().dtype();
	}
	Device Parameter::device() const noexcept
	{
		return getParam().device();
	}
	double Parameter::getInvBatch() const noexcept
	{
		if (m_accumulated_updates == 0)
			return 0.0;
		else
			return 1.0 / m_accumulated_updates;
	}
	int Parameter::getBatch() const noexcept
	{
		return m_accumulated_updates;
	}

	const Tensor& Parameter::getParam() const
	{
		return m_param;
	}
	Tensor& Parameter::getParam()
	{
		return m_param;
	}
	Tensor& Parameter::getUpdate()
	{
		if (isTrainable() == false)
			throw LogicError(METHOD_NAME, "parameter is set as non-trainable");

		if (m_update == nullptr)
			m_update = std::make_unique<Tensor>(shape(), dtype(), device());
		return *m_update;
	}

	void Parameter::moveTo(Device newDevice)
	{
		m_param.moveTo(newDevice);
		if (m_update != nullptr)
			m_update->moveTo(newDevice);
		if (m_optimizer != nullptr)
			m_optimizer->moveTo(newDevice);
	}
	void Parameter::convertTo(const Context &context, DataType newType)
	{
		m_param.convertTo(newType);
	}
	void Parameter::init(const Context &context)
	{
		if (isTrainable())
			getInitializer().init(*this);
	}
	void Parameter::learn(const Context &context)
	{
		if (isTrainable())
		{
			if (m_regularizer != nullptr)
				getRegularizer().apply(context, *this);
			getOptimizer().learn(context, *this);
		}
	}

	Json Parameter::serialize(SerializedObject &binary_data) const
	{
		Json result;
		result["is trainable"] = m_is_trainable;
		result["accumulated updates"] = m_accumulated_updates;
		result["param"] = m_param.serialize(binary_data);
		result["update"] = (m_update == nullptr) ? Json() : m_update->serialize(binary_data);
		result["optimizer"] = (m_optimizer == nullptr) ? Json() : m_optimizer->serialize(binary_data);
		result["regularizer"] = (m_regularizer == nullptr) ? Json() : m_regularizer->serialize(binary_data);
		result["initializer"] = (m_initializer == nullptr) ? Json() : m_initializer->serialize(binary_data);

		return result;
	}
	void Parameter::unserialize(const Json &json, const SerializedObject &binary_data)
	{
		m_param.unserialize(json["param"], binary_data);
		m_accumulated_updates = json["accumulated updates"];
		m_is_trainable = json["is trainable"];
		if (!json["update"].isNull())
			m_update = std::make_unique<Tensor>(json["update"], binary_data);
		if (!json["optimizer"].isNull())
			m_optimizer = loadOptimizer(json["optimizer"], binary_data);
		if (!json["regularizer"].isNull())
			m_regularizer = loadRegularizer(json["regularizer"], binary_data);
		if (!json["initializer"].isNull())
			m_initializer = loadInitializer(json["initializer"], binary_data);
	}

} /* namespace avocado */

