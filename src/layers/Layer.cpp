/*
 * Layer.cpp
 *
 *  Created on: Nov 10, 2020
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Layer.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Device.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>

#include <Avocado/optimizers/Optimizer.hpp>
#include <Avocado/regularizers/Regularizer.hpp>
#include <Avocado/initializers/Initializer.hpp>
#include <Avocado/initializers/RandomNormal.hpp>
#include <Avocado/initializers/RandomUniform.hpp>

#include <unordered_map>
#include <mutex>

namespace
{
	std::unordered_map<std::string, std::unique_ptr<avocado::Layer>>& registered_layers()
	{
		static std::unordered_map<std::string, std::unique_ptr<avocado::Layer>> result;
		return result;
	}
	size_t get_new_id()
	{
		static std::mutex m;
		static size_t current_id = 0;

		std::lock_guard lock(m);
		current_id++;
		if (current_id == 0)
			throw avocado::RuntimeError(METHOD_NAME, "Layer ID overflow");
		return current_id;
	}
}

namespace avocado
{
	Layer::Layer(const std::string &activation) :
			m_nonlinearity(nonlinearityFromString(activation)),
			m_id(get_new_id())
	{
	}

	NonlinearityType Layer::getNonlinearity() const noexcept
	{
		return m_nonlinearity;
	}
	void Layer::setNonlinearity(NonlinearityType act) noexcept
	{
		m_nonlinearity = act;
	}

	Json Layer::getConfig() const
	{
		Json result;
		result["name"] = name();
		result["nonlinearity"] = toString(m_nonlinearity);
		result["dtype"] = toString(m_dtype);
		return result;
	}
	Json Layer::saveParameters(SerializedObject &binary_data) const
	{
		Json result;
		result["weights"] = (m_weights == nullptr) ? Json() : m_weights->serialize(binary_data);
		result["bias"] = (m_bias == nullptr) ? Json() : m_bias->serialize(binary_data);
		return result;
	}
	void Layer::loadParameters(const Json &json, const SerializedObject &binary_data)
	{
		if (json.hasKey("weights") && !json["weights"].isNull())
			getWeights().unserialize(json["weights"], binary_data);
		if (json.hasKey("bias") && !json["bias"].isNull())
			getBias().unserialize(json["bias"], binary_data);
	}

	int Layer::numberOfInputs() const noexcept
	{
		return static_cast<int>(m_input_shapes.size());
	}
	void Layer::setInputShape(const Shape &shape)
	{
		m_input_shapes = { shape };
	}
	void Layer::setInputShape(const std::vector<Shape> &shapes)
	{
		m_input_shapes = shapes;
	}
	Shape Layer::getInputShape(int index) const
	{
		if (index < 0 || index >= static_cast<int>(m_input_shapes.size()))
			throw IndexOutOfBounds(METHOD_NAME, "index", index, m_input_shapes.size());
		return m_input_shapes[index];
	}
	Shape Layer::getWeightShape() const
	{
		return Shape();
	}
	Shape Layer::getBiasShape() const
	{
		return Shape();
	}

	Device Layer::device() const
	{
		if (m_context == nullptr)
			return Device::cpu();
		else
			return context().device();
	}
	DataType Layer::dtype() const noexcept
	{
		return m_dtype;
	}
	const Context& Layer::context() const
	{
		if (m_context == nullptr)
			throw UninitializedObject(METHOD_NAME, "context was not initialized");
		else
			return *m_context;
	}

	Parameter& Layer::getWeights()
	{
		if (m_weights == nullptr)
		{
			m_weights = std::make_unique<Parameter>(getWeightShape(), dtype(), device());
			m_weights->setInitializer(RandomNormal());
		}
		return *m_weights;
	}
	Parameter& Layer::getBias()
	{
		if (m_bias == nullptr)
		{
			m_bias = std::make_unique<Parameter>(getBiasShape(), dtype(), device());
			m_bias->setInitializer(RandomUniform());
		}
		return *m_bias;
	}
	const Parameter& Layer::getWeights() const
	{
		if (m_weights == nullptr)
			throw UninitializedObject(METHOD_NAME, "weights were not initialized");
		return *m_weights;
	}
	const Parameter& Layer::getBias() const
	{
		if (m_bias == nullptr)
			throw UninitializedObject(METHOD_NAME, "bias was not initialized");
		return *m_bias;
	}

	void Layer::changeContext(Context &context)
	{
		this->m_context = &context;
		if (m_weights != nullptr)
			getWeights().moveTo(device());
		if (m_bias != nullptr)
			getBias().moveTo(device());
	}

	void Layer::init()
	{
		getWeights().init(context());
		getBias().init(context());
	}
	Layer& Layer::setInitializer(const Initializer &initializer)
	{
		getWeights().setInitializer(initializer);
		getBias().setInitializer(initializer);
		return *this;
	}
	Layer& Layer::setOptimizer(const Optimizer &optimizer)
	{
		getWeights().setOptimizer(optimizer);
		getBias().setOptimizer(optimizer);
		return *this;
	}
	Layer& Layer::setRegularizer(const Regularizer &regularizer)
	{
		getWeights().setRegularizer(regularizer);
		getBias().setRegularizer(regularizer);
		return *this;
	}

	void Layer::learn()
	{
		getWeights().learn(context());
		getBias().learn(context());
	}

	bool sameId(const Layer &lhs, const Layer &rhs) noexcept
	{
		return lhs.m_id == rhs.m_id;
	}

	void registerLayer(const Layer &layer)
	{
		if (registered_layers().find(layer.name()) == registered_layers().end())
			registered_layers()[layer.name()] = std::unique_ptr<Layer>(layer.clone(layer.getConfig()));
		else
			throw LogicError(METHOD_NAME, "layer '" + layer.name() + "' has already been registered");
	}
	std::unique_ptr<Layer> loadLayer(const Json &json, const SerializedObject &binary_data)
	{
		auto opt = registered_layers().find(json["name"]);
		if (opt == registered_layers().end())
			throw LogicError(METHOD_NAME, "unknown layer '" + static_cast<std::string>(json["name"]) + "'");

		return std::unique_ptr<Layer>(opt->second->clone(json));
	}

} /* namespace avocado */

