/*
 * Graph.cpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/graph/Graph.hpp>
#include <Avocado/graph/GraphNode.hpp>
#include <Avocado/core/Device.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/layers/Input.hpp>
#include <Avocado/utils/json.hpp>

#include <Avocado/inference/calibration.hpp>

#include <algorithm>

namespace
{
	template<typename T>
	int indexOf(const std::vector<T> &vec, T value)
	{
		for (size_t i = 0; i < vec.size(); i++)
			if (vec[i] == value)
				return i;
		return -1;
	}
	template<typename T>
	void removeByIndex(std::vector<T> &vec, size_t idx)
	{
		if (idx < vec.size())
			vec.erase(vec.begin() + idx);
	}
	template<typename T>
	void removeByValue(std::vector<T> &vec, T value)
	{
		removeByIndex(vec, indexOf(vec, value));
	}
}

namespace avocado
{

	Graph::Graph(Device device) :
			m_context(device)
	{
	}

	Device Graph::device() const noexcept
	{
		return m_context.device();
	}
	DataType Graph::dtype() const noexcept
	{
		return m_datatype;
	}
	const Context& Graph::context() const noexcept
	{
		return m_context;
	}

	GraphNodeID Graph::addInput(const Shape &shape)
	{
		return add_node(Input(shape), { });
	}
	GraphNodeID Graph::add(const Layer &layer, GraphNodeID node)
	{
		return add_node(layer, { node });
	}
	GraphNodeID Graph::add(const Layer &layer, std::initializer_list<GraphNodeID> nodes)
	{
		if (nodes.size() == 0)
			throw LogicError(METHOD_NAME, "nodes list must not be empty");
		return add_node(layer, nodes);
	}
	void Graph::addOutput(GraphNodeID node, const LossFunction &loss)
	{
		m_output_nodes.push_back(get_node(node));
		m_losses.push_back(std::unique_ptr<LossFunction>(loss.clone()));
		m_targets.push_back(nullptr);

		bool successfully_combined = m_losses.back()->tryCombineWith(get_node(node)->getLayer());
		if (successfully_combined)
			get_node(node)->bypassDuringBackward();
	}
	void Graph::addOutput(GraphNodeID node)
	{
		m_output_nodes.push_back(get_node(node));
		m_losses.push_back(nullptr);
		m_targets.push_back(nullptr);
	}

	const Tensor& Graph::getInput(int index) const
	{
		return m_input_nodes.at(index)->getOutputTensor();
	}
	const Tensor& Graph::getOutput(int index) const
	{
		return m_output_nodes.at(index)->getOutputTensor();
	}
	const Tensor& Graph::getGradient(int index) const
	{
		return m_output_nodes.at(index)->getGradientTensor();
	}
	const Tensor& Graph::getTarget(int index) const
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		if (m_targets.at(index) == nullptr)
			throw UninitializedObject(METHOD_NAME, "target tensor was not initialized");
		return *(m_targets.at(index));
	}
	Tensor& Graph::getInput(int index)
	{
		return m_input_nodes.at(index)->getOutputTensor();
	}
	Tensor& Graph::getOutput(int index)
	{
		return m_output_nodes.at(index)->getOutputTensor();
	}
	Tensor& Graph::getGradient(int index)
	{
		return m_output_nodes.at(index)->getGradientTensor();
	}
	Tensor& Graph::getTarget(int index)
	{
		if (m_targets.at(index) == nullptr)
			m_targets.at(index) = std::make_unique<Tensor>(getOutput(index).shape(), dtype(), device());
		return *(m_targets.at(index));
	}

	Shape Graph::getInputShape(int index) const
	{
		return m_input_nodes.at(index)->getOutputShape();
	}
	Shape Graph::getOutputShape(int index) const
	{
		return m_output_nodes.at(index)->getOutputShape();
	}

	int Graph::numberOfInputs() const noexcept
	{
		return static_cast<int>(m_input_nodes.size());
	}
	int Graph::numberOfOutputs() const noexcept
	{
		return static_cast<int>(m_output_nodes.size());
	}
	int Graph::maxBatchSize() const
	{
		if (numberOfInputs() == 0)
			return 0;
		else
			return getOutputShape().firstDim();
	}
	void Graph::moveTo(Device newDevice)
	{
		if (newDevice == device())
			return;

		m_context = Context(newDevice);
		for (size_t i = 0; i < m_layers.size(); i++)
			m_layers.at(i)->changeContext(m_context);
		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes.at(i)->moveTo(newDevice);
		if (m_backup_tensor != nullptr)
			m_backup_tensor->moveTo(newDevice);
		for (size_t i = 0; i < m_targets.size(); i++)
			if (m_targets.at(i) != nullptr)
				m_targets.at(i)->moveTo(newDevice);
	}
	void Graph::setInputShape(const Shape &shape)
	{
		setInputShape(std::vector<Shape>( { shape }));
	}
	void Graph::setInputShape(const std::vector<Shape> &list)
	{
		for (int i = 0; i < numberOfInputs(); i++)
			m_input_nodes.at(i)->getLayer().setInputShape(list[i]);

		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes.at(i)->resolveInputShapes();
		m_backup_tensor = nullptr;
	}

	void Graph::setOptimizer(const Optimizer &optimizer)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		for (size_t i = 0; i < m_layers.size(); i++)
			m_layers.at(i)->setOptimizer(optimizer);
	}
	void Graph::setRegularizer(const Regularizer &regularizer)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		for (size_t i = 0; i < m_layers.size(); i++)
			m_layers.at(i)->setRegularizer(regularizer);
	}
	void Graph::init()
	{
		for (size_t i = 0; i < m_layers.size(); i++)
			m_layers.at(i)->init();
	}
	void Graph::forward(int batchSize)
	{
		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes.at(i)->forward(batchSize);
	}
	void Graph::backward(int batchSize)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");
		if (m_backup_tensor == nullptr)
			create_backup_tensor();
		for (size_t i = 0; i < m_nodes.size(); i++)
			m_nodes.at(i)->prepareForBackward();

		for (size_t i = 0; i < m_targets.size(); i++)
		{
			Shape tmp(getTarget(i).shape());
			tmp[0] = batchSize;
			Tensor gradient = getGradient(i).view(tmp);
			Tensor output = getOutput(i).view(tmp);
			Tensor target = getTarget(i).view(tmp);
			m_losses.at(i)->getGradient(context(), gradient, output, target);
			for (int j = 0; j < 10; j++)
				std::cout << gradient.get<float>( { 0, j }) << " ";
			std::cout << '\n';

			for (int j = 0; j < 10; j++)
				std::cout << output.get<float>( { 0, j }) << " ";
			std::cout << '\n';
			for (int j = 0; j < 10; j++)
				std::cout << output.get<float>( { 1, j }) << " ";
			std::cout << std::endl;

			for (int j = 0; j < 10; j++)
				std::cout << target.get<float>( { 0, j }) << " ";
			std::cout << '\n';
			for (int j = 0; j < 10; j++)
				std::cout << target.get<float>( { 1, j }) << " ";
			std::cout << std::endl << std::endl;
		}

		for (int i = static_cast<int>(m_nodes.size()) - 1; i >= 0; i--)
			m_nodes.at(i)->backward(batchSize, *m_backup_tensor);
	}
	std::vector<Scalar> Graph::getLoss(int batchSize)
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");

		std::vector<Scalar> result(numberOfOutputs());
		for (size_t i = 0; i < m_targets.size(); i++)
		{
			Shape tmp(getTarget(i).shape());
			tmp[0] = batchSize;
			Tensor output = getOutput(i).view(tmp);
			Tensor target = getTarget(i).view(tmp);
			result.at(i) = m_losses.at(i)->getLoss(context(), output, target);
		}
		return result;
	}
	void Graph::learn()
	{
		if (not isTrainable())
			throw LogicError(METHOD_NAME, "Graph is not trainable");

		for (int i = 0; i < numberOfLayers(); i++)
			m_layers.at(i)->learn();
	}

	void Graph::print() const
	{
		for (size_t i = 0; i < m_nodes.size(); i++)
		{
			GraphNode *node = m_nodes.at(i).get();
			std::cout << i << ' ' << m_nodes.at(i)->getLayer().name() << " (" << m_nodes.at(i)->getLayer().getNonlinearity() << ") : "
					<< node->getOutputShape() << " : {";
			for (int j = 0; j < node->numberOfInputs(); j++)
			{
				if (j != 0)
					std::cout << ',';
				std::cout << index_of_node(node->getInputNode(j));
			}
			std::cout << "} -> {";
			for (int j = 0; j < node->numberOfOutputs(); j++)
			{
				if (j != 0)
					std::cout << ',';
				std::cout << index_of_node(node->getOutputNode(j));
			}
			std::cout << "}\n";
		}
		for (size_t i = 0; i < m_output_nodes.size(); i++)
			std::cout << "Output:" << i << " : {" << index_of_node(m_output_nodes.at(i)) << "} : " << m_output_nodes.at(i)->getOutputShape()
					<< std::endl;
	}
	void Graph::makeNonTrainable()
	{
		for (int i = 0; i < numberOfLayers(); i++)
		{
			getLayer(i).getWeights().setTrainable(false);
			getLayer(i).getBias().setTrainable(false);
		}
		for (int i = 0; i < numberOfNodes(); i++)
			getNode(i).makeNonTrainable();
	}
	bool Graph::isTrainable() const noexcept
	{
		return m_targets.size() == m_output_nodes.size();
	}
	void Graph::calibrate(inference::CalibrationTable &table) const
	{
		for (size_t i = 0; i < m_nodes.size(); i++)
		{
			size_t indeOfLayer = index_of_layer(&(m_nodes.at(i)->getLayer()));
			table.getHistogram(indeOfLayer).collectStatistics(m_nodes.at(i)->getOutputTensor());
		}
	}

	int Graph::numberOfLayers() const noexcept
	{
		return static_cast<int>(m_layers.size());
	}
	const Layer& Graph::getLayer(int index) const
	{
		return *(m_layers.at(index));
	}
	Layer& Graph::getLayer(int index)
	{
		return *(m_layers.at(index));
	}
	int Graph::numberOfNodes() const noexcept
	{
		return static_cast<int>(m_nodes.size());
	}
	const GraphNode& Graph::getNode(int index) const
	{
		return *(m_nodes.at(index));
	}
	GraphNode& Graph::getNode(int index)
	{
		return *(m_nodes.at(index));
	}
	GraphNodeID Graph::getNodeID(const GraphNode *node) const noexcept
	{
		return index_of_node(node);
	}

	void Graph::clear()
	{
		m_context = Context();
		m_layers.clear();
		m_nodes.clear();
		m_losses.clear();
		m_targets.clear();

		m_input_nodes.clear();
		m_output_nodes.clear();

		m_backup_tensor.reset();

		m_datatype = DataType::FLOAT32;
	}
	Json Graph::save(SerializedObject &binary_data) const
	{
		Json result;
		result["losses"] = Json(JsonType::Array);
		for (size_t i = 0; i < m_losses.size(); i++)
			result["losses"][i] = m_losses[i]->serialize(binary_data);

		result["layers"] = Json(JsonType::Array);
		for (int i = 0; i < numberOfLayers(); i++)
		{
			Json tmp = getLayer(i).getConfig();
			tmp.append(getLayer(i).saveParameters(binary_data));
			result["layers"][i] = tmp;
		}

		result["nodes"] = Json(JsonType::Array);
		for (int i = 0; i < static_cast<int>(m_nodes.size()); i++)
			result["nodes"][i] = save_node(m_nodes[i].get());
		return result;
	}
	void Graph::load(const Json &json, const SerializedObject &binary_data)
	{
		clear();
		const Json &losses = json["losses"];
		for (int i = 0; i < losses.size(); i++)
		{
			m_losses.push_back(loadLossFunction(losses[i], binary_data));
			m_targets.push_back(nullptr);
		}

		const Json &layers = json["layers"];
		for (int i = 0; i < layers.size(); i++)
		{
			m_layers.push_back(loadLayer(layers[i], binary_data));
			m_layers.back()->changeContext(m_context);
		}

		const Json &nodes = json["nodes"];
		for (int i = 0; i < nodes.size(); i++)
			load_node(nodes[i]);

		for (int i = 0; i < numberOfLayers(); i++)
			getLayer(i).loadParameters(layers[i], binary_data);
	}

	GraphNodeID Graph::add_node(const Layer &layer, const std::vector<GraphNodeID> &inputs)
	{
		m_layers.push_back(std::unique_ptr<Layer>(layer.clone(layer.getConfig())));
		m_layers.back()->changeContext(m_context);

		std::vector<GraphNode*> tmp(inputs.size());
		for (size_t i = 0; i < inputs.size(); i++)
			tmp[i] = get_node(inputs[i]);
		m_nodes.push_back(std::make_unique<GraphNode>(m_layers.back().get(), tmp));

		if (m_nodes.back()->isInputNode())
			m_input_nodes.push_back(m_nodes.back().get());
		return static_cast<GraphNodeID>(m_nodes.size() - 1);
	}
	void Graph::insert_node_with_layer(std::unique_ptr<Layer> &&new_layer, const std::vector<GraphNode*> &inputs,
			const std::vector<GraphNode*> &outputs)
	{
		int last_of_input = 0;
		for (size_t i = 0; i < inputs.size(); i++)
		{
			int tmp = index_of_node(inputs[i]);
			if (tmp == -1)
				throw LogicError(METHOD_NAME, "no such node in this graph");
			last_of_input = std::max(last_of_input, tmp);
		}

		int first_of_output = numberOfNodes();
		for (size_t i = 0; i < outputs.size(); i++)
		{
			int tmp = index_of_node(outputs[i]);
			if (tmp == -1)
				throw LogicError(METHOD_NAME, "no such node in this graph");
			first_of_output = std::min(first_of_output, tmp);
		}

		if (last_of_input > first_of_output)
			throw LogicError(METHOD_NAME, "insertion would form a cycle");

		std::unique_ptr<GraphNode> tmp = std::make_unique<GraphNode>(new_layer.get(), inputs);
		GraphNode::link(tmp.get(), outputs);
		m_nodes.insert(m_nodes.begin() + last_of_input + 1, std::move(tmp));

		new_layer->changeContext(m_context);
		m_layers.push_back(std::move(new_layer));
	}
	void Graph::remove_node(GraphNode *node)
	{
		auto index_in_input_nodes = std::find(m_input_nodes.begin(), m_input_nodes.end(), node);
		auto index_in_output_nodes = std::find(m_output_nodes.begin(), m_output_nodes.end(), node);

		if (index_in_input_nodes != m_input_nodes.end())
		{
			if (node->numberOfOutputs() > 1)
				throw LogicError(METHOD_NAME, "trying to remove input node");
			else
				*index_in_input_nodes = node->getOutputNode(0);
		}
		if (index_in_output_nodes != m_output_nodes.end())
		{
			if (node->numberOfInputs() > 1)
				throw LogicError(METHOD_NAME, "trying to remove output node");
			else
				*index_in_output_nodes = node->getInputNode(0);
		}
		node->removeAllLinks();
		removeByIndex(m_layers, index_of_layer(&(node->getLayer())));
		removeByIndex(m_nodes, index_of_node(node));
	}
	std::unique_ptr<Layer> Graph::replaceLayer(int index, const Layer &newLayer)
	{
		std::unique_ptr<Layer> result = std::move(m_layers[index]);
		m_layers[index] = std::unique_ptr<Layer>(newLayer.clone(newLayer.getConfig()));
		m_layers[index]->changeContext(m_context);

		std::vector<Shape> tmp;
		for (int i = 0; i < result->numberOfInputs(); i++)
			tmp.push_back(result->getInputShape(i));
		m_layers[index]->setInputShape(tmp);

		for (size_t i = 0; i < m_nodes.size(); i++)
			if (&(m_nodes[i]->getLayer()) == result.get())
				m_nodes[i]->replaceLayer(m_layers[index].get());
		return result;
	}

	void Graph::create_backup_tensor()
	{
		int tmp = 0;
		for (size_t i = 0; i < m_nodes.size(); i++)
			tmp = std::max(tmp, m_nodes[i]->getBackupStorage());
		m_backup_tensor = std::make_unique<Tensor>(Shape( { tmp }), dtype(), device());
	}

	Json Graph::save_node(const GraphNode *node) const
	{
		Json result;
		result["is_input_node"] = node->isInputNode();
		result["is_output_node"] = node->isOutputNode();
		result["layer_id"] = index_of_layer(&(node->getLayer()));
		result["input_nodes"] = Json(JsonType::Array);
		for (int i = 0; i < node->numberOfInputs(); i++)
			result["input_nodes"][i] = index_of_node(node->getInputNode(i));
		return result;
	}
	void Graph::load_node(const Json &json)
	{
		Layer *layer = m_layers[static_cast<int>(json["layer_id"])].get();
		std::vector<GraphNode*> inputs;
		for (int i = 0; i < json["input_nodes"].size(); i++)
			inputs.push_back(m_nodes[static_cast<int>(json["input_nodes"][i])].get());
		m_nodes.push_back(std::make_unique<GraphNode>(layer, inputs));
		if (json["is_input_node"])
			m_input_nodes.push_back(m_nodes.back().get());
		if (json["is_output_node"])
			m_output_nodes.push_back(m_nodes.back().get());
	}
	int Graph::index_of_node(const GraphNode *node) const noexcept
	{
		for (size_t i = 0; i < m_nodes.size(); i++)
			if (m_nodes[i].get() == node)
				return i;
		return -1;
	}
	int Graph::index_of_layer(const Layer *layer) const noexcept
	{
		for (size_t i = 0; i < m_layers.size(); i++)
			if (m_layers[i].get() == layer)
				return i;
		return -1;
	}
	const GraphNode* Graph::get_node(GraphNodeID index) const
	{
		if (index < 0 || index >= numberOfNodes())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfNodes());
		return m_nodes[index].get();
	}
	GraphNode* Graph::get_node(GraphNodeID index)
	{
		if (index < 0 || index >= numberOfNodes())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfNodes());
		return m_nodes[index].get();
	}

} /* namespace avocado */

