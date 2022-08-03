/*
 * GraphNode.cpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/graph/GraphNode.hpp>
#include <Avocado/layers/Layer.hpp>
#include <Avocado/math/tensor_operations.hpp>
#include <Avocado/core/error_handling.hpp>
#include <Avocado/core/Scalar.hpp>

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
	void removeByIndex(std::vector<T> &vec, int idx)
	{
		if (idx < static_cast<int>(vec.size()) && idx >= 0)
			vec.erase(vec.begin() + idx);
	}
	template<typename T>
	void removeByValue(std::vector<T> &vec, T value)
	{
		int tmp = indexOf(vec, value);
		if (tmp == -1)
			throw std::logic_error("no such value");
		removeByIndex(vec, tmp);
	}

	avocado::Tensor change_batch(int batch_size, avocado::Tensor &other)
	{
		avocado::Shape tmp(other.shape());
		tmp[0] = batch_size;
		return other.view(tmp);
	}
}

namespace avocado
{

	GraphNode::GraphNode(Layer *layer, const std::vector<GraphNode*> input_nodes) :
			m_layer(layer)
	{
		if (layer == nullptr)
			throw UninitializedObject(METHOD_NAME, "trying to assign null Layer");
		GraphNode::link(input_nodes, this);
		resolveInputShapes();
	}
	bool GraphNode::isInputNode() const noexcept
	{
		return numberOfInputs() == 0;
	}
	bool GraphNode::isOutputNode() const noexcept
	{
		return numberOfOutputs() == 0;
	}
	Shape GraphNode::getOutputShape() const noexcept
	{
		return m_output_shape;
	}
	void GraphNode::resolveInputShapes()
	{
		if (!isInputNode())
		{
			std::vector<Shape> input_shapes(numberOfInputs());
			for (int i = 0; i < numberOfInputs(); i++)
				input_shapes[i] = getInputNode(i)->getOutputShape();
			getLayer().setInputShape(input_shapes);
		}
		m_output_shape = getLayer().getOutputShape();
	}
	int GraphNode::getBackupStorage()
	{
		int tmp_size = 0;
		for (int i = 0; i < numberOfInputs(); i++)
		{
			if (getInputNode(i)->m_done_backward == true)
				tmp_size += getInputNode(i)->m_output_shape.volume();
			else
				getInputNode(i)->m_done_backward = true;
		}
		return tmp_size;
	}

	int GraphNode::numberOfInputs() const noexcept
	{
		return static_cast<int>(m_input_nodes.size());
	}
	int GraphNode::numberOfOutputs() const noexcept
	{
		return static_cast<int>(m_output_nodes.size());
	}
	const GraphNode* GraphNode::getInputNode(int index) const
	{
		if (index < 0 || index >= numberOfInputs())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfInputs());
		return m_input_nodes[index];
	}
	GraphNode* GraphNode::getInputNode(int index)
	{
		if (index < 0 || index >= numberOfInputs())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfInputs());
		return m_input_nodes[index];
	}
	const GraphNode* GraphNode::getOutputNode(int index) const
	{
		if (index < 0 || index >= numberOfOutputs())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfOutputs());
		return m_output_nodes[index];
	}
	GraphNode* GraphNode::getOutputNode(int index)
	{
		if (index < 0 || index >= numberOfOutputs())
			throw IndexOutOfBounds(METHOD_NAME, "index", index, numberOfOutputs());
		return m_output_nodes[index];
	}
	std::vector<GraphNode*> GraphNode::getInputs() const
	{
		return m_input_nodes;
	}
	std::vector<GraphNode*> GraphNode::getOutputs() const
	{
		return m_output_nodes;
	}

	void GraphNode::forward(int batchSize)
	{
		if (this->isInputNode())
			return;

		std::vector<Tensor> input(numberOfInputs());
		for (int i = 0; i < numberOfInputs(); i++)
			input[i] = change_batch(batchSize, getInputNode(i)->getOutputTensor());
		Tensor output = change_batch(batchSize, this->getOutputTensor());

		getLayer().forward(input, output, 1, 0);
	}
	void GraphNode::backward(int batchSize, Tensor &backup_tensor)
	{
		if (isInputNode())
			return;

		std::vector<Tensor> input(numberOfInputs());
		std::vector<Tensor> gradient_prev(numberOfInputs());
		size_t offset = 0;
		for (int i = 0; i < numberOfInputs(); i++)
		{
			input[i] = change_batch(batchSize, getInputNode(i)->getOutputTensor());
			if (getInputNode(i)->m_done_backward == true) // gradient is propagated into temporary tensor and later added with the proper one
			{
				Shape tmp_shape(getInputNode(i)->getOutputShape());
				tmp_shape[0] = batchSize;
				gradient_prev[i] = backup_tensor.view(tmp_shape, offset);
				offset += gradient_prev[i].volume();
			}
			else
				gradient_prev[i] = change_batch(batchSize, getInputNode(i)->getGradientTensor());
		}
		Tensor output = change_batch(batchSize, this->getOutputTensor());
		Tensor gradient_next = change_batch(batchSize, this->getGradientTensor());

		if (m_is_bypassed_during_backward)
			math::copyTensor(m_layer->context(), gradient_prev[0], gradient_next);
		else
			m_layer->backward(input, output, gradient_prev, gradient_next, 1, 0);

		for (int i = 0; i < numberOfInputs(); i++)
		{
			if (getInputNode(i)->m_done_backward == true) // here the temporary gradient tensor is added to the appropriate tensor
			{
				Tensor tmp = getInputNode(i)->getGradientTensor().view(gradient_prev[i].shape());
				math::addTensors(m_layer->context(), gradient_prev[i], tmp, 1, 1);
			}
			else
				getInputNode(i)->m_done_backward = true;
		}
	}
	void GraphNode::prepareForBackward()
	{
		m_done_backward = false;
		getGradientTensor().zeroall();
	}

	const Layer& GraphNode::getLayer() const
	{
		if (m_layer == nullptr)
			throw LogicError(METHOD_NAME, "layer is null");
		return *m_layer;
	}
	Layer& GraphNode::getLayer()
	{
		if (m_layer == nullptr)
			throw LogicError(METHOD_NAME, "layer is null");
		return *m_layer;
	}
	const Tensor& GraphNode::getOutputTensor() const
	{
		if (m_output_tensor == nullptr)
			throw LogicError(METHOD_NAME, "output tensor is null");
		return *m_output_tensor;
	}
	Tensor& GraphNode::getOutputTensor()
	{
		if (m_output_tensor == nullptr)
			m_output_tensor = std::make_unique<Tensor>(getOutputShape(), getLayer().dtype(), getLayer().device());
		return *m_output_tensor;
	}
	const Tensor& GraphNode::getGradientTensor() const
	{
		if (m_gradient_tensor == nullptr)
			throw LogicError(METHOD_NAME, "gradient tensor is null");
		return *m_gradient_tensor;
	}
	Tensor& GraphNode::getGradientTensor()
	{
		if (m_gradient_tensor == nullptr)
			m_gradient_tensor = std::make_unique<Tensor>(getOutputShape(), getLayer().dtype(), getLayer().device());
		return *m_gradient_tensor;
	}

	void GraphNode::moveTo(Device newDevice)
	{
		if (m_output_tensor != nullptr)
			m_output_tensor->moveTo(newDevice);
		if (m_gradient_tensor != nullptr)
			m_gradient_tensor->moveTo(newDevice);
	}
	void GraphNode::makeNonTrainable() noexcept
	{
		m_gradient_tensor = nullptr;
	}
	void GraphNode::bypassDuringBackward() noexcept
	{
		m_is_bypassed_during_backward = false;
	}

	void GraphNode::link(GraphNode *prev, GraphNode *next)
	{
		prev->m_output_nodes.push_back(next);
		next->m_input_nodes.push_back(prev);
	}
	void GraphNode::link(const std::vector<GraphNode*> &prev, GraphNode *next)
	{
		for (size_t i = 0; i < prev.size(); i++)
		{
			prev[i]->m_output_nodes.push_back(next);
			next->m_input_nodes.push_back(prev[i]);
		}
	}
	void GraphNode::link(GraphNode *prev, const std::vector<GraphNode*> &next)
	{
		for (size_t i = 0; i < next.size(); i++)
		{
			prev->m_output_nodes.push_back(next[i]);
			next[i]->m_input_nodes.push_back(prev);
		}
	}
	void GraphNode::link(const std::vector<GraphNode*> &prev, const std::vector<GraphNode*> &next)
	{
		for (size_t i = 0; i < prev.size(); i++)
			for (size_t j = 0; j < next.size(); j++)
			{
				prev[i]->m_output_nodes.push_back(next[j]);
				next[j]->m_input_nodes.push_back(prev[i]);
			}
	}
	void GraphNode::removeLink(GraphNode *prev, GraphNode *next)
	{
		removeByValue(prev->m_output_nodes, next);
		removeByValue(next->m_input_nodes, prev);
	}
	void GraphNode::removeAllLinks()
	{
		while (m_input_nodes.size() > 0)
			removeLink(m_input_nodes[0], this);
		while (m_output_nodes.size() > 0)
			removeLink(this, m_output_nodes[0]);
	}

	void GraphNode::replaceLayer(Layer *new_layer)
	{
		if (getLayer().numberOfInputs() != new_layer->numberOfInputs())
			throw ShapeMismatch(METHOD_NAME, getLayer().numberOfInputs(), new_layer->numberOfInputs());

		for (int i = 0; i < getLayer().numberOfInputs(); i++)
			if (getLayer().getInputShape(i) != new_layer->getInputShape(i))
				throw ShapeMismatch(METHOD_NAME, getLayer().getInputShape(i), new_layer->getInputShape(i));

		if (getLayer().getOutputShape() != new_layer->getOutputShape())
			throw ShapeMismatch(METHOD_NAME, getLayer().getOutputShape(), new_layer->getOutputShape());

		m_layer = new_layer;
	}

} /* namespace avocado */

