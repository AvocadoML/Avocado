/*
 * Node.cpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/Node.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/core/error_handling.hpp>

#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace avocado
{
	namespace nodes
	{
		Node* Node::clone() const
		{
			return new Node();
		}
		void Node::calculateOutputShape()
		{
		}
		const Shape& Node::getOutputShape() const noexcept
		{
			return m_output_shape;
		}

		Expression& Node::getExpression() const
		{
			if (m_expression == nullptr)
				throw std::logic_error("Node::getExpression() : expression is null");
			return *m_expression;
		}
		void Node::setExpression(Expression &e) noexcept
		{
			m_expression = &e;
		}
		void Node::setIndex(size_t i) noexcept
		{
			m_index = i;
		}
		size_t Node::getIndex() const noexcept
		{
			return m_index;
		}

		size_t Node::numberOfInputs() const noexcept
		{
			return m_inputs.size();
		}
		Node& Node::getInput(size_t index)
		{
			return *(m_inputs.at(index).lock());
		}
		const Node& Node::getInput(size_t index) const
		{
			return *(m_inputs.at(index).lock());
		}
		std::weak_ptr<Node> Node::getInputNodePointer(size_t index) const
		{
			return m_inputs.at(index);
		}

		size_t Node::numberOfOutputs() const noexcept
		{
			return m_outputs.size();
		}
		Node& Node::getOutput(size_t index)
		{
			return *(m_outputs.at(index).lock());
		}
		const Node& Node::getOutput(size_t index) const
		{
			return *(m_outputs.at(index).lock());
		}
		std::weak_ptr<Node> Node::getOutputNodePointer(size_t index) const
		{
			return m_outputs.at(index);
		}

		std::string Node::text() const
		{
			return "x" + std::to_string(m_index) + ":" + getOutputShape().toString();
		}
		std::string Node::toString() const
		{
			return "";
		}
		Expression Node::getBackprop() const
		{
			return Expression();
//			Expression result;
//			switch (m_outputs.size())
//			{
//				case 0:
//					break;
//				case 1:
//				{
//					auto x = result.input();
//					result.output(x);
//					break;
//				}
//				default:
//				{
//					auto x = result.input();
//					for (size_t i = 1; i < m_outputs.size(); i++)
//					{
//						auto y = result.input();
//						x = x + y;
//					}
//					result.output(x);
//					break;
//				}
//			}
//			return result;
		}

		bool Node::areLinked(const std::weak_ptr<Node> &input, const std::weak_ptr<Node> &output)
		{
			auto tmp = input.lock();
			bool result1 = std::find_if(tmp->m_outputs.begin(), tmp->m_outputs.end(), [output](const std::weak_ptr<Node> &x)
			{	return x.lock() == output.lock();}) != tmp->m_outputs.end();
			tmp = output.lock();
			bool result2 = std::find_if(tmp->m_inputs.begin(), tmp->m_inputs.end(), [input](const std::weak_ptr<Node> &x)
			{	return x.lock() == input.lock();}) != tmp->m_inputs.end();
			return result1 and result2;
		}
		void Node::createLink(const std::weak_ptr<Node> &input, const std::weak_ptr<Node> &output)
		{
			if (areLinked(input, output))
				throw std::logic_error("createLink() : nodes are already linked");
			input.lock()->m_outputs.push_back(output);
			output.lock()->m_inputs.push_back(input);
			assert(areLinked(input, output));
		}
		void Node::removeLink(const std::weak_ptr<Node> &input, const std::weak_ptr<Node> &output)
		{
			if (not areLinked(input, output))
				throw std::logic_error("removeLink() : nodes are not linked");

			auto tmp = input.lock();
			auto iter1 = std::find_if(tmp->m_outputs.begin(), tmp->m_outputs.end(), [output](const std::weak_ptr<Node> &x)
			{	return x.lock() == output.lock();});
			assert(iter1 != tmp->m_outputs.end());
			tmp->m_outputs.erase(iter1);

			tmp = output.lock();
			auto iter2 = std::find_if(tmp->m_inputs.begin(), tmp->m_inputs.end(), [input](const std::weak_ptr<Node> &x)
			{	return x.lock() == input.lock();});
			assert(iter1 != tmp->m_inputs.end());
			tmp->m_inputs.erase(iter2);
			assert(!areLinked(input, output));
		}

		/*
		 *
		 */

		Shape getShapeAfterBroadcasting(const Shape &lhs, const Shape &rhs)
		{
			std::vector<int> lhs_shape(lhs.data(), lhs.data() + lhs.rank());
			std::vector<int> rhs_shape(rhs.data(), rhs.data() + rhs.rank());

			std::vector<int> &tmp = (lhs.rank() > rhs.rank()) ? rhs_shape : lhs_shape;
			for (int i = 0; i < std::abs(lhs.rank() - rhs.rank()); i++)
				tmp.insert(tmp.begin(), 1);

			std::vector<int> result;
			for (size_t i = 0; i < lhs_shape.size(); i++)
				if (lhs_shape[i] != rhs_shape[i] and lhs_shape[i] != 1 and rhs_shape[i] != 1)
					throw std::logic_error("shapes are not broadcastable");
				else
					result.push_back(std::max(lhs_shape[i], rhs_shape[i]));
			return Shape(result);
		}

		void Elementwise::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = getInput(0).getOutputShape();
		}

		void Broadcastable::calculateOutputShape()
		{
			if (numberOfInputs() != 2)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly two inputs");
			m_output_shape = getShapeAfterBroadcasting(getInput(0).getOutputShape(), getInput(1).getOutputShape());
		}

		Reduction::Reduction(std::vector<int> axes) :
				m_axes(axes)
		{
		}
		void Reduction::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			if (m_axes.empty())
			{
				m_axes = std::vector<int>(getInput(0).getOutputShape().rank(), 0);
				std::iota(m_axes.begin(), m_axes.end(), 0);
			}
			const Shape input_shape = getInput(0).getOutputShape();
			m_output_shape = Shape( { input_shape.volume() / input_shape.volumeOverDims(m_axes) });
		}

	} /* namespace nodes */
} /* namespace avocado */

