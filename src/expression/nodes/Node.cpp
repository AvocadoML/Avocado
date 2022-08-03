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
#include <numeric>

namespace avocado
{
	namespace nodes
	{
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

		std::string Node::text() const
		{
			return "x" + std::to_string(m_index) + ":" + getOutputShape().toString();
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

		void Node::createLink(const std::weak_ptr<Node> &input, const std::weak_ptr<Node> &output)
		{
			input.lock()->m_outputs.push_back(output);
			output.lock()->m_inputs.push_back(input);
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

		Reduction::Reduction(std::initializer_list<int> axes) :
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

