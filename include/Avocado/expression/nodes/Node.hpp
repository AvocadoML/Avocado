/*
 * Node.hpp
 *
 *  Created on: Jul 28, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_NODE_HPP_
#define AVOCADO_EXPRESSION_NODES_NODE_HPP_

#include <Avocado/core/Shape.hpp>

#include <vector>
#include <string>
#include <memory>

namespace avocado
{
	class Expression;

	namespace nodes
	{
		class Node
		{
			protected:
				Expression *m_expression = nullptr;
				size_t m_index = 0;
				std::vector<std::weak_ptr<Node>> m_inputs;
				std::vector<std::weak_ptr<Node>> m_outputs;
				Shape m_output_shape;

			public:
				Node() = default;
				Node(const Node &other) = delete;
				Node(Node &&other) = default;
				Node& operator=(const Node &other) = delete;
				Node& operator=(Node &&other) = default;
				virtual ~Node() = default;

				virtual void calculateOutputShape();
				const Shape& getOutputShape() const noexcept;

				Expression& getExpression() const;
				void setExpression(Expression &e) noexcept;
				void setIndex(size_t i) noexcept;

				size_t numberOfInputs() const noexcept;
				Node& getInput(size_t index);
				const Node& getInput(size_t index) const;

				size_t numberOfOutputs() const noexcept;
				Node& getOutput(size_t index);
				const Node& getOutput(size_t index) const;

				std::string text() const;
				virtual std::string toString() const = 0;
				virtual Expression getBackprop() const;

				static void createLink(const std::weak_ptr<Node> &input, const std::weak_ptr<Node> &output);
		};

		Shape getShapeAfterBroadcasting(const Shape &lhs, const Shape &rhs);

		class Elementwise: public Node
		{
			public:
				void calculateOutputShape();
		};

		class Broadcastable: public Node
		{
			public:
				void calculateOutputShape();
		};

		class Reduction: public Node
		{
			protected:
				std::vector<int> m_axes;
			public:
				Reduction(std::initializer_list<int> axes);
				void calculateOutputShape();
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_NODE_HPP_ */
