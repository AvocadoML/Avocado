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
	class node_reference;

	namespace nodes
	{
		class Node
		{
			protected:
				Expression *m_expression = nullptr;
				std::vector<std::weak_ptr<Node>> m_inputs;
				std::vector<std::weak_ptr<Node>> m_outputs;
				Shape m_output_shape;
				std::string m_name;

				static node_reference add_gradients(const std::vector<node_reference> &gradients);
			public:
				Node() = default;
				Node(const Node &other) = delete;
				Node(Node &&other) = default;
				Node& operator=(const Node &other) = delete;
				Node& operator=(Node &&other) = default;
				virtual ~Node() = default;

				virtual Node* clone() const;
				virtual void calculateOutputShape();
				const Shape& getOutputShape() const noexcept;

				Expression& getExpression() const;
				void setExpression(Expression &e) noexcept;

				size_t numberOfInputs() const noexcept;
				Node& getInput(size_t index);
				const Node& getInput(size_t index) const;
				std::weak_ptr<Node> getInputNodePointer(size_t index) const;

				size_t numberOfOutputs() const noexcept;
				Node& getOutput(size_t index);
				const Node& getOutput(size_t index) const;
				std::weak_ptr<Node> getOutputNodePointer(size_t index) const;

				std::weak_ptr<Node> getPointer() const;

				std::string text() const;
				virtual std::string toString() const;

				virtual std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;

				static bool areLinked(const Node &input, const Node &output);
				static void replaceInputLink(Node &oldInput, Node &newInput, Node &output);
				static void replaceOutputLink(Node &input, Node &oldOutput, Node &newOutput);
				static void createLink(Node &input, Node &output);
				static void removeLink(Node &input, Node &output);

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
				Reduction(std::vector<int> axes);
				void calculateOutputShape();
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_NODE_HPP_ */
