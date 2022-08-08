/*
 * Expression.hpp
 *
 *  Created on: Jul 27, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_EXPRESSION_HPP_
#define AVOCADO_EXPRESSION_EXPRESSION_HPP_

#include <Avocado/expression/node_reference.hpp>
#include <Avocado/expression/NodeType.hpp>

#include <vector>
#include <stdexcept>
#include <cstddef>

namespace avocado
{
	class Expression;
	class Shape;
	namespace nodes
	{
		class Node;
	}

	class Expression
	{
			friend class node_reference;
		private:
			std::vector<std::shared_ptr<nodes::Node>> m_list_of_nodes;

			std::vector<std::weak_ptr<nodes::Node>> m_inputs;
			std::vector<std::weak_ptr<nodes::Node>> m_outputs;
			std::vector<std::weak_ptr<nodes::Node>> m_targets;
			std::vector<std::weak_ptr<nodes::Node>> m_losses;
			std::vector<std::weak_ptr<nodes::Node>> m_metrics;

			node_reference add_node(std::shared_ptr<nodes::Node> newNode, std::initializer_list<node_reference> inputs);
		public:
			Expression clone() const;
			void sort();
			void invert();
			Expression getBackward() const;

			std::string toString() const;
			std::string toString2() const;

			node_reference input(const Shape &shape);
			/*
			 * \brief Marks the node as output node of the expression. Returns node to the learning target (can be ignored).
			 */
			node_reference output(const node_reference &x);
			/*
			 * \brief Marks this node as the loss function output that will be optimized during training.
			 */
			void loss(const node_reference &x);
			/*
			 * \brief Marks this node as the metric function that will be calculated during training but it's not optimized.
			 */
			void metric(const node_reference &x);
			node_reference view(const node_reference &x);
			node_reference view(const nodes::Node *x);
			node_reference view(const std::weak_ptr<nodes::Node> &x);

			node_reference identity(const node_reference &a);
			node_reference one();
			node_reference zero();
			node_reference constant(double value);
			node_reference constant(std::initializer_list<double> values);
			/*
			 * Arithmetic operators.
			 */
			node_reference neg(const node_reference &a);
			node_reference add(const node_reference &a, const node_reference &b);
			node_reference sub(const node_reference &a, const node_reference &b);
			node_reference mul(const node_reference &a, const node_reference &b);
			node_reference div(const node_reference &a, const node_reference &b);
			node_reference mod(const node_reference &a, const node_reference &b);
			/*
			 * Bitwise operators.
			 */
			node_reference bitwise_not(const node_reference &a);
			node_reference bitwise_and(const node_reference &a, const node_reference &b);
			node_reference bitwise_or(const node_reference &a, const node_reference &b);
			node_reference bitwise_xor(const node_reference &a, const node_reference &b);
			node_reference bitwise_shift_left(const node_reference &a, const node_reference &b);
			node_reference bitwise_shift_right(const node_reference &a, const node_reference &b);
			/*
			 * Comparison operators.
			 */
			node_reference equal(const node_reference &a, const node_reference &b);
			node_reference not_equal(const node_reference &a, const node_reference &b);
			node_reference lower_than(const node_reference &a, const node_reference &b);
			node_reference greater_than(const node_reference &a, const node_reference &b);
			node_reference lower_or_equal(const node_reference &a, const node_reference &b);
			node_reference greater_or_equal(const node_reference &a, const node_reference &b);
			/*
			 * Arithmetic functions
			 */
			node_reference abs(const node_reference &a);
			node_reference sign(const node_reference &a);
			node_reference floor(const node_reference &a);
			node_reference ceil(const node_reference &a);
			node_reference square(const node_reference &a);
			node_reference cube(const node_reference &a);
			node_reference pow(const node_reference &a, const node_reference &b);
			node_reference sqrt(const node_reference &a);
			node_reference cbrt(const node_reference &a);
			node_reference sin(const node_reference &a);
			node_reference cos(const node_reference &a);
			node_reference tan(const node_reference &a);
			node_reference sinh(const node_reference &a);
			node_reference cosh(const node_reference &a);
			node_reference tanh(const node_reference &a);
			node_reference exp(const node_reference &a);
			node_reference exp2(const node_reference &a);
			node_reference log(const node_reference &a);
			node_reference log10(const node_reference &a);
			node_reference log2(const node_reference &a);
			node_reference min(const node_reference &a, const node_reference &b);
			node_reference max(const node_reference &a, const node_reference &b);
			/*
			 * Special functions
			 */
			node_reference select(const node_reference &a, const node_reference &b, const node_reference &c);
			/*
			 * Activation functions
			 */
			node_reference sigmoid(const node_reference &a);
			node_reference relu(const node_reference &a);
			/*
			 * Reduction
			 */
			node_reference reduce_add(const node_reference &a, std::initializer_list<int> axes = { });
			node_reference reduce_mul(const node_reference &a, std::initializer_list<int> axes = { });
			node_reference reduce_min(const node_reference &a, std::initializer_list<int> axes = { });
			node_reference reduce_max(const node_reference &a, std::initializer_list<int> axes = { });
			node_reference reduce_and(const node_reference &a, std::initializer_list<int> axes = { });
			node_reference reduce_or(const node_reference &a, std::initializer_list<int> axes = { });
			/*
			 * Compound
			 */
			node_reference transpose(const node_reference &a, const std::vector<int> &order);
			node_reference matmul(const node_reference &a, const node_reference &b, char opA, char opB);
			node_reference conv(const node_reference &a, const node_reference &b, std::initializer_list<int> filterShape);
	};

	class ExpressionTopologyError: public std::logic_error
	{
		public:
			ExpressionTopologyError(const char *function, const std::string &comment);
	};

} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_EXPRESSION_HPP_ */
