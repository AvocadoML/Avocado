/*
 * Expression.cpp
 *
 *  Created on: Jul 27, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef EXPRESSION_EXPRESSION_CPP_
#define EXPRESSION_EXPRESSION_CPP_

#include <Avocado/expression/Expression.hpp>
#include <Avocado/expression/node_reference.hpp>

#include <Avocado/expression/nodes/activations.hpp>
#include <Avocado/expression/nodes/arithmetic.hpp>
#include <Avocado/expression/nodes/bitwise.hpp>
#include <Avocado/expression/nodes/comparison.hpp>
#include <Avocado/expression/nodes/compound.hpp>
#include <Avocado/expression/nodes/technical.hpp>
#include <Avocado/expression/nodes/reduction.hpp>
#include <Avocado/expression/nodes/functions.hpp>

#include <cassert>
#include <algorithm>
#include <stdexcept>

namespace avocado
{
	using namespace nodes;
	/*
	 * private
	 */
	node_reference Expression::add_node(std::shared_ptr<Node> newNode, std::initializer_list<node_reference> inputs)
	{
		newNode->setIndex(m_list_of_nodes.size());
		newNode->setExpression(*this);
		m_list_of_nodes.push_back(newNode);
		for (auto iter = inputs.begin(); iter < inputs.end(); iter++)
			Node::createLink(iter->getNode(), newNode);
		newNode->calculateOutputShape();
		return node_reference(newNode);
	}
	/*
	 * public
	 */
	std::string Expression::toString() const
	{
		std::string result;
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
			result += m_list_of_nodes[i]->toString() + '\n';
		return result;
	}
	std::string Expression::toString2() const
	{
		std::string result;
		for (size_t i = 0; i < m_list_of_nodes.size(); i++)
		{
			result += m_list_of_nodes[i]->toString() + ":\n";
			result += m_list_of_nodes[i]->getBackprop().toString() + '\n';
		}
		return result;
	}

	node_reference Expression::input(const Shape &shape)
	{
		return add_node(std::make_shared<Input>(shape), { });
	}
	node_reference Expression::output(const node_reference &a)
	{
		node_reference tmp = add_node(std::make_shared<Output>(), { a });
		return add_node(std::make_shared<Target>(tmp.getNode().lock()->getOutputShape()), { });
	}
	node_reference Expression::view(const node_reference &a)
	{
		return add_node(std::make_shared<View>(), { a });
	}
	node_reference Expression::view(const Node *x)
	{
		for (auto iter = m_list_of_nodes.begin(); iter < m_list_of_nodes.end(); iter++)
			if (iter->get() == x)
				return add_node(std::make_shared<View>(), { node_reference(*iter) });
		throw std::logic_error("Expression::view() : node is not a part of this expression");
	}
	node_reference Expression::view(const std::weak_ptr<nodes::Node> &x)
	{
		return add_node(std::make_shared<View>(), { node_reference(x) });
	}
	node_reference Expression::identity(const node_reference &a)
	{
		return add_node(std::make_shared<Identity>(), { a });
	}
	node_reference Expression::one()
	{
		return add_node(std::make_shared<One>(), { });
	}
	node_reference Expression::zero()
	{
		return add_node(std::make_shared<Zero>(), { });
	}
	node_reference Expression::constant(double value)
	{
		return add_node(std::make_shared<Constant>(value), { });
	}
	/*
	 * Arithmetic operators.
	 */
	node_reference Expression::neg(const node_reference &a)
	{
		return add_node(std::make_shared<Negation>(), { a });
	}
	node_reference Expression::add(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Addition>(), { a, b });
	}
	node_reference Expression::sub(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Subtraction>(), { a, b });
	}
	node_reference Expression::mul(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Multiplication>(), { a, b });
	}
	node_reference Expression::div(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Division>(), { a, b });
	}
	node_reference Expression::mod(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Modulo>(), { a, b });
	}
//	/*
//	 * Bitwise operators
//	 */
	node_reference Expression::bitwise_not(const node_reference &a)
	{
		return add_node(std::make_shared<BitwiseNot>(), { a });
	}
	node_reference Expression::bitwise_and(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseAnd>(), { a, b });
	}
	node_reference Expression::bitwise_or(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseOr>(), { a, b });
	}
	node_reference Expression::bitwise_xor(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseXor>(), { a, b });
	}
	node_reference Expression::bitwise_shift_left(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseShiftLeft>(), { a, b });
	}
	node_reference Expression::bitwise_shift_right(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<BitwiseShiftRight>(), { a, b });
	}
//	/*
//	 * Comparison operators.
//	 */
	node_reference Expression::equal(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Equal>(), { a, b });
	}
	node_reference Expression::not_equal(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<NotEqual>(), { a, b });
	}
	node_reference Expression::lower_than(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<LowerThan>(), { a, b });
	}
	node_reference Expression::greater_than(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<GreaterThan>(), { a, b });
	}
	node_reference Expression::lower_or_equal(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<LowerOrEqual>(), { a, b });
	}
	node_reference Expression::greater_or_equal(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<GreaterOrEqual>(), { a, b });
	}
//	/*
//	 * Arithmetic functions
//	 */
	node_reference Expression::abs(const node_reference &a)
	{
		return add_node(std::make_shared<AbsoluteValue>(), { a });
	}
	node_reference Expression::sign(const node_reference &a)
	{
		return add_node(std::make_shared<Sign>(), { a });
	}
	node_reference Expression::floor(const node_reference &a)
	{
		return add_node(std::make_shared<Floor>(), { a });
	}
	node_reference Expression::ceil(const node_reference &a)
	{
		return add_node(std::make_shared<Ceil>(), { a });
	}
	node_reference Expression::square(const node_reference &a)
	{
		return add_node(std::make_shared<Square>(), { a });
	}
	node_reference Expression::cube(const node_reference &a)
	{
		return add_node(std::make_shared<Cube>(), { a });
	}
	node_reference Expression::pow(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Power>(), { a, b });
	}
	node_reference Expression::sqrt(const node_reference &a)
	{
		return add_node(std::make_shared<SquareRoot>(), { a });
	}
	node_reference Expression::cbrt(const node_reference &a)
	{
		return add_node(std::make_shared<CubeRoot>(), { a });
	}
	node_reference Expression::sin(const node_reference &a)
	{
		return add_node(std::make_shared<Sine>(), { a });
	}
	node_reference Expression::cos(const node_reference &a)
	{
		return add_node(std::make_shared<Cosine>(), { a });
	}
	node_reference Expression::tan(const node_reference &a)
	{
		return add_node(std::make_shared<Tangent>(), { a });
	}
	node_reference Expression::sinh(const node_reference &a)
	{
		return add_node(std::make_shared<HyperbolicalSine>(), { a });
	}
	node_reference Expression::cosh(const node_reference &a)
	{
		return add_node(std::make_shared<HyperbolicalCosine>(), { a });
	}
	node_reference Expression::tanh(const node_reference &a)
	{
		return add_node(std::make_shared<HyperbolicalTangent>(), { a });
	}
	node_reference Expression::exp(const node_reference &a)
	{
		return add_node(std::make_shared<Exponential>(), { a });
	}
	node_reference Expression::exp2(const node_reference &a)
	{
		return add_node(std::make_shared<Exponential2>(), { a });
	}
	node_reference Expression::log(const node_reference &a)
	{
		return add_node(std::make_shared<LogarithmNatural>(), { a });
	}
	node_reference Expression::log10(const node_reference &a)
	{
		return add_node(std::make_shared<LogarithmBase10>(), { a });
	}
	node_reference Expression::log2(const node_reference &a)
	{
		return add_node(std::make_shared<LogarithmBase2>(), { a });
	}

	node_reference Expression::min(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Minimum>(), { a, b });
	}
	node_reference Expression::max(const node_reference &a, const node_reference &b)
	{
		return add_node(std::make_shared<Maximum>(), { a, b });
	}
//	/*
//	 * Special functions
//	 */
	node_reference Expression::select(const node_reference &a, const node_reference &b, const node_reference &c)
	{
		return add_node(std::make_shared<Select>(), { a, b, c });
	}
//	/*
//	 * Activation functions
//	 */
	node_reference Expression::sigmoid(const node_reference &a)
	{
		return add_node(std::make_shared<Sigmoid>(), { a });
	}
	node_reference Expression::relu(const node_reference &a)
	{
		return add_node(std::make_shared<ReLU>(), { a });
	}
	/*
	 * Reduction
	 */
	node_reference Expression::reduce_add(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceAdd>(axes), { a });
	}
	node_reference Expression::reduce_mul(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceMul>(axes), { a });
	}
	node_reference Expression::reduce_min(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceMin>(axes), { a });
	}
	node_reference Expression::reduce_max(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceMax>(axes), { a });
	}
	node_reference Expression::reduce_and(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceAnd>(axes), { a });
	}
	node_reference Expression::reduce_or(const node_reference &a, std::initializer_list<int> axes)
	{
		return add_node(std::make_shared<ReduceOr>(axes), { a });
	}
	/*
	 * Compound
	 */
	node_reference Expression::matmul(const node_reference &a, const node_reference &b, char opA, char opB)
	{
		return add_node(std::make_shared<MatrixMultiplication>(opA, opB), { a, b });
	}
	node_reference Expression::conv(const node_reference &a, const node_reference &b, std::initializer_list<int> filterShape)
	{
		return add_node(std::make_shared<Convolution>(filterShape), { a, b });
	}

	ExpressionTopologyError::ExpressionTopologyError(const char *function, const std::string &comment) :
			std::logic_error(std::string(function) + " : " + comment)
	{
	}

} /* namespace avocado */

#endif /* EXPRESSION_EXPRESSION_CPP_ */
