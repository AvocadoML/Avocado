/*
 * node_reference.cpp
 *
 *  Created on: Jul 27, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/node_reference.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/expression/nodes/Node.hpp>

#include <cassert>

namespace avocado
{

	node_reference::node_reference(const std::weak_ptr<nodes::Node> &node) :
			m_node(node)
	{
	}
	std::weak_ptr<nodes::Node>& node_reference::getNode()
	{
		return m_node;
	}
	const std::weak_ptr<nodes::Node>& node_reference::getNode() const
	{
		return m_node;
	}
	Expression& node_reference::getExpression() const
	{
		return m_node.lock()->getExpression();
	}

	node_reference operator+(const node_reference &a)
	{
		return a.getExpression().identity(a);
	}
	/*
	 * Arithmetic operators.
	 */
	node_reference operator-(const node_reference &a)
	{
		return a.getExpression().neg(a);
	}
	node_reference operator+(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().add(a, b);
	}
	node_reference operator-(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().sub(a, b);
	}
	node_reference operator*(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().mul(a, b);
	}
	node_reference operator/(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().div(a, b);
	}
	node_reference operator%(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().mod(a, b);
	}
	/*
	 * Bitwise operators.
	 */
	node_reference operator~(const node_reference &a)
	{
		return a.getExpression().bitwise_not(a);
	}
	node_reference operator&(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().bitwise_and(a, b);
	}
	node_reference operator|(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().bitwise_or(a, b);
	}
	node_reference operator^(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().bitwise_xor(a, b);
	}
	node_reference operator<<(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().bitwise_shift_left(a, b);
	}
	node_reference operator>>(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().bitwise_shift_right(a, b);
	}
	/*
	 * Logical operators.
	 */
	node_reference operator!(const node_reference &a)
	{
		return a.getExpression().bitwise_not(a);
	}
	node_reference operator&&(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().bitwise_and(a, b);
	}
	node_reference operator||(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().bitwise_or(a, b);
	}
	/*
	 * Comparison operators.
	 */
	node_reference operator==(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().equal(a, b);
	}
	node_reference operator!=(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().not_equal(a, b);
	}
	node_reference operator<(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().lower_than(a, b);
	}
	node_reference operator>(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().greater_than(a, b);
	}
	node_reference operator<=(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().lower_or_equal(a, b);
	}
	node_reference operator>=(const node_reference &a, const node_reference &b)
	{
		return a.getExpression().greater_or_equal(a, b);
	}

} /* namespace avocado */

