/*
 * node_reference.hpp
 *
 *  Created on: Jul 27, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef EXPRESSION_NODE_REFERENCE_HPP_
#define EXPRESSION_NODE_REFERENCE_HPP_

#include <cstddef>
#include <memory>

namespace avocado
{
	class Expression;
	namespace nodes
	{
		class Node;
	}
}

namespace avocado
{
	class node_reference
	{
			friend class Expression;
			friend class nodes::Node;

			std::weak_ptr<nodes::Node> m_node;

		public:
			node_reference(const std::weak_ptr<nodes::Node> &node);
			std::weak_ptr<nodes::Node>& getNode();
			const std::weak_ptr<nodes::Node>& getNode() const;
			Expression& getExpression() const;

			friend node_reference operator+(const node_reference &a);
			friend node_reference operator-(const node_reference &a);
			/*
			 * Arithmetic operators.
			 */
			friend node_reference operator+(const node_reference &a, const node_reference &b);
			friend node_reference operator-(const node_reference &a, const node_reference &b);
			friend node_reference operator*(const node_reference &a, const node_reference &b);
			friend node_reference operator/(const node_reference &a, const node_reference &b);
			friend node_reference operator%(const node_reference &a, const node_reference &b);
			/*
			 * Bitwise operators.
			 */
			friend node_reference operator~(const node_reference &a);
			friend node_reference operator&(const node_reference &a, const node_reference &b);
			friend node_reference operator|(const node_reference &a, const node_reference &b);
			friend node_reference operator^(const node_reference &a, const node_reference &b);
			friend node_reference operator<<(const node_reference &a, const node_reference &b);
			friend node_reference operator>>(const node_reference &a, const node_reference &b);
			/*
			 * Logical operators.
			 */
			friend node_reference operator!(const node_reference &a);
			friend node_reference operator&&(const node_reference &a, const node_reference &b);
			friend node_reference operator||(const node_reference &a, const node_reference &b);
			/*
			 * Comparison operators.
			 */
			friend node_reference operator==(const node_reference &a, const node_reference &b);
			friend node_reference operator!=(const node_reference &a, const node_reference &b);
			friend node_reference operator<(const node_reference &a, const node_reference &b);
			friend node_reference operator>(const node_reference &a, const node_reference &b);
			friend node_reference operator<=(const node_reference &a, const node_reference &b);
			friend node_reference operator>=(const node_reference &a, const node_reference &b);
	};

} /* namespace avocado */

#endif /* EXPRESSION_NODE_REFERENCE_HPP_ */
