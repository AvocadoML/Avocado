/*
 * bitwise.hpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_BITWISE_HPP_
#define AVOCADO_EXPRESSION_NODES_BITWISE_HPP_

#include <Avocado/expression/nodes/Node.hpp>
#include <Avocado/expression/node_reference.hpp>

namespace avocado
{
	class Expression;

	namespace nodes
	{
		class BitwiseNot: public Elementwise
		{
			public:
				std::string toString() const;
		};
		class BitwiseOr: public Broadcastable
		{
			public:
				std::string toString() const;
		};
		class BitwiseAnd: public Broadcastable
		{
			public:
				std::string toString() const;
		};
		class BitwiseXor: public Broadcastable
		{
			public:
				std::string toString() const;
		};
		class BitwiseShiftLeft: public Broadcastable
		{
			public:
				std::string toString() const;
		};
		class BitwiseShiftRight: public Broadcastable
		{
			public:
				std::string toString() const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_BITWISE_HPP_ */
