/*
 * arithmetic.hpp
 *
 *  Created on: Jul 28, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_ARITHMETIC_HPP_
#define AVOCADO_EXPRESSION_NODES_ARITHMETIC_HPP_

#include <Avocado/expression/nodes/Node.hpp>
#include <Avocado/expression/node_reference.hpp>

namespace avocado
{
	class Expression;

	namespace nodes
	{
		class Negation: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		class Addition: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		class Subtraction: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		class Multiplication: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		class Division: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		class Modulo: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_ARITHMETIC_HPP_ */
