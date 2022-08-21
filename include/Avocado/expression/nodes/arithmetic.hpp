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
				Negation* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		class Addition: public Broadcastable
		{
			public:
				Addition* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		class Subtraction: public Broadcastable
		{
			public:
				Subtraction* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		class Multiplication: public Broadcastable
		{
			public:
				Multiplication* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		class Division: public Broadcastable
		{
			public:
				Division* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		class Modulo: public Broadcastable
		{
			public:
				Modulo* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_ARITHMETIC_HPP_ */
