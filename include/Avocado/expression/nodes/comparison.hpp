/*
 * logical.hpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_COMPARISON_HPP_
#define AVOCADO_EXPRESSION_NODES_COMPARISON_HPP_

#include <Avocado/expression/nodes/Node.hpp>
#include <Avocado/expression/node_reference.hpp>

namespace avocado
{
	class Expression;

	namespace nodes
	{
		class Equal: public Broadcastable
		{
			public:
				Equal* clone() const;
				std::string toString() const;
		};
		class NotEqual: public Broadcastable
		{
			public:
				NotEqual* clone() const;
				std::string toString() const;
		};
		class LowerThan: public Broadcastable
		{
			public:
				LowerThan* clone() const;
				std::string toString() const;
		};
		class LowerOrEqual: public Broadcastable
		{
			public:
				LowerOrEqual* clone() const;
				std::string toString() const;
		};
		class GreaterThan: public Broadcastable
		{
			public:
				GreaterThan* clone() const;
				std::string toString() const;
		};
		class GreaterOrEqual: public Broadcastable
		{
			public:
				GreaterOrEqual* clone() const;
				std::string toString() const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_COMPARISON_HPP_ */
