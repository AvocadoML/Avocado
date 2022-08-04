/*
 * technival.hpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_TECHNICAL_HPP_
#define AVOCADO_EXPRESSION_NODES_TECHNICAL_HPP_

#include <Avocado/expression/nodes/Node.hpp>

namespace avocado
{
	class Shape;
	class Expression;

	namespace nodes
	{
		class Input: public Node
		{
			public:
				Input(const Shape &shape);
				std::string toString() const;
		};
		class Target: public Node
		{
			public:
				Target(const Shape &shape);
				std::string toString() const;
		};

		class Output: public Node
		{
			public:
				void calculateOutputShape();
				std::string toString() const;
		};
		class Loss: public Node
		{
			public:
				void calculateOutputShape();
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Metric: public Node
		{
			public:
				void calculateOutputShape();
				std::string toString() const;
		};

		class View: public Node
		{
			public:
				void calculateOutputShape();
				std::string toString() const;
		};

		class Identity: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Constant values
		 */
		class One: public Node
		{
			public:
				void calculateOutputShape();
				std::string toString() const;
		};
		class Zero: public Node
		{
			public:
				void calculateOutputShape();
				std::string toString() const;
		};
		class Constant: public Node
		{
				std::vector<double> m_values;
			public:
				Constant(double value);
				Constant(const std::vector<double> &values);
				void calculateOutputShape();
				std::string toString() const;
		};

		/*
		 * Selection
		 */
		class Select: public Node
		{
			public:
				void calculateOutputShape();
				std::string toString() const;
				Expression getBackprop() const;
		};
	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_TECHNICAL_HPP_ */
