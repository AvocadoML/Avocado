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
				Input* clone() const;
				std::string toString() const;
		};
		class Target: public Node
		{
			public:
				Target(const Shape &shape);
				Target* clone() const;
				std::string toString() const;
		};

		class Output: public Node
		{
			public:
				Output* clone() const;
				void calculateOutputShape();
				std::string toString() const;
		};
		class Loss: public Node
		{
			public:
				Loss* clone() const;
				void calculateOutputShape();
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Metric: public Node
		{
			public:
				Metric* clone() const;
				void calculateOutputShape();
				std::string toString() const;
		};

		class View: public Node
		{
			public:
				View* clone() const;
				void calculateOutputShape();
				std::string toString() const;
		};

		class Identity: public Elementwise
		{
			public:
				Identity* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		/*
		 * Constant values
		 */
		class Constant: public Node
		{
				std::vector<double> m_values;
			public:
				Constant(double value);
				Constant(const std::vector<double> &values);
				Constant* clone() const;
				void calculateOutputShape();
				std::string toString() const;
		};
		class One: public Constant
		{
			public:
				One();
				One* clone() const;
				std::string toString() const;
		};
		class Zero: public Constant
		{
			public:
				Zero();
				Zero* clone() const;
				std::string toString() const;
		};

		/*
		 * Selection
		 */
		class Select: public Node
		{
			public:
				Select* clone() const;
				void calculateOutputShape();
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_TECHNICAL_HPP_ */
