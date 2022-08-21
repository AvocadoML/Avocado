/*
 * compound.hpp
 *
 *  Created on: Aug 3, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_COMPOUND_HPP_
#define AVOCADO_EXPRESSION_NODES_COMPOUND_HPP_

#include <Avocado/expression/nodes/Node.hpp>

namespace avocado
{
	class Shape;
	class Expression;

	namespace nodes
	{
		class Transpose: public Node
		{
				std::vector<int> m_order;
			public:
				Transpose(const std::vector<int> &order);
				Transpose* clone() const;
				void calculateOutputShape();
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		class MatrixMultiplication: public Node
		{
				char m_opA, m_opB;
			public:
				MatrixMultiplication(char opA, char opB);
				MatrixMultiplication* clone() const;
				void calculateOutputShape();
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		class Convolution: public Node
		{
			public:
				Convolution(const Shape &filterShape);
				Convolution* clone() const;
				void calculateOutputShape();
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_COMPOUND_HPP_ */
