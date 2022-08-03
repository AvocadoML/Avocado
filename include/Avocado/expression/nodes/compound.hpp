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
		class MatrixMultiplication: public Node
		{
				char m_opA, m_opB;
			public:
				MatrixMultiplication(char opA, char opB);
				void calculateOutputShape();
				std::string toString() const;
				Expression getBackprop() const;
		};

		class Convolution: public Node
		{
			public:
				Convolution(const Shape &filterShape);
				void calculateOutputShape();
				std::string toString() const;
				Expression getBackprop() const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_COMPOUND_HPP_ */
