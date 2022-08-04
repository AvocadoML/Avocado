/*
 * compound.cpp
 *
 *  Created on: Aug 3, 2022
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/expression/nodes/compound.hpp>
#include <Avocado/expression/Expression.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/error_handling.hpp>

namespace
{
	char transpose(char c) noexcept
	{
		switch (c)
		{
			case 'n':
				return 't';
			case 't':
				return 'n';
			default:
				return c;
		}
	}
}

namespace avocado
{
	namespace nodes
	{
		Transpose::Transpose(const std::vector<int> &order) :
				m_order(order)
		{
		}
		void Transpose::calculateOutputShape()
		{
			if (numberOfInputs() != 1)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly one input");
			m_output_shape = Shape(getInput(0).getOutputShape());
			for (size_t i = 0; i < m_order.size(); i++)
				m_output_shape[i] = getInput(0).getOutputShape()[m_order[i]];
		}
		std::string Transpose::toString() const
		{
			return this->text() + " = transpose(" + getInput(0).text() + "), new axis order = "; // TODO finish this
		}
		Expression Transpose::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			result.output(result.transpose(dy, m_order));
			return result;
		}

		MatrixMultiplication::MatrixMultiplication(char opA, char opB) :
				m_opA(opA),
				m_opB(opB)
		{
		}
		void MatrixMultiplication::calculateOutputShape()
		{
			if (numberOfInputs() != 2)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly two inputs");
			const Shape lhs = getInput(0).getOutputShape();
			const Shape rhs = getInput(1).getOutputShape();
			if (m_opA == 'n')
			{
				if (m_opB == 'n')
					m_output_shape = Shape( { lhs[0], rhs[1] });
				else
					m_output_shape = Shape( { lhs[0], rhs[0] });
			}
			else
			{
				if (m_opB == 'n')
					m_output_shape = Shape( { lhs[1], rhs[1] });
				else
					m_output_shape = Shape( { lhs[1], rhs[0] });
			}
		}
		std::string MatrixMultiplication::toString() const
		{
			return this->text() + " = matmul_" + m_opA + m_opB + "(" + getInput(0).text() + ", " + getInput(1).text() + ")";
		}
		Expression MatrixMultiplication::getBackprop() const
		{
			Expression result;
			auto dy = result.input(this->getOutputShape());
			auto x1 = result.view(m_inputs.at(0));
			auto x2 = result.view(m_inputs.at(1));
			result.output(result.matmul(dy, x1, m_opA, transpose(m_opB)));
			result.output(result.matmul(dy, x2, transpose(m_opA), transpose(m_opB)));
			return result;
		}

		Convolution::Convolution(const Shape &filterShape)
		{
		}
		void Convolution::calculateOutputShape()
		{
			if (numberOfInputs() != 2)
				throw ExpressionTopologyError(METHOD_NAME, "node must have exactly two inputs");
//					const Shape lhs = getInput(0).getOutputShape();
//					const Shape rhs = getInput(1).getOutputShape();
//					if (m_opA == 'n')
//					{
//						if (m_opB == 'n')
//							m_output_shape = Shape( { lhs[0], rhs[1] });
//						else
//							m_output_shape = Shape( { lhs[0], rhs[0] });
//					}
//					else
//					{
//						if (m_opB == 'n')
//							m_output_shape = Shape( { lhs[1], rhs[1] });
//						else
//							m_output_shape = Shape( { lhs[1], rhs[0] });
//					}
		}
		std::string Convolution::toString() const
		{
			return this->text() + " = convolution()";
		}
		Expression Convolution::getBackprop() const
		{
			Expression result;
//					auto dy = result.input(this->getOutputShape());
//					auto x1 = result.view(m_inputs.at(0));
//					auto x2 = result.view(m_inputs.at(1));
//					result.output(result.matmul(dy, x1, m_opA, transpose(m_opB)));
//					result.output(result.matmul(dy, x2, transpose(m_opA), transpose(m_opB)));
			return result;
		}

	} /* namespace nodes */
} /* namespace avocado */

