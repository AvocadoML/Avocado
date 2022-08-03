/*
 * NodeType.hpp
 *
 *  Created on: Jul 27, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODETYPE_HPP_
#define AVOCADO_EXPRESSION_NODETYPE_HPP_

namespace avocado
{
	enum class NodeType
	{
		INPUT,
		OUTPUT,
		IDENTITY,
		/*
		 * Arithmetic operators
		 */
		NEGATION,
		ADD,
		SUBTRACT,
		MULTIPLY,
		DIVIDE,
		MODULO,
		/*
		 * Bitwise operators
		 */
		BITWISE_NOT,
		BITWISE_AND,
		BITWISE_OR,
		BITWISE_XOR,
		BITWISE_SHIFT_LEFT,
		BITWISE_SHIFT_RIGHT,
		/*
		 * Comparison operators
		 */
		EQUAL,
		NOT_EQUAL,
		LOWER_THAN,
		GREATER_THAN,
		LOWER_OR_EQUAL,
		GREATER_OR_EQUAL,
		/*
		 * Arithmetic functions
		 */
		ABS,
		SIGN,
		FLOOR,
		CEIL,
		RCP, // reciprocal value
		RSQRT, // reciprocal of a square root
		SQUARE,
		CUBE,
		SQRT, // square root
		CBRT, // cube root
		SIN,
		COS,
		TAN,
		SINH,
		COSH,
		TANH,
		EXP,
		EXP2,
		EXPM1, // exp(x) - 1
		LOG, // natural logarithm
		LOG1P, // ln(1 + x)
		LOG10,
		LOG2,
		POW,
		MIN,
		MAX,
		/*
		 * Special functions
		 */
		SELECT, // a ? b : c
		MUL_ADD, // a * b + c
		MUL_SUB, // a * b - c
		NEG_MUL_ADD, // -(a * b) + c
		NEG_MUL_SUB, // -(a * b) - c
		/*
		 * Activation functions
		 */
		SIGMOID, // 1 / (1 + exp(-a))
		RELU, // max(0, a)
		/*
		 * Reduction
		 */
		REDUCE_ADD,
		REDUCE_MUL,
		REDUCE_MIN,
		REDUCE_MAX,
		REDUCE_AND,
		REDUCE_OR
		/*
		 * Compound functions
		 */

	};
}

#endif /* AVOCADO_EXPRESSION_NODETYPE_HPP_ */
