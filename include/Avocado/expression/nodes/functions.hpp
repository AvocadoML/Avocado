/*
 * functions.hpp
 *
 *  Created on: Jul 30, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_FUNCTIONS_HPP_
#define AVOCADO_EXPRESSION_NODES_FUNCTIONS_HPP_

#include <Avocado/expression/nodes/Node.hpp>
#include <Avocado/expression/node_reference.hpp>

namespace avocado
{
	class Expression;

	namespace nodes
	{
		class AbsoluteValue: public Elementwise
		{
			public:
				AbsoluteValue* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Sign: public Elementwise
		{
			public:
				Sign* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};

		class Floor: public Elementwise
		{
			public:
				Floor* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Ceil: public Elementwise
		{
			public:
				Ceil* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Power
		 */
		class Square: public Elementwise
		{
			public:
				Square* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Cube: public Elementwise
		{
			public:
				Cube* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Power: public Broadcastable
		{
			public:
				Power* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class SquareRoot: public Elementwise
		{
			public:
				SquareRoot* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class CubeRoot: public Elementwise
		{
			public:
				CubeRoot* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Trigonometrical
		 */
		class Sine: public Elementwise
		{
			public:
				Sine* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Cosine: public Elementwise
		{
			public:
				Cosine* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Tangent: public Elementwise
		{
			public:
				Tangent* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Hyperbolical
		 */
		class HyperbolicalSine: public Elementwise
		{
			public:
				HyperbolicalSine* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class HyperbolicalCosine: public Elementwise
		{
			public:
				HyperbolicalCosine* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class HyperbolicalTangent: public Elementwise
		{
			public:
				HyperbolicalTangent* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Exponential
		 */
		class Exponential: public Elementwise
		{
			public:
				Exponential* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Exponential2: public Elementwise
		{
			public:
				Exponential2* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Logarithmic
		 */
		class LogarithmNatural: public Elementwise
		{
			public:
				LogarithmNatural* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class LogarithmBase10: public Elementwise
		{
			public:
				LogarithmBase10* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class LogarithmBase2: public Elementwise
		{
			public:
				LogarithmBase2* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Min/Max
		 */
		class Minimum: public Broadcastable
		{
			public:
				Minimum* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Maximum: public Broadcastable
		{
			public:
				Maximum* clone() const;
				std::string toString() const;
				Expression getBackprop() const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_FUNCTIONS_HPP_ */
