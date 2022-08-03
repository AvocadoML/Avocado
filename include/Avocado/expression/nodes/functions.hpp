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
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Sign: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		class Floor: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Ceil: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Power
		 */
		class Square: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Cube: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Power: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class SquareRoot: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class CubeRoot: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Trigonometrical
		 */
		class Sine: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Cosine: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Tangent: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Hyperbolical
		 */
		class HyperbolicalSine: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class HyperbolicalCosine: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class HyperbolicalTangent: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Exponential
		 */
		class Exponential: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Exponential2: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Logarithmic
		 */
		class LogarithmNatural: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class LogarithmBase10: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class LogarithmBase2: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		/*
		 * Min/Max
		 */
		class Minimum: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};
		class Maximum: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_FUNCTIONS_HPP_ */
