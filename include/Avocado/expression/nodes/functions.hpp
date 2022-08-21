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
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Sign: public Elementwise
		{
			public:
				Sign* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		class Floor: public Elementwise
		{
			public:
				Floor* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Ceil: public Elementwise
		{
			public:
				Ceil* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		/*
		 * Power
		 */
		class Square: public Elementwise
		{
			public:
				Square* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Cube: public Elementwise
		{
			public:
				Cube* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Power: public Broadcastable
		{
			public:
				Power* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class SquareRoot: public Elementwise
		{
			public:
				SquareRoot* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class CubeRoot: public Elementwise
		{
			public:
				CubeRoot* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		/*
		 * Trigonometrical
		 */
		class Sine: public Elementwise
		{
			public:
				Sine* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Cosine: public Elementwise
		{
			public:
				Cosine* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Tangent: public Elementwise
		{
			public:
				Tangent* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		/*
		 * Hyperbolical
		 */
		class HyperbolicalSine: public Elementwise
		{
			public:
				HyperbolicalSine* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class HyperbolicalCosine: public Elementwise
		{
			public:
				HyperbolicalCosine* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class HyperbolicalTangent: public Elementwise
		{
			public:
				HyperbolicalTangent* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		/*
		 * Exponential
		 */
		class Exponential: public Elementwise
		{
			public:
				Exponential* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Exponential2: public Elementwise
		{
			public:
				Exponential2* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		/*
		 * Logarithmic
		 */
		class LogarithmNatural: public Elementwise
		{
			public:
				LogarithmNatural* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class LogarithmBase10: public Elementwise
		{
			public:
				LogarithmBase10* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class LogarithmBase2: public Elementwise
		{
			public:
				LogarithmBase2* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

		/*
		 * Min/Max
		 */
		class Minimum: public Broadcastable
		{
			public:
				Minimum* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};
		class Maximum: public Broadcastable
		{
			public:
				Maximum* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_FUNCTIONS_HPP_ */
