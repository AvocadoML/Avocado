/*
 * activations.hpp
 *
 *  Created on: Aug 3, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_ACTIVATIONS_HPP_
#define AVOCADO_EXPRESSION_NODES_ACTIVATIONS_HPP_

#include <Avocado/expression/nodes/Node.hpp>

namespace avocado
{
	class Expression;

	namespace nodes
	{
		class Sigmoid: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

		class ReLU: public Elementwise
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

	} /* namespace nodes */
} /* namespace avocado */



#endif /* AVOCADO_EXPRESSION_NODES_ACTIVATIONS_HPP_ */
