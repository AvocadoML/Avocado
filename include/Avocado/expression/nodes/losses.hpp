/*
 * losses.hpp
 *
 *  Created on: Jul 31, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_LOSSES_HPP_
#define AVOCADO_EXPRESSION_NODES_LOSSES_HPP_

#include <Avocado/expression/nodes/Node.hpp>
#include <Avocado/expression/node_reference.hpp>

namespace avocado
{
	class Expression;

	namespace nodes
	{
		class MeanSquareLoss: public Broadcastable
		{
			public:
				std::string toString() const;
				Expression getBackprop() const;
		};

	} /* namespace nodes */
} /* namespace avocado */



#endif /* AVOCADO_EXPRESSION_NODES_LOSSES_HPP_ */
