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
				MeanSquareLoss* clone() const;
				std::string toString() const;
				std::vector<node_reference> getBackprop(Expression &e, const std::vector<node_reference> &gradients) const;
		};

	} /* namespace nodes */
} /* namespace avocado */



#endif /* AVOCADO_EXPRESSION_NODES_LOSSES_HPP_ */
