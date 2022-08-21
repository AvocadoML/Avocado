/*
 * autograd.hpp
 *
 *  Created on: Aug 13, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_AUTOGRAD_HPP_
#define AVOCADO_EXPRESSION_AUTOGRAD_HPP_

#include <Avocado/expression/node_reference.hpp>
#include <Avocado/expression/NodeType.hpp>

#include <vector>
#include <stdexcept>
#include <cstddef>

namespace avocado
{
	class Expression;

	class Autograd
	{

			static void join(Expression &prev, Expression &next);
		public:
			static Expression getBackward(const Expression &e);
	};

} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_AUTOGRAD_HPP_ */
