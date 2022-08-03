/*
 * reduction.hpp
 *
 *  Created on: Aug 1, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_EXPRESSION_NODES_REDUCTION_HPP_
#define AVOCADO_EXPRESSION_NODES_REDUCTION_HPP_

#include <Avocado/expression/nodes/Node.hpp>
#include <Avocado/expression/node_reference.hpp>

namespace avocado
{
	class Expression;

	namespace nodes
	{

		class ReduceAdd: public Reduction
		{
			public:
				ReduceAdd(std::initializer_list<int> axes);
				std::string toString() const;
				Expression getBackprop() const;
		};
		class ReduceMul: public Reduction
		{
			public:
				ReduceMul(std::initializer_list<int> axes);
				std::string toString() const;
				Expression getBackprop() const;
		};
		class ReduceMin: public Reduction
		{
			public:
				ReduceMin(std::initializer_list<int> axes);
				std::string toString() const;
				Expression getBackprop() const;
		};
		class ReduceMax: public Reduction
		{
			public:
				ReduceMax(std::initializer_list<int> axes);
				std::string toString() const;
				Expression getBackprop() const;
		};

		class ReduceAnd: public Reduction
		{
			public:
				ReduceAnd(std::initializer_list<int> axes);
				std::string toString() const;
		};
		class ReduceOr: public Reduction
		{
			public:
				ReduceOr(std::initializer_list<int> axes);
				std::string toString() const;
		};

	} /* namespace nodes */
} /* namespace avocado */

#endif /* AVOCADO_EXPRESSION_NODES_REDUCTION_HPP_ */
