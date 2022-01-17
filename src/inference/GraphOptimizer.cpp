/*
 * GraphOptimizer.cpp
 *
 *  Created on: Feb 28, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/inference/GraphOptimizer.hpp>
#include <Avocado/core/error_handling.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

namespace
{
	using namespace avocado::inference;
	std::vector<std::unique_ptr<GraphOptimizer>>& registered_graph_optimizers()
	{
		static std::vector<std::unique_ptr<GraphOptimizer>> result;
		return result;
	}
}

namespace avocado
{
	namespace inference
	{
		void registerGraphOptimizer(const GraphOptimizer &graphOptimizer)
		{
			auto tmp = std::find_if(registered_graph_optimizers().begin(), registered_graph_optimizers().end(),
					[&](std::unique_ptr<GraphOptimizer> &gc)
					{	return gc->name() == graphOptimizer.name();});

			if (tmp == registered_graph_optimizers().end())
				registered_graph_optimizers().push_back(graphOptimizer.clone());
			else
				throw LogicError(METHOD_NAME, "graph optimizer '" + graphOptimizer.name() + "' has already been registered");
		}
	} /* namespace inference */
} /* namespace avocado */

