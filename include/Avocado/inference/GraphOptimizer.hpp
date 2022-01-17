/*
 * GraphOptimizer.hpp
 *
 *  Created on: Feb 28, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_INFERENCE_GRAPHOPTIMIZER_HPP_
#define AVOCADO_INFERENCE_GRAPHOPTIMIZER_HPP_

#include <Avocado/graph/Graph.hpp>

#include <memory>
#include <string>

namespace avocado
{
	namespace inference
	{
		class GraphOptimizer
		{
			public:
				virtual ~GraphOptimizer() = default;

				virtual std::string name() const = 0;
				virtual std::unique_ptr<GraphOptimizer> clone() const = 0;

				virtual bool optimize(Graph &graph) const = 0;
		};

		void registerGraphOptimizer(const GraphOptimizer &graphOptimizer);
	} /* namespace inference */
} /* namespace avocado */

#endif /* AVOCADO_INFERENCE_GRAPHOPTIMIZER_HPP_ */
